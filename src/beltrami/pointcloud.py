"""Patch-space Beltrami point-cloud filtering with noisy-data PCA frames.

The local height fit and update follow the 2012 point-cloud formulation.
Historical figure reproduction also requires the original noise and parameters.
"""
from functools import lru_cache
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy import sparse
from scipy.ndimage import convolve
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

def patch_step(z, beta=1., alpha=.21, radius=3):
    """PB3D/patchBeltrami.m exactly (beta is code's inverse metric scale)."""
    dx = z - z[np.r_[0,np.arange(z.shape[0]-1)],:]
    dy = z - z[:,np.r_[0,np.arange(z.shape[1]-1)]]
    h = np.ones((2*radius+1,)*2)
    g11 = beta + convolve(dx*dx,h,mode='constant')
    g12 = convolve(dx*dy,h,mode='constant')
    g22 = beta + convolve(dy*dy,h,mode='constant')
    q = (g11*g22-g12*g12)**-.5
    fx = q*(g22*dx-g12*dy)
    fy = q*(-g12*dx+g11*dy)
    div = fx[np.r_[np.arange(1,z.shape[0]),z.shape[0]-1],:]-fx
    div += fy[:,np.r_[np.arange(1,z.shape[1]),z.shape[1]-1]]-fy
    return alpha*beta*q*div

@lru_cache(None)
def regularizer(ny,nx):
    # Default gridfit 'gradient', uniform nodes with autoscale on.
    ind=np.arange(ny*nx).reshape((ny,nx),order='F')
    a=ind[1:-1,:].ravel(order='F')
    b=ind[:,1:-1].ravel(order='F')
    y=sparse.coo_matrix((np.tile([-1.,2.,-1.],len(a)),
        (np.repeat(a,3),np.column_stack((a-1,a,a+1)).ravel())),shape=(ny*nx,)*2)
    x=sparse.coo_matrix((np.tile([-1.,2.,-1.],len(b)),
        (np.repeat(b,3),np.column_stack((b-ny,b,b+ny)).ravel())),shape=(ny*nx,)*2)
    r=sparse.vstack([y,x],format='csc')
    return (r.T@r).tocsc(), abs(r).sum(axis=0).max()

def gridfit(points, vec, smoothness=1.):
    """Default uniform-grid gridfit.m triangle interpolation, gradient regularizer.

    This is a regularized least-squares fit, not scipy.griddata interpolation.
    """
    x,y,z=points.T
    nx=ny=len(vec); g=nx*ny
    xv=vec.copy(); yv=vec.copy()
    # Match original's extend='warning' behavior, including original cell widths.
    d=np.diff(vec)
    xv[0]=min(xv[0],x.min()); xv[-1]=max(xv[-1],x.max())
    yv[0]=min(yv[0],y.min()); yv[-1]=max(yv[-1],y.max())
    ix=np.clip(np.searchsorted(xv,x,side='right')-1,0,nx-2)
    iy=np.clip(np.searchsorted(yv,y,side='right')-1,0,ny-2)
    ind=iy+ny*ix
    tx=np.clip((x-xv[ix])/d[ix],0,1)
    ty=np.clip((y-yv[iy])/d[iy],0,1)
    lo=np.minimum(tx,ty); hi=np.maximum(tx,ty)
    cols=np.column_stack([ind,ind+ny+1,ind+np.where(tx>ty,ny,1)])
    vals=np.column_stack([1-hi,lo,hi-lo])
    a=sparse.coo_matrix((vals.ravel(),(np.repeat(np.arange(len(x)),3),cols.ravel())),shape=(len(x),g)).tocsc()
    reg,nr=regularizer(ny,nx)
    na=abs(a).sum(axis=0).max()
    lhs=(a.T@a + (smoothness*na/nr)**2 * reg).tocsc()
    zz=spsolve(lhs,a.T@z).reshape((ny,nx),order='F')
    return zz,xv,yv

def denoise_points(v, noise, iterations=10, neighbors=24, grid=21, radius=3,
            beta=1.,alpha=.21,multiplier=3., mode='historical2012', callback=None):
    """Denoise an N×3 point array using PCA frames and simultaneous updates.

    The noise standard deviation uses the same units as the input coordinates.
    Returns the final point array and fixed-neighborhood metadata.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError("points must be a finite (N, 3) array with N >= 4")
    if noise < 0 or not np.isfinite(noise):
        raise ValueError("noise standard deviation must be finite and nonnegative")
    if not 4 <= neighbors <= len(v):
        raise ValueError("neighbors must be between 4 and the number of points")
    if iterations < 0 or grid < 3 or grid % 2 != 1 or radius < 0:
        raise ValueError("require iterations >= 0, odd grid >= 3, radius >= 0")
    if beta <= 0 or alpha <= 0 or not np.isfinite([beta, alpha, multiplier]).all():
        raise ValueError("beta and alpha must be finite and positive")
    if mode not in ("historical2012", "demo_new"):
        raise ValueError("unknown point-cloud flow mode")
    if mode == "demo_new" and noise == 0:
        raise ValueError("demo_new requires a positive noise standard deviation")
    v=v.copy(); tree=cKDTree(v)
    d,idx=tree.query(v,k=neighbors,workers=1)
    r=float(np.quantile(d[:,-1],.9))
    if r <= 0:
        raise ValueError("neighborhood radius is zero; remove repeated points or increase neighbors")
    R=max(5*noise,r)
    if mode=='historical2012':
        count=min(len(v),int(np.ceil(neighbors*(R/r)**2)))
        d,idx=tree.query(v,k=count,workers=1)
    else:
        rs=d[:,3]
        if np.any(rs <= 0):
            raise ValueError("demo_new requires distinct local neighbors; remove repeated points")
    vec=np.linspace(-R,R,grid)
    for it in range(iterations):
        start=time.perf_counter(); update=np.zeros_like(v)
        for i in range(len(v)):
            patch=v[idx[i]]-v[i]
            _,_,vh=np.linalg.svd(patch-patch.mean(axis=0),full_matrices=False)
            frame=vh.T; patch=patch@frame
            if mode=='historical2012':
                patch=patch[np.sum(patch[:,:2]**2,axis=1)<R*R]
                z,_,_=gridfit(patch,vec)
                delta=patch_step(z,beta,alpha,radius)[grid//2,grid//2]
                update[i]=multiplier*delta*frame[:,2]
            else:
                Rlocal=rs[i]*4
                vec=np.linspace(-Rlocal,Rlocal,grid)
                z,_,_=gridfit(patch,vec)
                bi=np.exp(1.65*np.log(noise/rs[i]*.09)+1.95)
                # DEMO_New updates with z+delta at center, including local fit offset.
                delta=(z+patch_step(z+2*noise/rs[i]*.09,bi,.21*.8,radius))[grid//2,grid//2]
                update[i]=delta*frame[:,2]
        v+=update
        record={'iteration':it+1,'seconds':time.perf_counter()-start,
          'max_update':float(np.linalg.norm(update,axis=1).max()),'finite':bool(np.isfinite(v).all())}
        if callback: callback(it+1,v,record)
    return v,{'neighbors':idx.shape[1], 'radius':R, 'initial_neighbor_radius_q90':r}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="finite (N, 3) .npy point array")
    parser.add_argument("output", type=Path, help="filtered .npy point array")
    parser.add_argument("--noise-std", type=float, required=True, help="noise sigma in coordinate units")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--grid", type=int, default=21)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--alpha", type=float, default=.21)
    parser.add_argument("--multiplier", type=float, default=3.)
    parser.add_argument("--mode", choices=["historical2012", "demo_new"], default="historical2012")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npy":
        parser.error("output must end in .npy")
    points = np.load(args.input, allow_pickle=False)
    started = time.perf_counter()
    history = []
    def progress(iteration, current, record):
        if not record["finite"]:
            raise FloatingPointError(f"non-finite coordinates at iteration {iteration}")
        history.append(record)
        print(json.dumps(record), flush=True)
    try:
        filtered, metadata = denoise_points(points, args.noise_std, args.iterations,
            args.neighbors, args.grid, args.radius, args.beta, args.alpha,
            args.multiplier, args.mode, progress)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, filtered)
    report = dict(status="complete", points=len(filtered), finite=bool(np.isfinite(filtered).all()),
        iterations=args.iterations, noise_std=args.noise_std, mode=args.mode,
        normals="estimated from noisy local neighborhoods by PCA", grid=args.grid,
        patch_radius=args.radius, beta=args.beta, alpha=args.alpha, multiplier=args.multiplier,
        seconds=time.perf_counter()-started, neighborhood=metadata, history=history)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
