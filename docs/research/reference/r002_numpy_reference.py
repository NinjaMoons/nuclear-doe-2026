import sys, json, platform
import numpy as np
lam=532e-9
zs=[.0241,.0253,.0265,.0277]
def coords(n,d):
    return (np.arange(n)-n//2)*d
def propagate_fft(u,dy,dx,z):
    ny,nx=u.shape
    yo,xo=np.meshgrid(coords(ny,lam*z/(ny*dy)),coords(nx,lam*z/(nx*dx)),indexing="ij")
    yi,xi=np.meshgrid(coords(ny,dy),coords(nx,dx),indexing="ij")
    pref=np.exp(2j*np.pi*z/lam)/(1j*lam*z)
    a=u*np.exp(1j*np.pi*(xi*xi+yi*yi)/(lam*z))
    return pref*np.exp(1j*np.pi*(xo*xo+yo*yo)/(lam*z))*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))*dx*dy
def matrix(ny,nx,dy,dx,z):
    yi,xi=np.meshgrid(coords(ny,dy),coords(nx,dx),indexing="ij")
    yo,xo=np.meshgrid(coords(ny,lam*z/(ny*dy)),coords(nx,lam*z/(nx*dx)),indexing="ij")
    d2=(yo.ravel()[:,None]-yi.ravel()[None,:])**2+(xo.ravel()[:,None]-xi.ravel()[None,:])**2
    return np.exp(2j*np.pi*z/lam)/(1j*lam*z)*np.exp(1j*np.pi*d2/(lam*z))*dx*dy
rng=np.random.default_rng(42)
cases=[]
for ny,nx in [(4,6),(5,7)]:
    dy,dx=11e-6,9e-6
    u=rng.normal(size=(ny,nx))+1j*rng.normal(size=(ny,nx))
    for z in zs:
        out=propagate_fft(u,dy,dx,z)
        ref=(matrix(ny,nx,dy,dx,z)@u.ravel()).reshape(ny,nx)
        rel=np.linalg.norm(out-ref)/np.linalg.norm(ref)
        ratio=np.sum(abs(out)**2)*(lam*z/(ny*dy))*(lam*z/(nx*dx))/(np.sum(abs(u)**2)*dx*dy)
        assert rel<1e-11 and abs(ratio-1)<1e-12
        cases.append({"shape":[ny,nx],"z_m":z,"complex_relative_l2":float(rel),"power_ratio":float(ratio)})
ny,nx=4,6;dy,dx=11e-6,9e-6
phi=rng.normal(size=(ny,nx));u=np.exp(1j*phi).ravel()
targets=rng.random((4,ny*nx))
mats=[matrix(ny,nx,dy,dx,z) for z in zs]
def loss(p):
    uu=np.exp(1j*p).ravel()
    return sum(np.mean((abs(a@uu)**2-t)**2) for a,t in zip(mats,targets))/4
grad=np.zeros(ny*nx)
for a,t in zip(mats,targets):
    v=a@u
    dldi=2*(abs(v)**2-t)/(4*ny*nx)
    grad+=2*np.real(np.sum(dldi[:,None]*np.conj(v)[:,None]*(1j*a*u[None,:]),axis=0))
checks=[]
for j in [0,5,13,23]:
    p1=phi.copy().ravel();p2=p1.copy();p1[j]+=1e-5;p2[j]-=1e-5
    fd=(loss(p1.reshape(ny,nx))-loss(p2.reshape(ny,nx)))/(2e-5)
    err=abs(fd-grad[j])
    assert err<1e-8
    checks.append({"pixel":j,"direct_analytic":float(grad[j]),"central_difference":float(fd),"absolute_error":float(err)})
report={"scope":"Coordinator NumPy reference ONLY; not repository code, not PyTorch autograd, not Gemini output, not continuous diffraction validation",
"python":platform.python_version(),"numpy":np.__version__,"seed":42,"tests":cases,"shared_phase_gradient_checks":checks,
"output_pitch_um":[lam*z/(128*10e-6)*1e6 for z in zs],
"field_of_view_mm":[lam*z/10e-6*1e3 for z in zs],
"input_chirp_heuristic_threshold_mm":128*(10e-6)**2/lam*1e3}
print(json.dumps(report,indent=2))
