"""Diagnostic finite-pixel Fresnel reference. Does not replace mandatory 128-point single FFT.
Assumes piecewise constant full-fill square cells centered on integer grid.
Cell quadrature (Gauss 8/16) is validated against independent mpmath C/S antiderivatives.
"""
import argparse,sys,json,platform
p=argparse.ArgumentParser();p.add_argument("--deps");a=p.parse_args()
if a.deps:sys.path.insert(0,a.deps)
import numpy as np
import mpmath as mp
mp.mp.dps=40
N=128;d=10e-6;lam=532e-9;zs=[.0241,.0253,.0265,.0277]
u=(np.arange(N)-N//2)*d
Y,X=np.meshgrid(u,u,indexing="ij")
fields={
 "uniform_lens_f25.3":np.exp(-1j*np.pi*(X*X+Y*Y)/(lam*.0253)),
 "soft_amplitude_lens_sigma150um":np.exp(-(X*X+Y*Y)/(2*(150e-6)**2))*np.exp(-1j*np.pi*(X*X+Y*Y)/(lam*.0253)),
 "uniform_tilt15_per_mm":np.exp(2j*np.pi*15000*X),
 "uniform_tilt35_per_mm":np.exp(2j*np.pi*35000*X)}
def rel(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))
def cell_kernel(x,z,q):
    nodes,weights=np.polynomial.legendre.leggauss(q)
    src=u[None,:,None]+d/2*nodes[None,None,:]
    return np.sum(np.exp(1j*np.pi*(x[:,None,None]-src)**2/(lam*z))*weights[None,None,:],axis=-1)*d/2
def fft(field,z,x):
    ox,oy=np.meshgrid(x,x,indexing="xy")
    q=field*np.exp(1j*np.pi*(X*X+Y*Y)/(lam*z))
    return np.exp(2j*np.pi*z/lam)/(1j*lam*z)*np.exp(1j*np.pi*(ox*ox+oy*oy)/(lam*z))*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(q)))*d*d
rows=[];check=[]
for z in zs:
    dxo=lam*z/(N*d);x=(np.arange(N)-N//2)*dxo
    k8=cell_kernel(x,z,8);k16=cell_kernel(x,z,16)
    for oi,si in [(64,64),(0,127),(127,0),(84,93)]:
        s=np.sqrt(2/(lam*z))
        low=mp.mpf(float((u[si]-d/2-x[oi])*s));high=mp.mpf(float((u[si]+d/2-x[oi])*s))
        exact=complex((mp.fresnelc(high)-mp.fresnelc(low))+1j*(mp.fresnels(high)-mp.fresnels(low)))*np.sqrt(lam*z/2)
        err=abs(k16[oi,si]-exact)/d
        assert err<1e-10
        check.append({"z":z,"out_index":oi,"source_index":si,"abs_kernel_error_over_cell_width":float(err)})
    pref=np.exp(2j*np.pi*z/lam)/(1j*lam*z)
    roi=(abs(x)<=320e-6);ix=np.ix_(roi,roi)
    for name,field in fields.items():
        cell8=pref*(k8@field@k8.T);cell16=pref*(k16@field@k16.T)
        reference_change=rel(cell8,cell16);assert reference_change<1e-8
        out=fft(field,z,x)
        input_energy=np.sum(abs(field)**2)*d*d
        discrete_power=np.sum(abs(out)**2)*dxo*dxo
        rows.append({"field":name,"z_m":z,"gauss8_vs_16_complex_relative_l2":reference_change,
            "fft_vs_cell_complex_relative_l2_full":rel(out,cell16),
            "fft_vs_cell_intensity_relative_l2_full":rel(abs(out)**2,abs(cell16)**2),
            "fft_vs_cell_intensity_relative_l2_central_roi":rel(abs(out[ix])**2,abs(cell16[ix])**2),
            "fft_vs_cell_intensity_relative_l2_central64":rel(abs(out[32:96,32:96])**2,abs(cell16[32:96,32:96])**2),
            "mandatory_fft_area_power_ratio":float(discrete_power/input_energy)})
print(json.dumps({"scope":"Full-fill 10um piecewise-constant pixel model, 128 cells per axis, exact output grid; reference converged Gauss quadrature with C/S spot checks. Different physical interpolation than smooth-aperture reference. Not SLM calibration, RS, production code or proof all DOE accurate.",
"environment":{"python":platform.python_version(),"numpy":np.__version__,"mpmath":mp.__version__,"mp_dps":mp.mp.dps},
"spot_checks":check,"cases":rows},indent=2))
