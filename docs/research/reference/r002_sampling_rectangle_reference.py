"""Independent continuous Fresnel rectangle reference; official FFT samples vs exact C/S integrals.
Diagnostic only: not a replacement forward, not SLM calibration or full arbitrary DOE validation.
Dependencies: numpy, mpmath==1.3.0. Optional --deps inserts a local isolated package directory.
"""
import argparse,sys,json,platform
p=argparse.ArgumentParser();p.add_argument("--deps");args=p.parse_args()
if args.deps:sys.path.insert(0,args.deps)
import numpy as np
import mpmath as mp
mp.mp.dps=40
N=128;d=10e-6;lam=532e-9
zs=[.0241,.0253,.0265,.0277]
centers=(np.arange(N)-N//2)*d
# Uniform 64-cell-width aperture; integer-centered samples mean its continuous cell edges
# are (-32.5*d,31.5*d), NOT (-32*d,32*d).
a,b=-32.5*d,31.5*d
active=np.arange(32,96)
def integral_cs(x,z):
    scale=np.sqrt(lam*z/2)
    ts=np.sqrt(2/(lam*z))
    out=[]
    for value in np.asarray(x):
        lo=mp.mpf(float((a-value)*ts));hi=mp.mpf(float((b-value)*ts))
        h=(mp.fresnelc(hi)-mp.fresnelc(lo))+1j*(mp.fresnels(hi)-mp.fresnels(lo))
        out.append(complex(h)*scale)
    return np.asarray(out)
def analytic_q(x,z,f):
    shift=lam*z*f
    return np.exp(1j*2*np.pi*f*x-1j*np.pi*lam*z*f*f)*integral_cs(x-shift,z)
def midpoint_q(x,z,f,sub):
    step=d/sub
    u=a+(np.arange(64*sub)+.5)*step
    return np.exp(1j*np.pi*(x[:,None]-u[None,:])**2/(lam*z))@np.exp(2j*np.pi*f*u)*step
def fft_field(z,f):
    u=np.zeros((N,N),dtype=complex)
    mask=np.zeros(N);mask[active]=1
    u=mask[:,None]*mask[None,:]*np.exp(2j*np.pi*f*centers[None,:])
    X,Y=np.meshgrid(centers,centers,indexing="xy")
    q=u*np.exp(1j*np.pi*(X*X+Y*Y)/(lam*z))
    dxo=lam*z/(N*d);coords=(np.arange(N)-N//2)*dxo
    OX,OY=np.meshgrid(coords,coords,indexing="xy")
    pref=np.exp(2j*np.pi*z/lam)/(1j*lam*z)
    field=pref*np.exp(1j*np.pi*(OX*OX+OY*OY)/(lam*z))*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(q)))*d*d
    return coords,field
def error(v,r):
    return float(np.linalg.norm(v-r)/np.linalg.norm(r))
rows=[]
for z in zs:
    for f in [0.,.45/d]:
        x,fft=fft_field(z,f)
        qy=analytic_q(x,z,0.);qx=analytic_q(x,z,f)
        pref=np.exp(2j*np.pi*z/lam)/(1j*lam*z)
        exact=pref*qy[:,None]*qx[None,:]
        refined=[]
        for sub in [1,2,4,8,16,32]:
            mx=midpoint_q(x,z,f,sub);my=midpoint_q(x,z,0.,sub)
            approx=pref*my[:,None]*mx[None,:]
            refined.append({"subdivision":sub,"complex_relative_l2":error(approx,exact),"intensity_relative_l2":error(abs(approx)**2,abs(exact)**2)})
            if sub==1:
                discrete_error=error(fft,approx)
                assert discrete_error<1e-10
        assert refined[-1]["complex_relative_l2"]<refined[0]["complex_relative_l2"]
        rows.append({"z_m":z,"tilt_cycles_per_m":f,"max_input_chirp_frequency_ratio_for_full_128":N*d*d/(lam*z),
                     "fft_vs_same_midpoint_sum_relative_l2":discrete_error,
                     "refinement":refined})
print(json.dumps({"scope":"Continuous uniform rectangular aperture; optional smooth phase tilt. Exact Fresnel C/S antiderivative, fixed official output coordinates, no scale fitting, no physical pixel-cell calibration, no Rayleigh-Sommerfeld validation.",
"environment":{"python":platform.python_version(),"numpy":np.__version__,"mpmath":mp.__version__,"mp_dps":mp.mp.dps},
"geometry":{"N":N,"lambda_m":lam,"input_pitch_m":d,"aperture_edges_m":[a,b],"active_sample_indices":[32,95]},"cases":rows},indent=2))
