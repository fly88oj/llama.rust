#!/usr/bin/env python3
"""mk_q5k_desc.py — build a descriptor.bin for parity/ref_q5k_dump (agent W).

Reads a *real* GGUF tensor (parity/ref_q5k_dump computes the reference's
production mul_mat on it) and writes

    u32 type_id | u32 n_per_row | u32 n_rows | u32 n_act_rows |
    f32 act[n_act_rows * n_per_row] | weight bytes

Activations are LCG rows with the same generator as parity/ref_mulmat_dump.c
(seed 0x1234abcd) scaled by the amplitudes you pass.

usage: mk_q5k_desc.py <gguf> <tensor_name> <ty> <all|slice_rows> <out.bin> <scale[,..]>

The artifacts committed in parity/ were produced with:
  granite-4.0-h-tiny-Q4_K_M.gguf  blk.0.ffn_gate_shexp.weight  13 all  .75,3,.75,.75,.75 -> q5k_real_ref.bin
  granite-4.0-h-tiny-Q4_K_M.gguf  blk.0.ffn_down_shexp.weight  14 8    .75,3             -> q6k_granite_ref.bin
  granite-4.0-h-tiny-Q4_K_M.gguf  blk.0.ssm_out.weight         12 8    .75,3,.75         -> q4k_granite_ref.bin
  gpt-oss-20b-Q4_K_M.gguf         blk.0.attn_output.weight     12 8    .75,3,.75         -> q4k_goss_ref.bin
  gpt-oss-20b-Q4_K_M.gguf         blk.0.attn_q.weight           6 8    .75,3,.75,.75,.75 -> q5_0_goss_ref.bin
  gpt-oss-20b-Q4_K_M.gguf         blk.0.attn_v.weight           8 8    .75,3,.75,.75,.75 -> q8_0_goss_ref.bin

then re-run: ./parity/ref_q5k_dump <out.bin> parity/<artifact>.bin
"""
import struct, sys, os
def rd(path):
    f=open(path,'rb'); assert f.read(4)==b'GGUF'
    ver=struct.unpack('<I',f.read(4))[0]; n_t=struct.unpack('<Q',f.read(8))[0]; n_kv=struct.unpack('<Q',f.read(8))[0]
    def rs():
        n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode('utf-8','replace')
    def rv(t):
        F={0:('<B',1),1:('<b',1),2:('<H',2),3:('<h',2),4:('<I',4),5:('<i',4),6:('<f',4),7:('<?',1),10:('<Q',8),11:('<q',8),12:('<d',8)}
        if t==8: return rs()
        if t==9:
            et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
            return [rv(et) for _ in range(n)]
        fmt,sz=F[t]; return struct.unpack(fmt,f.read(sz))[0]
    kv={}
    for _ in range(n_kv):
        k=rs(); t=struct.unpack('<I',f.read(4))[0]; kv[k]=rv(t)
    tens={}
    for _ in range(n_t):
        name=rs(); nd=struct.unpack('<I',f.read(4))[0]
        ne=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
        t=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
        tens[name]=(t,ne,off)
    align=kv.get('general.alignment',32)
    data_off=(f.tell()+align-1)//align*align
    return f, tens, data_off, kv
# ggml type sizes as (block_elems, block_bytes)
TS={0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),30:(1,2),39:(32,17),20:(32,18),23:(256,136),21:(256,110),22:(256,82),16:(256,66),17:(256,74),18:(256,98),19:(256,50),29:(256,56),24:(1,1),25:(1,2),26:(1,4)}

def row_size(t,n):
    be,bb=TS[t]
    assert n%be==0, (t,n)
    return n//be*bb

def make_descriptor(raw, ty, n, nrows, out, acts, seed=0x1234abcd):
    data=open(raw,'rb').read()
    f=open(out,'wb'); f.write(struct.pack('<IIII',ty,n,nrows,len(acts)))
    st=seed
    for a in acts:
        for _ in range(n):
            st=(st*1664525+1013904223)&0xffffffff
            v=struct.unpack('<i',struct.pack('<I',st))[0]/float(1<<28)*a
            f.write(struct.pack('<f',v))
    f.write(data); f.close()

def main(argv):
    gguf, name, ty, slice_rows, out = argv[0], argv[1], int(argv[2]), argv[3], argv[4]
    acts=[float(x) for x in argv[5].split(',')]
    f, tens, data_off, _kv = rd(gguf)
    t, ne, off = tens[name]
    assert t == ty, f"{name}: gguf type {t} != requested {ty}"
    n = ne[0]; rows = 1
    for d in ne[1:]: rows *= d
    if slice_rows != 'all': rows = min(rows, int(slice_rows))
    nbytes = row_size(t,n)*rows
    f.seek(data_off + off); raw = f.read(nbytes)
    tmp = '/tmp/mk_q5k_desc_raw.bin'
    open(tmp,'wb').write(raw)
    make_descriptor(tmp, ty, n, rows, out, acts)
    print(f'wrote {out}: {name} ty={ty} ne={ne} n={n} nrows={rows} nact={len(acts)} wbytes={nbytes}')

if __name__=='__main__':
    main(sys.argv[1:])
