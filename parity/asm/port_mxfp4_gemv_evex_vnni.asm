
/home/jeffrey/works/personal/github/fly88oj/llama.rust/target/release/deps/ggml-c87c5157cc8a9cd3:     file format elf64-x86-64


Disassembly of section .text:

000000000019f9f0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8614gemv_evex_vnni>:
  19f9f0:	49 c1 e8 03          	shr    $0x3,%r8
  19f9f4:	0f 84 7a 02 00 00    	je     19fc74 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8614gemv_evex_vnni+0x284>
  19f9fa:	48 c1 ef 05          	shr    $0x5,%rdi
  19f9fe:	0f 84 74 02 00 00    	je     19fc78 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8614gemv_evex_vnni+0x288>
  19fa04:	48 89 f8             	mov    %rdi,%rax
  19fa07:	48 c1 e0 07          	shl    $0x7,%rax
  19fa0b:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  19fa0f:	45 31 c9             	xor    %r9d,%r9d
  19fa12:	c5 fd 6f 05 06 50 e9 	vmovdqa -0x16affa(%rip),%ymm0        # 34a20 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x6f0>
  19fa19:	ff 
  19fa1a:	c5 fd 6f 0d 5e 58 e9 	vmovdqa -0x16a7a2(%rip),%ymm1        # 35280 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  19fa21:	ff 
  19fa22:	62 e1 fd 08 6f 05 64 	vmovdqa64 -0x16b69c(%rip),%xmm16        # 34390 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x60>
  19fa29:	49 e9 ff 
  19fa2c:	62 e2 7d 28 58 0d da 	vpbroadcastd -0x16a426(%rip),%ymm17        # 35610 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x3d0>
  19fa33:	5b e9 ff 
  19fa36:	c5 d9 76 e4          	vpcmpeqd %xmm4,%xmm4,%xmm4
  19fa3a:	c5 fd 6f 2d 5e 54 e9 	vmovdqa -0x16aba2(%rip),%ymm5        # 34ea0 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0xb70>
  19fa41:	ff 
  19fa42:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  19fa49:	1f 84 00 00 00 00 00 
  19fa50:	49 89 fa             	mov    %rdi,%r10
  19fa53:	45 31 db             	xor    %r11d,%r11d
  19fa56:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
  19fa5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  19fa60:	c4 a1 7e 6f 7c 9a 08 	vmovdqu 0x8(%rdx,%r11,4),%ymm7
  19fa67:	c4 21 7e 6f 44 9a 28 	vmovdqu 0x28(%rdx,%r11,4),%ymm8
  19fa6e:	c4 21 7e 6f 4c 9a 48 	vmovdqu 0x48(%rdx,%r11,4),%ymm9
  19fa75:	c4 21 7e 6f 74 9a 68 	vmovdqu 0x68(%rdx,%r11,4),%ymm14
  19fa7c:	c5 45 db d0          	vpand  %ymm0,%ymm7,%ymm10
  19fa80:	c5 3d db d8          	vpand  %ymm0,%ymm8,%ymm11
  19fa84:	c4 42 75 00 fa       	vpshufb %ymm10,%ymm1,%ymm15
  19fa89:	c4 c2 75 00 d3       	vpshufb %ymm11,%ymm1,%ymm2
  19fa8e:	c5 35 db d0          	vpand  %ymm0,%ymm9,%ymm10
  19fa92:	c5 0d db d8          	vpand  %ymm0,%ymm14,%ymm11
  19fa96:	c4 42 75 00 ea       	vpshufb %ymm10,%ymm1,%ymm13
  19fa9b:	c4 42 75 00 e3       	vpshufb %ymm11,%ymm1,%ymm12
  19faa0:	c5 c5 71 d7 04       	vpsrlw $0x4,%ymm7,%ymm7
  19faa5:	c5 c5 db f8          	vpand  %ymm0,%ymm7,%ymm7
  19faa9:	c4 62 75 00 d7       	vpshufb %ymm7,%ymm1,%ymm10
  19faae:	c4 c1 45 71 d0 04    	vpsrlw $0x4,%ymm8,%ymm7
  19fab4:	c5 c5 db f8          	vpand  %ymm0,%ymm7,%ymm7
  19fab8:	c4 c1 3d 71 d1 04    	vpsrlw $0x4,%ymm9,%ymm8
  19fabe:	c4 62 75 00 df       	vpshufb %ymm7,%ymm1,%ymm11
  19fac3:	c5 bd db f8          	vpand  %ymm0,%ymm8,%ymm7
  19fac7:	c4 c1 3d 71 d6 04    	vpsrlw $0x4,%ymm14,%ymm8
  19facd:	c5 3d db c0          	vpand  %ymm0,%ymm8,%ymm8
  19fad1:	c4 62 75 00 cf       	vpshufb %ymm7,%ymm1,%ymm9
  19fad6:	c4 42 75 00 c0       	vpshufb %ymm8,%ymm1,%ymm8
  19fadb:	c4 a2 7d 58 7c 19 02 	vpbroadcastd 0x2(%rcx,%r11,1),%ymm7
  19fae2:	c5 7d 70 f2 a0       	vpshufd $0xa0,%ymm2,%ymm14
  19fae7:	c4 43 05 02 f6 aa    	vpblendd $0xaa,%ymm14,%ymm15,%ymm14
  19faed:	c4 c2 45 08 de       	vpsignb %ymm14,%ymm7,%ymm3
  19faf2:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  19faf6:	c4 42 0d 08 f6       	vpsignb %ymm14,%ymm14,%ymm14
  19fafb:	c4 41 7d 70 ff f5    	vpshufd $0xf5,%ymm15,%ymm15
  19fb01:	c4 e3 05 02 d2 aa    	vpblendd $0xaa,%ymm2,%ymm15,%ymm2
  19fb07:	c4 62 6d 08 fa       	vpsignb %ymm2,%ymm2,%ymm15
  19fb0c:	62 f2 0d 28 50 fb    	vpdpbusd %ymm3,%ymm14,%ymm7
  19fb12:	c4 a2 7d 58 5c 19 06 	vpbroadcastd 0x6(%rcx,%r11,1),%ymm3
  19fb19:	c4 e2 65 08 d2       	vpsignb %ymm2,%ymm3,%ymm2
  19fb1e:	62 f2 05 28 50 fa    	vpdpbusd %ymm2,%ymm15,%ymm7
  19fb24:	c4 c1 7d 70 d4 a0    	vpshufd $0xa0,%ymm12,%ymm2
  19fb2a:	c4 a2 7d 58 5c 19 0a 	vpbroadcastd 0xa(%rcx,%r11,1),%ymm3
  19fb31:	c4 e3 15 02 d2 aa    	vpblendd $0xaa,%ymm2,%ymm13,%ymm2
  19fb37:	c4 62 6d 08 f2       	vpsignb %ymm2,%ymm2,%ymm14
  19fb3c:	c4 e2 65 08 d2       	vpsignb %ymm2,%ymm3,%ymm2
  19fb41:	c4 c1 7d 70 dd f5    	vpshufd $0xf5,%ymm13,%ymm3
  19fb47:	62 f2 0d 28 50 fa    	vpdpbusd %ymm2,%ymm14,%ymm7
  19fb4d:	c4 c3 65 02 d4 aa    	vpblendd $0xaa,%ymm12,%ymm3,%ymm2
  19fb53:	c4 a2 7d 58 5c 19 0e 	vpbroadcastd 0xe(%rcx,%r11,1),%ymm3
  19fb5a:	c4 62 6d 08 e2       	vpsignb %ymm2,%ymm2,%ymm12
  19fb5f:	c4 e2 65 08 d2       	vpsignb %ymm2,%ymm3,%ymm2
  19fb64:	c4 c1 7d 70 db a0    	vpshufd $0xa0,%ymm11,%ymm3
  19fb6a:	c4 e3 2d 02 db aa    	vpblendd $0xaa,%ymm3,%ymm10,%ymm3
  19fb70:	62 f2 1d 28 50 fa    	vpdpbusd %ymm2,%ymm12,%ymm7
  19fb76:	c4 e2 65 08 d3       	vpsignb %ymm3,%ymm3,%ymm2
  19fb7b:	c4 22 7d 58 64 19 12 	vpbroadcastd 0x12(%rcx,%r11,1),%ymm12
  19fb82:	c4 e2 1d 08 db       	vpsignb %ymm3,%ymm12,%ymm3
  19fb87:	c4 41 7d 70 d2 f5    	vpshufd $0xf5,%ymm10,%ymm10
  19fb8d:	c4 43 2d 02 d3 aa    	vpblendd $0xaa,%ymm11,%ymm10,%ymm10
  19fb93:	c4 42 2d 08 da       	vpsignb %ymm10,%ymm10,%ymm11
  19fb98:	62 f2 6d 28 50 fb    	vpdpbusd %ymm3,%ymm2,%ymm7
  19fb9e:	c4 a2 7d 58 54 19 16 	vpbroadcastd 0x16(%rcx,%r11,1),%ymm2
  19fba5:	c4 c2 6d 08 d2       	vpsignb %ymm10,%ymm2,%ymm2
  19fbaa:	62 f2 25 28 50 fa    	vpdpbusd %ymm2,%ymm11,%ymm7
  19fbb0:	c4 c1 7d 70 d0 a0    	vpshufd $0xa0,%ymm8,%ymm2
  19fbb6:	c4 a2 7d 58 5c 19 1a 	vpbroadcastd 0x1a(%rcx,%r11,1),%ymm3
  19fbbd:	c4 e3 35 02 d2 aa    	vpblendd $0xaa,%ymm2,%ymm9,%ymm2
  19fbc3:	c4 41 7d 70 c9 f5    	vpshufd $0xf5,%ymm9,%ymm9
  19fbc9:	c4 22 7d 58 54 19 1e 	vpbroadcastd 0x1e(%rcx,%r11,1),%ymm10
  19fbd0:	c4 62 6d 08 da       	vpsignb %ymm2,%ymm2,%ymm11
  19fbd5:	c4 43 35 02 c0 aa    	vpblendd $0xaa,%ymm8,%ymm9,%ymm8
  19fbdb:	c4 21 79 c4 0c 19 00 	vpinsrw $0x0,(%rcx,%r11,1),%xmm0,%xmm9
  19fbe2:	c4 e2 65 08 d2       	vpsignb %ymm2,%ymm3,%ymm2
  19fbe7:	c4 c2 3d 08 d8       	vpsignb %ymm8,%ymm8,%ymm3
  19fbec:	c4 42 79 13 c9       	vcvtph2ps %xmm9,%xmm9
  19fbf1:	62 f2 25 28 50 fa    	vpdpbusd %ymm2,%ymm11,%ymm7
  19fbf7:	c4 a1 7a 7e 14 9a    	vmovq  (%rdx,%r11,4),%xmm2
  19fbfd:	62 b3 6d 08 3e c8 01 	vpcmpltub %xmm16,%xmm2,%k1
  19fc04:	c4 62 7d 31 da       	vpmovzxbd %xmm2,%ymm11
  19fc09:	c4 42 2d 08 c0       	vpsignb %ymm8,%ymm10,%ymm8
  19fc0e:	c5 e9 fc d4          	vpaddb %xmm4,%xmm2,%xmm2
  19fc12:	c4 e2 7d 31 d2       	vpmovzxbd %xmm2,%ymm2
  19fc17:	c5 ed 72 f2 17       	vpslld $0x17,%ymm2,%ymm2
  19fc1c:	62 d2 65 28 50 f8    	vpdpbusd %ymm8,%ymm3,%ymm7
  19fc22:	62 d2 75 21 47 d3    	vpsllvd %ymm11,%ymm17,%ymm2{%k1}
  19fc28:	c4 e2 55 36 d2       	vpermd %ymm2,%ymm5,%ymm2
  19fc2d:	c4 c2 7d 18 d9       	vbroadcastss %xmm9,%ymm3
  19fc32:	c5 fc 5b ff          	vcvtdq2ps %ymm7,%ymm7
  19fc36:	c5 e4 59 d2          	vmulps %ymm2,%ymm3,%ymm2
  19fc3a:	c4 e2 45 b8 f2       	vfmadd231ps %ymm2,%ymm7,%ymm6
  19fc3f:	49 83 c3 22          	add    $0x22,%r11
  19fc43:	49 ff ca             	dec    %r10
  19fc46:	0f 85 14 fe ff ff    	jne    19fa60 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8614gemv_evex_vnni+0x70>
  19fc4c:	4d 8d 51 01          	lea    0x1(%r9),%r10
  19fc50:	c5 cc c6 d6 d8       	vshufps $0xd8,%ymm6,%ymm6,%ymm2
  19fc55:	c4 e3 fd 01 d2 d8    	vpermpd $0xd8,%ymm2,%ymm2
  19fc5b:	49 c1 e1 05          	shl    $0x5,%r9
  19fc5f:	c4 a1 7c 11 14 0e    	vmovups %ymm2,(%rsi,%r9,1)
  19fc65:	48 01 c2             	add    %rax,%rdx
  19fc68:	4d 89 d1             	mov    %r10,%r9
  19fc6b:	4d 39 c2             	cmp    %r8,%r10
  19fc6e:	0f 85 dc fd ff ff    	jne    19fa50 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8614gemv_evex_vnni+0x60>
  19fc74:	c5 f8 77             	vzeroupper
  19fc77:	c3                   	ret
  19fc78:	49 c1 e0 05          	shl    $0x5,%r8
  19fc7c:	48 89 f7             	mov    %rsi,%rdi
  19fc7f:	31 f6                	xor    %esi,%esi
  19fc81:	4c 89 c2             	mov    %r8,%rdx
  19fc84:	ff 25 16 31 18 00    	jmp    *0x183116(%rip)        # 322da0 <memset@GLIBC_2.2.5>
  19fc8a:	cc                   	int3
  19fc8b:	cc                   	int3
  19fc8c:	cc                   	int3
  19fc8d:	cc                   	int3
  19fc8e:	cc                   	int3
  19fc8f:	cc                   	int3

000000000019fc90 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex>:
  19fc90:	55                   	push   %rbp
  19fc91:	48 89 e5             	mov    %rsp,%rbp
  19fc94:	41 57                	push   %r15
  19fc96:	41 56                	push   %r14
  19fc98:	41 55                	push   %r13
  19fc9a:	41 54                	push   %r12
  19fc9c:	53                   	push   %rbx
  19fc9d:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  19fca1:	48 81 ec 40 07 00 00 	sub    $0x740,%rsp
  19fca8:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
  19fcad:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  19fcb2:	48 8b 45 10          	mov    0x10(%rbp),%rax
  19fcb6:	48 c1 ef 05          	shr    $0x5,%rdi
  19fcba:	49 c1 e9 02          	shr    $0x2,%r9
  19fcbe:	48 b9 fc ff ff ff ff 	movabs $0x3ffffffffffffffc,%rcx
  19fcc5:	ff ff 3f 
  19fcc8:	4c 21 c9             	and    %r9,%rcx
  19fccb:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
  19fcd0:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
  19fcd5:	0f 84 9d 0e 00 00    	je     1a0b78 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xee8>
  19fcdb:	48 c1 e8 03          	shr    $0x3,%rax
  19fcdf:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  19fce4:	0f 84 96 0e 00 00    	je     1a0b80 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xef0>
  19fcea:	48 85 ff             	test   %rdi,%rdi
  19fced:	0f 84 a2 0e 00 00    	je     1a0b95 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xf05>
  19fcf3:	48 89 f8             	mov    %rdi,%rax
  19fcf6:	48 c1 e0 07          	shl    $0x7,%rax
  19fcfa:	4c 8d 34 f8          	lea    (%rax,%rdi,8),%r14
  19fcfe:	49 8d 80 80 00 00 00 	lea    0x80(%r8),%rax
  19fd05:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  19fd0a:	48 69 c7 20 02 00 00 	imul   $0x220,%rdi,%rax
  19fd11:	48 89 84 24 28 01 00 	mov    %rax,0x128(%rsp)
  19fd18:	00 
  19fd19:	45 31 ed             	xor    %r13d,%r13d
  19fd1c:	c4 e2 7d 79 1d f9 87 	vpbroadcastw -0x167807(%rip),%ymm3        # 3851e <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x366>
  19fd23:	e9 ff 
  19fd25:	c4 e2 7d 18 05 e2 58 	vbroadcastss -0x16a71e(%rip),%ymm0        # 35610 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x3d0>
  19fd2c:	e9 ff 
  19fd2e:	c5 fc 29 84 24 80 05 	vmovaps %ymm0,0x580(%rsp)
  19fd35:	00 00 
  19fd37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  19fd3e:	00 00 
  19fd40:	4a 8d 04 ad 00 00 00 	lea    0x0(,%r13,4),%rax
  19fd47:	00 
  19fd48:	48 0f af c2          	imul   %rdx,%rax
  19fd4c:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  19fd53:	00 
  19fd54:	4a 8d 04 ad 01 00 00 	lea    0x1(,%r13,4),%rax
  19fd5b:	00 
  19fd5c:	48 0f af c2          	imul   %rdx,%rax
  19fd60:	48 89 84 24 90 01 00 	mov    %rax,0x190(%rsp)
  19fd67:	00 
  19fd68:	4a 8d 04 ad 02 00 00 	lea    0x2(,%r13,4),%rax
  19fd6f:	00 
  19fd70:	48 0f af c2          	imul   %rdx,%rax
  19fd74:	48 89 84 24 88 01 00 	mov    %rax,0x188(%rsp)
  19fd7b:	00 
  19fd7c:	4a 8d 04 ad 03 00 00 	lea    0x3(,%r13,4),%rax
  19fd83:	00 
  19fd84:	48 0f af c2          	imul   %rdx,%rax
  19fd88:	48 89 84 24 80 01 00 	mov    %rax,0x180(%rsp)
  19fd8f:	00 
  19fd90:	4a 8d 04 ad 04 00 00 	lea    0x4(,%r13,4),%rax
  19fd97:	00 
  19fd98:	48 0f af c2          	imul   %rdx,%rax
  19fd9c:	48 89 84 24 78 01 00 	mov    %rax,0x178(%rsp)
  19fda3:	00 
  19fda4:	4a 8d 04 ad 05 00 00 	lea    0x5(,%r13,4),%rax
  19fdab:	00 
  19fdac:	48 0f af c2          	imul   %rdx,%rax
  19fdb0:	48 89 84 24 70 01 00 	mov    %rax,0x170(%rsp)
  19fdb7:	00 
  19fdb8:	4a 8d 04 ad 06 00 00 	lea    0x6(,%r13,4),%rax
  19fdbf:	00 
  19fdc0:	48 0f af c2          	imul   %rdx,%rax
  19fdc4:	48 89 84 24 68 01 00 	mov    %rax,0x168(%rsp)
  19fdcb:	00 
  19fdcc:	4a 8d 04 ad 07 00 00 	lea    0x7(,%r13,4),%rax
  19fdd3:	00 
  19fdd4:	48 0f af c2          	imul   %rdx,%rax
  19fdd8:	48 89 84 24 60 01 00 	mov    %rax,0x160(%rsp)
  19fddf:	00 
  19fde0:	4a 8d 04 ad 08 00 00 	lea    0x8(,%r13,4),%rax
  19fde7:	00 
  19fde8:	48 0f af c2          	imul   %rdx,%rax
  19fdec:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
  19fdf3:	00 
  19fdf4:	4a 8d 04 ad 09 00 00 	lea    0x9(,%r13,4),%rax
  19fdfb:	00 
  19fdfc:	48 0f af c2          	imul   %rdx,%rax
  19fe00:	48 89 84 24 50 01 00 	mov    %rax,0x150(%rsp)
  19fe07:	00 
  19fe08:	4a 8d 04 ad 0a 00 00 	lea    0xa(,%r13,4),%rax
  19fe0f:	00 
  19fe10:	48 0f af c2          	imul   %rdx,%rax
  19fe14:	48 89 84 24 48 01 00 	mov    %rax,0x148(%rsp)
  19fe1b:	00 
  19fe1c:	4a 8d 04 ad 0b 00 00 	lea    0xb(,%r13,4),%rax
  19fe23:	00 
  19fe24:	48 0f af c2          	imul   %rdx,%rax
  19fe28:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
  19fe2f:	00 
  19fe30:	4a 8d 04 ad 0c 00 00 	lea    0xc(,%r13,4),%rax
  19fe37:	00 
  19fe38:	48 0f af c2          	imul   %rdx,%rax
  19fe3c:	48 89 84 24 38 01 00 	mov    %rax,0x138(%rsp)
  19fe43:	00 
  19fe44:	4a 8d 04 ad 0d 00 00 	lea    0xd(,%r13,4),%rax
  19fe4b:	00 
  19fe4c:	48 0f af c2          	imul   %rdx,%rax
  19fe50:	4e 8d 14 ad 0e 00 00 	lea    0xe(,%r13,4),%r10
  19fe57:	00 
  19fe58:	4c 0f af d2          	imul   %rdx,%r10
  19fe5c:	4c 89 ac 24 30 01 00 	mov    %r13,0x130(%rsp)
  19fe63:	00 
  19fe64:	4a 8d 0c ad 0f 00 00 	lea    0xf(,%r13,4),%rcx
  19fe6b:	00 
  19fe6c:	48 0f af ca          	imul   %rdx,%rcx
  19fe70:	31 db                	xor    %ebx,%ebx
  19fe72:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  19fe79:	1f 84 00 00 00 00 00 
  19fe80:	4c 89 f6             	mov    %r14,%rsi
  19fe83:	48 0f af f3          	imul   %rbx,%rsi
  19fe87:	48 03 74 24 30       	add    0x30(%rsp),%rsi
  19fe8c:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  19fe90:	4c 8b 4c 24 38       	mov    0x38(%rsp),%r9
  19fe95:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
  19fe99:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
  19fe9d:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
  19fea1:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
  19fea6:	c4 41 30 57 c9       	vxorps %xmm9,%xmm9,%xmm9
  19feab:	c4 41 28 57 d2       	vxorps %xmm10,%xmm10,%xmm10
  19feb0:	c4 41 20 57 db       	vxorps %xmm11,%xmm11,%xmm11
  19feb5:	c4 41 18 57 e4       	vxorps %xmm12,%xmm12,%xmm12
  19feba:	c4 41 10 57 ed       	vxorps %xmm13,%xmm13,%xmm13
  19febf:	c4 41 08 57 f6       	vxorps %xmm14,%xmm14,%xmm14
  19fec4:	c4 41 00 57 ff       	vxorps %xmm15,%xmm15,%xmm15
  19fec9:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  19fecd:	c5 fc 29 84 24 80 02 	vmovaps %ymm0,0x280(%rsp)
  19fed4:	00 00 
  19fed6:	c5 fc 29 84 24 60 02 	vmovaps %ymm0,0x260(%rsp)
  19fedd:	00 00 
  19fedf:	c5 fc 29 84 24 40 02 	vmovaps %ymm0,0x240(%rsp)
  19fee6:	00 00 
  19fee8:	c5 fc 29 84 24 20 02 	vmovaps %ymm0,0x220(%rsp)
  19feef:	00 00 
  19fef1:	45 31 e4             	xor    %r12d,%r12d
  19fef4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  19fefb:	00 00 00 00 00 
  19ff00:	c5 7c 29 bc 24 a0 05 	vmovaps %ymm15,0x5a0(%rsp)
  19ff07:	00 00 
  19ff09:	c5 7c 29 b4 24 c0 05 	vmovaps %ymm14,0x5c0(%rsp)
  19ff10:	00 00 
  19ff12:	c5 7c 29 ac 24 e0 05 	vmovaps %ymm13,0x5e0(%rsp)
  19ff19:	00 00 
  19ff1b:	c5 7c 29 a4 24 00 06 	vmovaps %ymm12,0x600(%rsp)
  19ff22:	00 00 
  19ff24:	c5 7c 29 9c 24 20 06 	vmovaps %ymm11,0x620(%rsp)
  19ff2b:	00 00 
  19ff2d:	c5 7c 29 94 24 40 06 	vmovaps %ymm10,0x640(%rsp)
  19ff34:	00 00 
  19ff36:	c5 7c 29 8c 24 60 06 	vmovaps %ymm9,0x660(%rsp)
  19ff3d:	00 00 
  19ff3f:	c5 7c 29 84 24 80 06 	vmovaps %ymm8,0x680(%rsp)
  19ff46:	00 00 
  19ff48:	c5 fc 29 bc 24 a0 06 	vmovaps %ymm7,0x6a0(%rsp)
  19ff4f:	00 00 
  19ff51:	c5 fc 29 b4 24 c0 06 	vmovaps %ymm6,0x6c0(%rsp)
  19ff58:	00 00 
  19ff5a:	c5 fc 29 ac 24 e0 06 	vmovaps %ymm5,0x6e0(%rsp)
  19ff61:	00 00 
  19ff63:	c5 fc 29 8c 24 00 07 	vmovaps %ymm1,0x700(%rsp)
  19ff6a:	00 00 
  19ff6c:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  19ff70:	c5 fc 29 84 24 60 05 	vmovaps %ymm0,0x560(%rsp)
  19ff77:	00 00 
  19ff79:	c5 fc 29 84 24 40 05 	vmovaps %ymm0,0x540(%rsp)
  19ff80:	00 00 
  19ff82:	c5 fc 29 84 24 20 05 	vmovaps %ymm0,0x520(%rsp)
  19ff89:	00 00 
  19ff8b:	c5 fc 29 84 24 00 05 	vmovaps %ymm0,0x500(%rsp)
  19ff92:	00 00 
  19ff94:	c5 fc 29 84 24 e0 04 	vmovaps %ymm0,0x4e0(%rsp)
  19ff9b:	00 00 
  19ff9d:	c5 fc 29 84 24 c0 04 	vmovaps %ymm0,0x4c0(%rsp)
  19ffa4:	00 00 
  19ffa6:	c5 fc 29 84 24 a0 04 	vmovaps %ymm0,0x4a0(%rsp)
  19ffad:	00 00 
  19ffaf:	c5 fc 29 84 24 80 04 	vmovaps %ymm0,0x480(%rsp)
  19ffb6:	00 00 
  19ffb8:	c5 fc 29 84 24 60 04 	vmovaps %ymm0,0x460(%rsp)
  19ffbf:	00 00 
  19ffc1:	c5 fc 29 84 24 40 04 	vmovaps %ymm0,0x440(%rsp)
  19ffc8:	00 00 
  19ffca:	c5 fc 29 84 24 20 04 	vmovaps %ymm0,0x420(%rsp)
  19ffd1:	00 00 
  19ffd3:	c5 fc 29 84 24 00 04 	vmovaps %ymm0,0x400(%rsp)
  19ffda:	00 00 
  19ffdc:	c5 fc 29 84 24 e0 03 	vmovaps %ymm0,0x3e0(%rsp)
  19ffe3:	00 00 
  19ffe5:	c5 fc 29 84 24 c0 03 	vmovaps %ymm0,0x3c0(%rsp)
  19ffec:	00 00 
  19ffee:	c5 fc 29 84 24 a0 03 	vmovaps %ymm0,0x3a0(%rsp)
  19fff5:	00 00 
  19fff7:	c5 fc 29 84 24 80 03 	vmovaps %ymm0,0x380(%rsp)
  19fffe:	00 00 
  1a0000:	c5 fc 29 84 24 80 00 	vmovaps %ymm0,0x80(%rsp)
  1a0007:	00 00 
  1a0009:	c5 fc 29 44 24 60    	vmovaps %ymm0,0x60(%rsp)
  1a000f:	4d 89 e3             	mov    %r12,%r11
  1a0012:	49 c1 e3 07          	shl    $0x7,%r11
  1a0016:	4f 8d 2c e3          	lea    (%r11,%r12,8),%r13
  1a001a:	c4 a1 7e 6f 4c 2e 08 	vmovdqu 0x8(%rsi,%r13,1),%ymm1
  1a0021:	c4 a1 7e 6f 54 2e 28 	vmovdqu 0x28(%rsi,%r13,1),%ymm2
  1a0028:	c4 21 7e 6f 44 2e 48 	vmovdqu 0x48(%rsi,%r13,1),%ymm8
  1a002f:	c4 a1 7e 6f 44 2e 68 	vmovdqu 0x68(%rsi,%r13,1),%ymm0
  1a0036:	c5 dd 71 d1 04       	vpsrlw $0x4,%ymm1,%ymm4
  1a003b:	c4 62 7d 78 1d dc 84 	vpbroadcastb -0x167b24(%rip),%ymm11        # 38520 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x368>
  1a0042:	e9 ff 
  1a0044:	c5 a5 db e4          	vpand  %ymm4,%ymm11,%ymm4
  1a0048:	c5 7d 6f 25 30 52 e9 	vmovdqa -0x16add0(%rip),%ymm12        # 35280 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  1a004f:	ff 
  1a0050:	c4 62 1d 00 cc       	vpshufb %ymm4,%ymm12,%ymm9
  1a0055:	c5 dd 71 d2 04       	vpsrlw $0x4,%ymm2,%ymm4
  1a005a:	c5 a5 db e4          	vpand  %ymm4,%ymm11,%ymm4
  1a005e:	c4 62 1d 00 d4       	vpshufb %ymm4,%ymm12,%ymm10
  1a0063:	c4 c1 7d 70 e2 a0    	vpshufd $0xa0,%ymm10,%ymm4
  1a0069:	c4 e3 35 02 e4 aa    	vpblendd $0xaa,%ymm4,%ymm9,%ymm4
  1a006f:	c5 fd 7f 64 24 40    	vmovdqa %ymm4,0x40(%rsp)
  1a0075:	c4 e2 5d 08 e4       	vpsignb %ymm4,%ymm4,%ymm4
  1a007a:	c5 fd 7f a4 24 c0 02 	vmovdqa %ymm4,0x2c0(%rsp)
  1a0081:	00 00 
  1a0083:	c5 a5 db c9          	vpand  %ymm1,%ymm11,%ymm1
  1a0087:	c4 e2 1d 00 c9       	vpshufb %ymm1,%ymm12,%ymm1
  1a008c:	c5 a5 db d2          	vpand  %ymm2,%ymm11,%ymm2
  1a0090:	c4 e2 1d 00 d2       	vpshufb %ymm2,%ymm12,%ymm2
  1a0095:	c5 fd 70 e1 f5       	vpshufd $0xf5,%ymm1,%ymm4
  1a009a:	c4 e3 5d 02 e2 aa    	vpblendd $0xaa,%ymm2,%ymm4,%ymm4
  1a00a0:	c5 fd 7f a4 24 00 01 	vmovdqa %ymm4,0x100(%rsp)
  1a00a7:	00 00 
  1a00a9:	c4 e2 5d 08 e4       	vpsignb %ymm4,%ymm4,%ymm4
  1a00ae:	c5 fd 7f a4 24 e0 00 	vmovdqa %ymm4,0xe0(%rsp)
  1a00b5:	00 00 
  1a00b7:	c5 fd 70 d2 a0       	vpshufd $0xa0,%ymm2,%ymm2
  1a00bc:	c4 e3 75 02 ca aa    	vpblendd $0xaa,%ymm2,%ymm1,%ymm1
  1a00c2:	c5 fd 7f 8c 24 c0 00 	vmovdqa %ymm1,0xc0(%rsp)
  1a00c9:	00 00 
  1a00cb:	c4 e2 75 08 c9       	vpsignb %ymm1,%ymm1,%ymm1
  1a00d0:	c5 fd 7f 8c 24 a0 02 	vmovdqa %ymm1,0x2a0(%rsp)
  1a00d7:	00 00 
  1a00d9:	c4 c1 3d db cb       	vpand  %ymm11,%ymm8,%ymm1
  1a00de:	c4 e2 1d 00 c9       	vpshufb %ymm1,%ymm12,%ymm1
  1a00e3:	c5 a5 db d0          	vpand  %ymm0,%ymm11,%ymm2
  1a00e7:	c4 e2 1d 00 d2       	vpshufb %ymm2,%ymm12,%ymm2
  1a00ec:	c5 fd 70 e2 a0       	vpshufd $0xa0,%ymm2,%ymm4
  1a00f1:	c4 e3 75 02 e4 aa    	vpblendd $0xaa,%ymm4,%ymm1,%ymm4
  1a00f7:	c5 fd 7f a4 24 a0 00 	vmovdqa %ymm4,0xa0(%rsp)
  1a00fe:	00 00 
  1a0100:	c4 e2 5d 08 e4       	vpsignb %ymm4,%ymm4,%ymm4
  1a0105:	c5 fd 7f a4 24 e0 01 	vmovdqa %ymm4,0x1e0(%rsp)
  1a010c:	00 00 
  1a010e:	c5 fd 70 c9 f5       	vpshufd $0xf5,%ymm1,%ymm1
  1a0113:	c4 e3 75 02 ca aa    	vpblendd $0xaa,%ymm2,%ymm1,%ymm1
  1a0119:	c5 fd 7f 8c 24 00 02 	vmovdqa %ymm1,0x200(%rsp)
  1a0120:	00 00 
  1a0122:	c4 e2 75 08 c9       	vpsignb %ymm1,%ymm1,%ymm1
  1a0127:	c5 fd 7f 8c 24 c0 01 	vmovdqa %ymm1,0x1c0(%rsp)
  1a012e:	00 00 
  1a0130:	c4 c1 7d 70 c9 f5    	vpshufd $0xf5,%ymm9,%ymm1
  1a0136:	c4 c3 75 02 ca aa    	vpblendd $0xaa,%ymm10,%ymm1,%ymm1
  1a013c:	c5 fd 7f 8c 24 60 03 	vmovdqa %ymm1,0x360(%rsp)
  1a0143:	00 00 
  1a0145:	c4 e2 75 08 c9       	vpsignb %ymm1,%ymm1,%ymm1
  1a014a:	c5 fd 7f 8c 24 a0 01 	vmovdqa %ymm1,0x1a0(%rsp)
  1a0151:	00 00 
  1a0153:	c4 c1 3d 71 d0 04    	vpsrlw $0x4,%ymm8,%ymm8
  1a0159:	c4 41 3d db c3       	vpand  %ymm11,%ymm8,%ymm8
  1a015e:	c4 42 1d 00 d0       	vpshufb %ymm8,%ymm12,%ymm10
  1a0163:	c5 fd 71 d0 04       	vpsrlw $0x4,%ymm0,%ymm0
  1a0168:	c5 a5 db c0          	vpand  %ymm0,%ymm11,%ymm0
  1a016c:	c4 e2 1d 00 c0       	vpshufb %ymm0,%ymm12,%ymm0
  1a0171:	c5 7d 70 c0 a0       	vpshufd $0xa0,%ymm0,%ymm8
  1a0176:	c4 c3 2d 02 c8 aa    	vpblendd $0xaa,%ymm8,%ymm10,%ymm1
  1a017c:	c5 fd 7f 8c 24 40 03 	vmovdqa %ymm1,0x340(%rsp)
  1a0183:	00 00 
  1a0185:	c4 e2 75 08 c9       	vpsignb %ymm1,%ymm1,%ymm1
  1a018a:	c5 fd 7f 8c 24 20 03 	vmovdqa %ymm1,0x320(%rsp)
  1a0191:	00 00 
  1a0193:	c4 41 7d 70 d2 f5    	vpshufd $0xf5,%ymm10,%ymm10
  1a0199:	c4 e3 2d 02 c0 aa    	vpblendd $0xaa,%ymm0,%ymm10,%ymm0
  1a019f:	c5 fd 7f 84 24 00 03 	vmovdqa %ymm0,0x300(%rsp)
  1a01a6:	00 00 
  1a01a8:	c4 e2 7d 08 c0       	vpsignb %ymm0,%ymm0,%ymm0
  1a01ad:	c5 fd 7f 84 24 e0 02 	vmovdqa %ymm0,0x2e0(%rsp)
  1a01b4:	00 00 
  1a01b6:	4d 89 cf             	mov    %r9,%r15
  1a01b9:	45 31 db             	xor    %r11d,%r11d
  1a01bc:	0f 1f 40 00          	nopl   0x0(%rax)
  1a01c0:	c4 41 7a 7e 57 e8    	vmovq  -0x18(%r15),%xmm10
  1a01c6:	c4 41 7a 7e 6f c8    	vmovq  -0x38(%r15),%xmm13
  1a01cc:	c4 42 7d 58 e5       	vpbroadcastd %xmm13,%ymm12
  1a01d1:	c4 62 1d 08 64 24 40 	vpsignb 0x40(%rsp),%ymm12,%ymm12
  1a01d8:	c4 41 7a 7e 7f 88    	vmovq  -0x78(%r15),%xmm15
  1a01de:	c5 fd 6f b4 24 c0 02 	vmovdqa 0x2c0(%rsp),%ymm6
  1a01e5:	00 00 
  1a01e7:	c4 42 4d 04 e4       	vpmaddubsw %ymm12,%ymm6,%ymm12
  1a01ec:	c4 42 7d 58 f7       	vpbroadcastd %xmm15,%ymm14
  1a01f1:	c4 62 0d 08 b4 24 c0 	vpsignb 0xc0(%rsp),%ymm14,%ymm14
  1a01f8:	00 00 00 
  1a01fb:	c5 fd 6f bc 24 a0 02 	vmovdqa 0x2a0(%rsp),%ymm7
  1a0202:	00 00 
  1a0204:	c4 42 45 04 f6       	vpmaddubsw %ymm14,%ymm7,%ymm14
  1a0209:	c5 1d f5 e3          	vpmaddwd %ymm3,%ymm12,%ymm12
  1a020d:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a0211:	c4 41 1d fe e6       	vpaddd %ymm14,%ymm12,%ymm12
  1a0216:	c4 41 7a 7e 77 a8    	vmovq  -0x58(%r15),%xmm14
  1a021c:	c4 41 79 70 ff 55    	vpshufd $0x55,%xmm15,%xmm15
  1a0222:	c4 42 7d 58 ff       	vpbroadcastd %xmm15,%ymm15
  1a0227:	c5 7d 6f 9c 24 00 01 	vmovdqa 0x100(%rsp),%ymm11
  1a022e:	00 00 
  1a0230:	c4 42 05 08 fb       	vpsignb %ymm11,%ymm15,%ymm15
  1a0235:	c5 fd 6f 84 24 e0 00 	vmovdqa 0xe0(%rsp),%ymm0
  1a023c:	00 00 
  1a023e:	c4 42 7d 04 ff       	vpmaddubsw %ymm15,%ymm0,%ymm15
  1a0243:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a0247:	c4 41 05 fe e4       	vpaddd %ymm12,%ymm15,%ymm12
  1a024c:	c4 42 7d 58 fe       	vpbroadcastd %xmm14,%ymm15
  1a0251:	c5 7d 6f 8c 24 a0 00 	vmovdqa 0xa0(%rsp),%ymm9
  1a0258:	00 00 
  1a025a:	c4 42 05 08 f9       	vpsignb %ymm9,%ymm15,%ymm15
  1a025f:	c5 fd 6f a4 24 e0 01 	vmovdqa 0x1e0(%rsp),%ymm4
  1a0266:	00 00 
  1a0268:	c4 42 5d 04 ff       	vpmaddubsw %ymm15,%ymm4,%ymm15
  1a026d:	c4 41 79 70 f6 55    	vpshufd $0x55,%xmm14,%xmm14
  1a0273:	c4 42 7d 58 f6       	vpbroadcastd %xmm14,%ymm14
  1a0278:	c5 7d 6f 84 24 00 02 	vmovdqa 0x200(%rsp),%ymm8
  1a027f:	00 00 
  1a0281:	c4 42 0d 08 f0       	vpsignb %ymm8,%ymm14,%ymm14
  1a0286:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a028a:	c5 fd 6f 84 24 c0 01 	vmovdqa 0x1c0(%rsp),%ymm0
  1a0291:	00 00 
  1a0293:	c4 42 7d 04 f6       	vpmaddubsw %ymm14,%ymm0,%ymm14
  1a0298:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a029c:	c4 41 79 70 ed 55    	vpshufd $0x55,%xmm13,%xmm13
  1a02a2:	c4 41 05 fe f6       	vpaddd %ymm14,%ymm15,%ymm14
  1a02a7:	c4 42 7d 58 ed       	vpbroadcastd %xmm13,%ymm13
  1a02ac:	c5 fd 6f 8c 24 60 03 	vmovdqa 0x360(%rsp),%ymm1
  1a02b3:	00 00 
  1a02b5:	c4 62 15 08 e9       	vpsignb %ymm1,%ymm13,%ymm13
  1a02ba:	c4 42 7d 58 fa       	vpbroadcastd %xmm10,%ymm15
  1a02bf:	c5 fd 6f 84 24 a0 01 	vmovdqa 0x1a0(%rsp),%ymm0
  1a02c6:	00 00 
  1a02c8:	c4 42 7d 04 ed       	vpmaddubsw %ymm13,%ymm0,%ymm13
  1a02cd:	c4 62 05 08 bc 24 40 	vpsignb 0x340(%rsp),%ymm15,%ymm15
  1a02d4:	03 00 00 
  1a02d7:	c5 fd 6f 84 24 20 03 	vmovdqa 0x320(%rsp),%ymm0
  1a02de:	00 00 
  1a02e0:	c4 42 7d 04 ff       	vpmaddubsw %ymm15,%ymm0,%ymm15
  1a02e5:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a02e9:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a02ed:	c4 41 15 fe ef       	vpaddd %ymm15,%ymm13,%ymm13
  1a02f2:	c4 41 79 70 d2 55    	vpshufd $0x55,%xmm10,%xmm10
  1a02f8:	c4 42 7d 58 d2       	vpbroadcastd %xmm10,%ymm10
  1a02fd:	c4 41 1d fe e6       	vpaddd %ymm14,%ymm12,%ymm12
  1a0302:	c5 fd 6f ac 24 00 03 	vmovdqa 0x300(%rsp),%ymm5
  1a0309:	00 00 
  1a030b:	c4 62 2d 08 d5       	vpsignb %ymm5,%ymm10,%ymm10
  1a0310:	c5 fd 6f 84 24 e0 02 	vmovdqa 0x2e0(%rsp),%ymm0
  1a0317:	00 00 
  1a0319:	c4 42 7d 04 d2       	vpmaddubsw %ymm10,%ymm0,%ymm10
  1a031e:	c5 2d f5 d3          	vpmaddwd %ymm3,%ymm10,%ymm10
  1a0322:	c4 41 15 fe d2       	vpaddd %ymm10,%ymm13,%ymm10
  1a0327:	c4 41 1d fe d2       	vpaddd %ymm10,%ymm12,%ymm10
  1a032c:	c4 21 7d 7f 94 dc 80 	vmovdqa %ymm10,0x380(%rsp,%r11,8)
  1a0333:	03 00 00 
  1a0336:	c4 41 7a 7e 57 f0    	vmovq  -0x10(%r15),%xmm10
  1a033c:	c4 41 7a 7e 6f d0    	vmovq  -0x30(%r15),%xmm13
  1a0342:	c4 42 7d 58 e5       	vpbroadcastd %xmm13,%ymm12
  1a0347:	c5 fd 6f 54 24 40    	vmovdqa 0x40(%rsp),%ymm2
  1a034d:	c4 62 1d 08 e2       	vpsignb %ymm2,%ymm12,%ymm12
  1a0352:	c4 41 7a 7e 77 90    	vmovq  -0x70(%r15),%xmm14
  1a0358:	c4 42 4d 04 e4       	vpmaddubsw %ymm12,%ymm6,%ymm12
  1a035d:	c4 42 7d 58 fe       	vpbroadcastd %xmm14,%ymm15
  1a0362:	c5 fd 6f b4 24 c0 00 	vmovdqa 0xc0(%rsp),%ymm6
  1a0369:	00 00 
  1a036b:	c4 62 05 08 fe       	vpsignb %ymm6,%ymm15,%ymm15
  1a0370:	c4 42 45 04 ff       	vpmaddubsw %ymm15,%ymm7,%ymm15
  1a0375:	c5 1d f5 e3          	vpmaddwd %ymm3,%ymm12,%ymm12
  1a0379:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a037d:	c4 41 1d fe e7       	vpaddd %ymm15,%ymm12,%ymm12
  1a0382:	c4 41 7a 7e 7f b0    	vmovq  -0x50(%r15),%xmm15
  1a0388:	c4 41 79 70 f6 55    	vpshufd $0x55,%xmm14,%xmm14
  1a038e:	c4 42 7d 58 f6       	vpbroadcastd %xmm14,%ymm14
  1a0393:	c4 42 0d 08 f3       	vpsignb %ymm11,%ymm14,%ymm14
  1a0398:	c5 fd 6f 84 24 e0 00 	vmovdqa 0xe0(%rsp),%ymm0
  1a039f:	00 00 
  1a03a1:	c4 42 7d 04 f6       	vpmaddubsw %ymm14,%ymm0,%ymm14
  1a03a6:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a03aa:	c4 41 0d fe e4       	vpaddd %ymm12,%ymm14,%ymm12
  1a03af:	c4 42 7d 58 f7       	vpbroadcastd %xmm15,%ymm14
  1a03b4:	c4 42 0d 08 f1       	vpsignb %ymm9,%ymm14,%ymm14
  1a03b9:	c4 42 5d 04 f6       	vpmaddubsw %ymm14,%ymm4,%ymm14
  1a03be:	c5 7d 6f cc          	vmovdqa %ymm4,%ymm9
  1a03c2:	c4 41 79 70 ff 55    	vpshufd $0x55,%xmm15,%xmm15
  1a03c8:	c4 42 7d 58 ff       	vpbroadcastd %xmm15,%ymm15
  1a03cd:	c4 42 05 08 f8       	vpsignb %ymm8,%ymm15,%ymm15
  1a03d2:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a03d6:	c5 7d 6f 9c 24 c0 01 	vmovdqa 0x1c0(%rsp),%ymm11
  1a03dd:	00 00 
  1a03df:	c4 42 25 04 ff       	vpmaddubsw %ymm15,%ymm11,%ymm15
  1a03e4:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a03e8:	c4 41 79 70 ed 55    	vpshufd $0x55,%xmm13,%xmm13
  1a03ee:	c4 41 0d fe f7       	vpaddd %ymm15,%ymm14,%ymm14
  1a03f3:	c4 42 7d 58 ed       	vpbroadcastd %xmm13,%ymm13
  1a03f8:	c4 62 15 08 e9       	vpsignb %ymm1,%ymm13,%ymm13
  1a03fd:	c5 fd 6f f9          	vmovdqa %ymm1,%ymm7
  1a0401:	c4 42 7d 58 fa       	vpbroadcastd %xmm10,%ymm15
  1a0406:	c5 7d 6f 84 24 a0 01 	vmovdqa 0x1a0(%rsp),%ymm8
  1a040d:	00 00 
  1a040f:	c4 42 3d 04 ed       	vpmaddubsw %ymm13,%ymm8,%ymm13
  1a0414:	c5 fd 6f 8c 24 40 03 	vmovdqa 0x340(%rsp),%ymm1
  1a041b:	00 00 
  1a041d:	c4 62 05 08 f9       	vpsignb %ymm1,%ymm15,%ymm15
  1a0422:	c5 fd 6f a4 24 20 03 	vmovdqa 0x320(%rsp),%ymm4
  1a0429:	00 00 
  1a042b:	c4 42 5d 04 ff       	vpmaddubsw %ymm15,%ymm4,%ymm15
  1a0430:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a0434:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a0438:	c4 41 15 fe ef       	vpaddd %ymm15,%ymm13,%ymm13
  1a043d:	c4 41 79 70 d2 55    	vpshufd $0x55,%xmm10,%xmm10
  1a0443:	c4 42 7d 58 d2       	vpbroadcastd %xmm10,%ymm10
  1a0448:	c4 41 1d fe e6       	vpaddd %ymm14,%ymm12,%ymm12
  1a044d:	c4 62 2d 08 d5       	vpsignb %ymm5,%ymm10,%ymm10
  1a0452:	c5 fd 6f ac 24 e0 02 	vmovdqa 0x2e0(%rsp),%ymm5
  1a0459:	00 00 
  1a045b:	c4 42 55 04 d2       	vpmaddubsw %ymm10,%ymm5,%ymm10
  1a0460:	c5 2d f5 d3          	vpmaddwd %ymm3,%ymm10,%ymm10
  1a0464:	c4 41 15 fe d2       	vpaddd %ymm10,%ymm13,%ymm10
  1a0469:	c4 41 1d fe d2       	vpaddd %ymm10,%ymm12,%ymm10
  1a046e:	c4 21 7d 7f 94 dc a0 	vmovdqa %ymm10,0x3a0(%rsp,%r11,8)
  1a0475:	03 00 00 
  1a0478:	c4 41 7a 7e 57 f8    	vmovq  -0x8(%r15),%xmm10
  1a047e:	c4 41 7a 7e 6f d8    	vmovq  -0x28(%r15),%xmm13
  1a0484:	c4 42 7d 58 e5       	vpbroadcastd %xmm13,%ymm12
  1a0489:	c4 62 1d 08 e2       	vpsignb %ymm2,%ymm12,%ymm12
  1a048e:	c4 41 7a 7e 77 98    	vmovq  -0x68(%r15),%xmm14
  1a0494:	c5 fd 6f 94 24 c0 02 	vmovdqa 0x2c0(%rsp),%ymm2
  1a049b:	00 00 
  1a049d:	c4 42 6d 04 e4       	vpmaddubsw %ymm12,%ymm2,%ymm12
  1a04a2:	c4 42 7d 58 fe       	vpbroadcastd %xmm14,%ymm15
  1a04a7:	c4 62 05 08 fe       	vpsignb %ymm6,%ymm15,%ymm15
  1a04ac:	c5 fd 6f b4 24 a0 02 	vmovdqa 0x2a0(%rsp),%ymm6
  1a04b3:	00 00 
  1a04b5:	c4 42 4d 04 ff       	vpmaddubsw %ymm15,%ymm6,%ymm15
  1a04ba:	c5 1d f5 e3          	vpmaddwd %ymm3,%ymm12,%ymm12
  1a04be:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a04c2:	c4 41 1d fe e7       	vpaddd %ymm15,%ymm12,%ymm12
  1a04c7:	c4 41 7a 7e 7f b8    	vmovq  -0x48(%r15),%xmm15
  1a04cd:	c4 41 79 70 f6 55    	vpshufd $0x55,%xmm14,%xmm14
  1a04d3:	c4 42 7d 58 f6       	vpbroadcastd %xmm14,%ymm14
  1a04d8:	c4 62 0d 08 b4 24 00 	vpsignb 0x100(%rsp),%ymm14,%ymm14
  1a04df:	01 00 00 
  1a04e2:	c4 42 7d 04 f6       	vpmaddubsw %ymm14,%ymm0,%ymm14
  1a04e7:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a04eb:	c4 41 0d fe e4       	vpaddd %ymm12,%ymm14,%ymm12
  1a04f0:	c4 42 7d 58 f7       	vpbroadcastd %xmm15,%ymm14
  1a04f5:	c4 62 0d 08 b4 24 a0 	vpsignb 0xa0(%rsp),%ymm14,%ymm14
  1a04fc:	00 00 00 
  1a04ff:	c4 42 35 04 f6       	vpmaddubsw %ymm14,%ymm9,%ymm14
  1a0504:	c4 41 79 70 ff 55    	vpshufd $0x55,%xmm15,%xmm15
  1a050a:	c4 42 7d 58 ff       	vpbroadcastd %xmm15,%ymm15
  1a050f:	c4 62 05 08 bc 24 00 	vpsignb 0x200(%rsp),%ymm15,%ymm15
  1a0516:	02 00 00 
  1a0519:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a051d:	c4 42 25 04 ff       	vpmaddubsw %ymm15,%ymm11,%ymm15
  1a0522:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a0526:	c4 41 79 70 ed 55    	vpshufd $0x55,%xmm13,%xmm13
  1a052c:	c4 41 0d fe f7       	vpaddd %ymm15,%ymm14,%ymm14
  1a0531:	c4 42 7d 58 ed       	vpbroadcastd %xmm13,%ymm13
  1a0536:	c4 62 15 08 ef       	vpsignb %ymm7,%ymm13,%ymm13
  1a053b:	c4 42 7d 58 fa       	vpbroadcastd %xmm10,%ymm15
  1a0540:	c4 42 3d 04 ed       	vpmaddubsw %ymm13,%ymm8,%ymm13
  1a0545:	c4 62 05 08 f9       	vpsignb %ymm1,%ymm15,%ymm15
  1a054a:	c4 42 5d 04 ff       	vpmaddubsw %ymm15,%ymm4,%ymm15
  1a054f:	c5 05 f5 fb          	vpmaddwd %ymm3,%ymm15,%ymm15
  1a0553:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a0557:	c4 41 15 fe ef       	vpaddd %ymm15,%ymm13,%ymm13
  1a055c:	c4 41 79 70 d2 55    	vpshufd $0x55,%xmm10,%xmm10
  1a0562:	c4 42 7d 58 d2       	vpbroadcastd %xmm10,%ymm10
  1a0567:	c4 41 1d fe e6       	vpaddd %ymm14,%ymm12,%ymm12
  1a056c:	c5 7d 6f bc 24 00 03 	vmovdqa 0x300(%rsp),%ymm15
  1a0573:	00 00 
  1a0575:	c4 42 2d 08 d7       	vpsignb %ymm15,%ymm10,%ymm10
  1a057a:	c4 42 55 04 d2       	vpmaddubsw %ymm10,%ymm5,%ymm10
  1a057f:	c5 2d f5 d3          	vpmaddwd %ymm3,%ymm10,%ymm10
  1a0583:	c4 41 15 fe d2       	vpaddd %ymm10,%ymm13,%ymm10
  1a0588:	c4 41 1d fe d2       	vpaddd %ymm10,%ymm12,%ymm10
  1a058d:	c4 21 7d 7f 94 dc c0 	vmovdqa %ymm10,0x3c0(%rsp,%r11,8)
  1a0594:	03 00 00 
  1a0597:	c4 41 7a 7e 57 e0    	vmovq  -0x20(%r15),%xmm10
  1a059d:	c4 42 7d 58 e2       	vpbroadcastd %xmm10,%ymm12
  1a05a2:	c4 62 1d 08 64 24 40 	vpsignb 0x40(%rsp),%ymm12,%ymm12
  1a05a9:	c4 42 6d 04 e4       	vpmaddubsw %ymm12,%ymm2,%ymm12
  1a05ae:	c4 41 7a 7e 6f a0    	vmovq  -0x60(%r15),%xmm13
  1a05b4:	c5 1d f5 e3          	vpmaddwd %ymm3,%ymm12,%ymm12
  1a05b8:	c4 42 7d 58 f5       	vpbroadcastd %xmm13,%ymm14
  1a05bd:	c4 62 0d 08 b4 24 c0 	vpsignb 0xc0(%rsp),%ymm14,%ymm14
  1a05c4:	00 00 00 
  1a05c7:	c4 42 4d 04 f6       	vpmaddubsw %ymm14,%ymm6,%ymm14
  1a05cc:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a05d0:	c4 41 1d fe e6       	vpaddd %ymm14,%ymm12,%ymm12
  1a05d5:	c4 41 7a 7e 77 c0    	vmovq  -0x40(%r15),%xmm14
  1a05db:	c4 41 79 70 ed 55    	vpshufd $0x55,%xmm13,%xmm13
  1a05e1:	c4 42 7d 58 ed       	vpbroadcastd %xmm13,%ymm13
  1a05e6:	c4 62 15 08 ac 24 00 	vpsignb 0x100(%rsp),%ymm13,%ymm13
  1a05ed:	01 00 00 
  1a05f0:	c5 fd 6f 94 24 e0 00 	vmovdqa 0xe0(%rsp),%ymm2
  1a05f7:	00 00 
  1a05f9:	c4 42 6d 04 ed       	vpmaddubsw %ymm13,%ymm2,%ymm13
  1a05fe:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a0602:	c4 41 15 fe e4       	vpaddd %ymm12,%ymm13,%ymm12
  1a0607:	c4 42 7d 58 ee       	vpbroadcastd %xmm14,%ymm13
  1a060c:	c4 62 15 08 ac 24 a0 	vpsignb 0xa0(%rsp),%ymm13,%ymm13
  1a0613:	00 00 00 
  1a0616:	c4 41 79 70 f6 55    	vpshufd $0x55,%xmm14,%xmm14
  1a061c:	c5 fd 6f 94 24 e0 01 	vmovdqa 0x1e0(%rsp),%ymm2
  1a0623:	00 00 
  1a0625:	c4 42 6d 04 ed       	vpmaddubsw %ymm13,%ymm2,%ymm13
  1a062a:	c4 42 7d 58 f6       	vpbroadcastd %xmm14,%ymm14
  1a062f:	c4 62 0d 08 b4 24 00 	vpsignb 0x200(%rsp),%ymm14,%ymm14
  1a0636:	02 00 00 
  1a0639:	c4 42 25 04 f6       	vpmaddubsw %ymm14,%ymm11,%ymm14
  1a063e:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a0642:	c5 0d f5 f3          	vpmaddwd %ymm3,%ymm14,%ymm14
  1a0646:	c4 41 15 fe ee       	vpaddd %ymm14,%ymm13,%ymm13
  1a064b:	c4 41 7a 7e 37       	vmovq  (%r15),%xmm14
  1a0650:	c4 41 79 70 d2 55    	vpshufd $0x55,%xmm10,%xmm10
  1a0656:	c4 42 7d 58 d2       	vpbroadcastd %xmm10,%ymm10
  1a065b:	c4 62 2d 08 d7       	vpsignb %ymm7,%ymm10,%ymm10
  1a0660:	c4 42 3d 04 d2       	vpmaddubsw %ymm10,%ymm8,%ymm10
  1a0665:	c4 41 1d fe e5       	vpaddd %ymm13,%ymm12,%ymm12
  1a066a:	c4 42 7d 58 ee       	vpbroadcastd %xmm14,%ymm13
  1a066f:	c4 62 15 08 e9       	vpsignb %ymm1,%ymm13,%ymm13
  1a0674:	c4 42 5d 04 ed       	vpmaddubsw %ymm13,%ymm4,%ymm13
  1a0679:	c5 2d f5 d3          	vpmaddwd %ymm3,%ymm10,%ymm10
  1a067d:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a0681:	c4 41 2d fe d5       	vpaddd %ymm13,%ymm10,%ymm10
  1a0686:	c4 41 79 70 ee 55    	vpshufd $0x55,%xmm14,%xmm13
  1a068c:	c4 42 7d 58 ed       	vpbroadcastd %xmm13,%ymm13
  1a0691:	c4 42 15 08 ef       	vpsignb %ymm15,%ymm13,%ymm13
  1a0696:	c4 42 55 04 ed       	vpmaddubsw %ymm13,%ymm5,%ymm13
  1a069b:	c5 15 f5 eb          	vpmaddwd %ymm3,%ymm13,%ymm13
  1a069f:	c4 41 2d fe d5       	vpaddd %ymm13,%ymm10,%ymm10
  1a06a4:	c4 41 1d fe d2       	vpaddd %ymm10,%ymm12,%ymm10
  1a06a9:	c4 21 7d 7f 94 dc e0 	vmovdqa %ymm10,0x3e0(%rsp,%r11,8)
  1a06b0:	03 00 00 
  1a06b3:	c4 42 79 13 57 80    	vcvtph2ps -0x80(%r15),%xmm10
  1a06b9:	c4 21 78 11 54 1c 60 	vmovups %xmm10,0x60(%rsp,%r11,1)
  1a06c0:	49 83 c3 10          	add    $0x10,%r11
  1a06c4:	4d 01 f7             	add    %r14,%r15
  1a06c7:	49 83 fb 40          	cmp    $0x40,%r11
  1a06cb:	0f 85 ef fa ff ff    	jne    1a01c0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x530>
  1a06d1:	c4 a1 7a 7e 04 2e    	vmovq  (%rsi,%r13,1),%xmm0
  1a06d7:	c5 f9 de 0d b1 3c e9 	vpmaxub -0x16c34f(%rip),%xmm0,%xmm1        # 34390 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x60>
  1a06de:	ff 
  1a06df:	c5 f9 74 c9          	vpcmpeqb %xmm1,%xmm0,%xmm1
  1a06e3:	c5 d9 76 e4          	vpcmpeqd %xmm4,%xmm4,%xmm4
  1a06e7:	c5 f1 ef cc          	vpxor  %xmm4,%xmm1,%xmm1
  1a06eb:	c4 e2 7d 21 c9       	vpmovsxbd %xmm1,%ymm1
  1a06f0:	c4 e2 7d 31 d0       	vpmovzxbd %xmm0,%ymm2
  1a06f5:	c5 fd 6f ac 24 80 05 	vmovdqa 0x580(%rsp),%ymm5
  1a06fc:	00 00 
  1a06fe:	c4 e2 55 47 d2       	vpsllvd %ymm2,%ymm5,%ymm2
  1a0703:	c5 f9 fc c4          	vpaddb %xmm4,%xmm0,%xmm0
  1a0707:	c4 e2 7d 31 c0       	vpmovzxbd %xmm0,%ymm0
  1a070c:	c5 fd 72 f0 17       	vpslld $0x17,%ymm0,%ymm0
  1a0711:	c4 e3 7d 4a c2 10    	vblendvps %ymm1,%ymm2,%ymm0,%ymm0
  1a0717:	c5 fc 5b 8c 24 80 03 	vcvtdq2ps 0x380(%rsp),%ymm1
  1a071e:	00 00 
  1a0720:	c4 e2 7d 18 54 24 60 	vbroadcastss 0x60(%rsp),%ymm2
  1a0727:	c5 fc 28 25 71 47 e9 	vmovaps -0x16b88f(%rip),%ymm4        # 34ea0 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0xb70>
  1a072e:	ff 
  1a072f:	c4 e2 5d 16 c0       	vpermps %ymm0,%ymm4,%ymm0
  1a0734:	c5 ec 59 d0          	vmulps %ymm0,%ymm2,%ymm2
  1a0738:	c5 fc 5b a4 24 a0 03 	vcvtdq2ps 0x3a0(%rsp),%ymm4
  1a073f:	00 00 
  1a0741:	c5 fc 28 ac 24 20 02 	vmovaps 0x220(%rsp),%ymm5
  1a0748:	00 00 
  1a074a:	c4 e2 75 b8 ea       	vfmadd231ps %ymm2,%ymm1,%ymm5
  1a074f:	c5 fc 29 ac 24 20 02 	vmovaps %ymm5,0x220(%rsp)
  1a0756:	00 00 
  1a0758:	c4 e2 7d 18 4c 24 64 	vbroadcastss 0x64(%rsp),%ymm1
  1a075f:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a0763:	c5 fc 5b 94 24 c0 03 	vcvtdq2ps 0x3c0(%rsp),%ymm2
  1a076a:	00 00 
  1a076c:	c4 e2 7d 18 6c 24 68 	vbroadcastss 0x68(%rsp),%ymm5
  1a0773:	c5 fc 28 b4 24 40 02 	vmovaps 0x240(%rsp),%ymm6
  1a077a:	00 00 
  1a077c:	c4 e2 5d b8 f1       	vfmadd231ps %ymm1,%ymm4,%ymm6
  1a0781:	c5 fc 29 b4 24 40 02 	vmovaps %ymm6,0x240(%rsp)
  1a0788:	00 00 
  1a078a:	c5 d4 59 c8          	vmulps %ymm0,%ymm5,%ymm1
  1a078e:	c5 fc 5b a4 24 e0 03 	vcvtdq2ps 0x3e0(%rsp),%ymm4
  1a0795:	00 00 
  1a0797:	c5 fc 28 ac 24 60 02 	vmovaps 0x260(%rsp),%ymm5
  1a079e:	00 00 
  1a07a0:	c4 e2 6d b8 e9       	vfmadd231ps %ymm1,%ymm2,%ymm5
  1a07a5:	c5 fc 29 ac 24 60 02 	vmovaps %ymm5,0x260(%rsp)
  1a07ac:	00 00 
  1a07ae:	c4 e2 7d 18 4c 24 6c 	vbroadcastss 0x6c(%rsp),%ymm1
  1a07b5:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a07b9:	c5 fc 5b 94 24 00 04 	vcvtdq2ps 0x400(%rsp),%ymm2
  1a07c0:	00 00 
  1a07c2:	c4 e2 7d 18 6c 24 70 	vbroadcastss 0x70(%rsp),%ymm5
  1a07c9:	c5 fc 28 b4 24 80 02 	vmovaps 0x280(%rsp),%ymm6
  1a07d0:	00 00 
  1a07d2:	c4 e2 5d b8 f1       	vfmadd231ps %ymm1,%ymm4,%ymm6
  1a07d7:	c5 fc 29 b4 24 80 02 	vmovaps %ymm6,0x280(%rsp)
  1a07de:	00 00 
  1a07e0:	c5 d4 59 c8          	vmulps %ymm0,%ymm5,%ymm1
  1a07e4:	c5 fc 5b a4 24 20 04 	vcvtdq2ps 0x420(%rsp),%ymm4
  1a07eb:	00 00 
  1a07ed:	c5 7c 28 bc 24 a0 05 	vmovaps 0x5a0(%rsp),%ymm15
  1a07f4:	00 00 
  1a07f6:	c4 62 6d b8 f9       	vfmadd231ps %ymm1,%ymm2,%ymm15
  1a07fb:	c4 e2 7d 18 4c 24 74 	vbroadcastss 0x74(%rsp),%ymm1
  1a0802:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a0806:	c5 fc 5b 94 24 40 04 	vcvtdq2ps 0x440(%rsp),%ymm2
  1a080d:	00 00 
  1a080f:	c4 e2 7d 18 6c 24 78 	vbroadcastss 0x78(%rsp),%ymm5
  1a0816:	c5 7c 28 b4 24 c0 05 	vmovaps 0x5c0(%rsp),%ymm14
  1a081d:	00 00 
  1a081f:	c4 62 5d b8 f1       	vfmadd231ps %ymm1,%ymm4,%ymm14
  1a0824:	c5 d4 59 c8          	vmulps %ymm0,%ymm5,%ymm1
  1a0828:	c5 fc 5b a4 24 60 04 	vcvtdq2ps 0x460(%rsp),%ymm4
  1a082f:	00 00 
  1a0831:	c5 7c 28 ac 24 e0 05 	vmovaps 0x5e0(%rsp),%ymm13
  1a0838:	00 00 
  1a083a:	c4 62 6d b8 e9       	vfmadd231ps %ymm1,%ymm2,%ymm13
  1a083f:	c4 e2 7d 18 4c 24 7c 	vbroadcastss 0x7c(%rsp),%ymm1
  1a0846:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a084a:	c5 fc 5b 94 24 80 04 	vcvtdq2ps 0x480(%rsp),%ymm2
  1a0851:	00 00 
  1a0853:	c4 e2 7d 18 ac 24 80 	vbroadcastss 0x80(%rsp),%ymm5
  1a085a:	00 00 00 
  1a085d:	c5 7c 28 a4 24 00 06 	vmovaps 0x600(%rsp),%ymm12
  1a0864:	00 00 
  1a0866:	c4 62 5d b8 e1       	vfmadd231ps %ymm1,%ymm4,%ymm12
  1a086b:	c5 d4 59 c8          	vmulps %ymm0,%ymm5,%ymm1
  1a086f:	c5 fc 5b a4 24 a0 04 	vcvtdq2ps 0x4a0(%rsp),%ymm4
  1a0876:	00 00 
  1a0878:	c5 7c 28 9c 24 20 06 	vmovaps 0x620(%rsp),%ymm11
  1a087f:	00 00 
  1a0881:	c4 62 6d b8 d9       	vfmadd231ps %ymm1,%ymm2,%ymm11
  1a0886:	c5 fb 10 8c 24 84 00 	vmovsd 0x84(%rsp),%xmm1
  1a088d:	00 00 
  1a088f:	c4 e2 7d 18 c9       	vbroadcastss %xmm1,%ymm1
  1a0894:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a0898:	c5 7c 28 94 24 40 06 	vmovaps 0x640(%rsp),%ymm10
  1a089f:	00 00 
  1a08a1:	c4 62 5d b8 d1       	vfmadd231ps %ymm1,%ymm4,%ymm10
  1a08a6:	c5 fc 5b 8c 24 c0 04 	vcvtdq2ps 0x4c0(%rsp),%ymm1
  1a08ad:	00 00 
  1a08af:	c5 fb 12 94 24 88 00 	vmovddup 0x88(%rsp),%xmm2
  1a08b6:	00 00 
  1a08b8:	c4 e2 7d 18 d2       	vbroadcastss %xmm2,%ymm2
  1a08bd:	c5 ec 59 d0          	vmulps %ymm0,%ymm2,%ymm2
  1a08c1:	c5 fc 5b a4 24 e0 04 	vcvtdq2ps 0x4e0(%rsp),%ymm4
  1a08c8:	00 00 
  1a08ca:	c5 7c 28 8c 24 60 06 	vmovaps 0x660(%rsp),%ymm9
  1a08d1:	00 00 
  1a08d3:	c4 62 75 b8 ca       	vfmadd231ps %ymm2,%ymm1,%ymm9
  1a08d8:	c4 e2 7d 18 8c 24 8c 	vbroadcastss 0x8c(%rsp),%ymm1
  1a08df:	00 00 00 
  1a08e2:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a08e6:	c5 fc 5b 94 24 00 05 	vcvtdq2ps 0x500(%rsp),%ymm2
  1a08ed:	00 00 
  1a08ef:	c4 e2 7d 18 ac 24 90 	vbroadcastss 0x90(%rsp),%ymm5
  1a08f6:	00 00 00 
  1a08f9:	c5 7c 28 84 24 80 06 	vmovaps 0x680(%rsp),%ymm8
  1a0900:	00 00 
  1a0902:	c4 62 5d b8 c1       	vfmadd231ps %ymm1,%ymm4,%ymm8
  1a0907:	c5 d4 59 c8          	vmulps %ymm0,%ymm5,%ymm1
  1a090b:	c5 fc 5b a4 24 20 05 	vcvtdq2ps 0x520(%rsp),%ymm4
  1a0912:	00 00 
  1a0914:	c5 fc 28 bc 24 a0 06 	vmovaps 0x6a0(%rsp),%ymm7
  1a091b:	00 00 
  1a091d:	c4 e2 6d b8 f9       	vfmadd231ps %ymm1,%ymm2,%ymm7
  1a0922:	c4 e2 7d 18 8c 24 94 	vbroadcastss 0x94(%rsp),%ymm1
  1a0929:	00 00 00 
  1a092c:	c5 f4 59 c8          	vmulps %ymm0,%ymm1,%ymm1
  1a0930:	c5 fc 5b 94 24 40 05 	vcvtdq2ps 0x540(%rsp),%ymm2
  1a0937:	00 00 
  1a0939:	c4 e2 7d 18 ac 24 98 	vbroadcastss 0x98(%rsp),%ymm5
  1a0940:	00 00 00 
  1a0943:	c5 fc 28 b4 24 c0 06 	vmovaps 0x6c0(%rsp),%ymm6
  1a094a:	00 00 
  1a094c:	c4 e2 5d b8 f1       	vfmadd231ps %ymm1,%ymm4,%ymm6
  1a0951:	c5 d4 59 c8          	vmulps %ymm0,%ymm5,%ymm1
  1a0955:	c4 e2 7d 18 a4 24 9c 	vbroadcastss 0x9c(%rsp),%ymm4
  1a095c:	00 00 00 
  1a095f:	c5 fc 28 ac 24 e0 06 	vmovaps 0x6e0(%rsp),%ymm5
  1a0966:	00 00 
  1a0968:	c4 e2 6d b8 e9       	vfmadd231ps %ymm1,%ymm2,%ymm5
  1a096d:	c5 dc 59 c0          	vmulps %ymm0,%ymm4,%ymm0
  1a0971:	c5 fc 5b 8c 24 60 05 	vcvtdq2ps 0x560(%rsp),%ymm1
  1a0978:	00 00 
  1a097a:	c5 fc 28 94 24 00 07 	vmovaps 0x700(%rsp),%ymm2
  1a0981:	00 00 
  1a0983:	c4 e2 75 b8 d0       	vfmadd231ps %ymm0,%ymm1,%ymm2
  1a0988:	c5 fc 28 ca          	vmovaps %ymm2,%ymm1
  1a098c:	49 ff c4             	inc    %r12
  1a098f:	49 81 c1 88 00 00 00 	add    $0x88,%r9
  1a0996:	49 39 fc             	cmp    %rdi,%r12
  1a0999:	0f 85 61 f5 ff ff    	jne    19ff00 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x270>
  1a099f:	48 89 de             	mov    %rbx,%rsi
  1a09a2:	48 c1 e6 05          	shl    $0x5,%rsi
  1a09a6:	48 03 74 24 10       	add    0x10(%rsp),%rsi
  1a09ab:	c4 e3 7d 04 84 24 20 	vpermilps $0xd8,0x220(%rsp),%ymm0
  1a09b2:	02 00 00 d8 
  1a09b6:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a09bc:	4c 8b 8c 24 98 01 00 	mov    0x198(%rsp),%r9
  1a09c3:	00 
  1a09c4:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a09ca:	c4 e3 7d 04 84 24 40 	vpermilps $0xd8,0x240(%rsp),%ymm0
  1a09d1:	02 00 00 d8 
  1a09d5:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a09db:	4c 8b 8c 24 90 01 00 	mov    0x190(%rsp),%r9
  1a09e2:	00 
  1a09e3:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a09e9:	c4 e3 7d 04 84 24 60 	vpermilps $0xd8,0x260(%rsp),%ymm0
  1a09f0:	02 00 00 d8 
  1a09f4:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a09fa:	4c 8b 8c 24 88 01 00 	mov    0x188(%rsp),%r9
  1a0a01:	00 
  1a0a02:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0a08:	c4 e3 7d 04 84 24 80 	vpermilps $0xd8,0x280(%rsp),%ymm0
  1a0a0f:	02 00 00 d8 
  1a0a13:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0a19:	4c 8b 8c 24 80 01 00 	mov    0x180(%rsp),%r9
  1a0a20:	00 
  1a0a21:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0a27:	c4 c1 04 c6 c7 d8    	vshufps $0xd8,%ymm15,%ymm15,%ymm0
  1a0a2d:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0a33:	4c 8b 8c 24 78 01 00 	mov    0x178(%rsp),%r9
  1a0a3a:	00 
  1a0a3b:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0a41:	c4 c1 0c c6 c6 d8    	vshufps $0xd8,%ymm14,%ymm14,%ymm0
  1a0a47:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0a4d:	4c 8b 8c 24 70 01 00 	mov    0x170(%rsp),%r9
  1a0a54:	00 
  1a0a55:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0a5b:	c4 c1 14 c6 c5 d8    	vshufps $0xd8,%ymm13,%ymm13,%ymm0
  1a0a61:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0a67:	4c 8b 8c 24 68 01 00 	mov    0x168(%rsp),%r9
  1a0a6e:	00 
  1a0a6f:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0a75:	c4 c1 1c c6 c4 d8    	vshufps $0xd8,%ymm12,%ymm12,%ymm0
  1a0a7b:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0a81:	4c 8b 8c 24 60 01 00 	mov    0x160(%rsp),%r9
  1a0a88:	00 
  1a0a89:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0a8f:	c4 c1 24 c6 c3 d8    	vshufps $0xd8,%ymm11,%ymm11,%ymm0
  1a0a95:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0a9b:	4c 8b 8c 24 58 01 00 	mov    0x158(%rsp),%r9
  1a0aa2:	00 
  1a0aa3:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0aa9:	c4 c1 2c c6 c2 d8    	vshufps $0xd8,%ymm10,%ymm10,%ymm0
  1a0aaf:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0ab5:	4c 8b 8c 24 50 01 00 	mov    0x150(%rsp),%r9
  1a0abc:	00 
  1a0abd:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0ac3:	c4 c1 34 c6 c1 d8    	vshufps $0xd8,%ymm9,%ymm9,%ymm0
  1a0ac9:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0acf:	4c 8b 8c 24 48 01 00 	mov    0x148(%rsp),%r9
  1a0ad6:	00 
  1a0ad7:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0add:	c4 c1 3c c6 c0 d8    	vshufps $0xd8,%ymm8,%ymm8,%ymm0
  1a0ae3:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0ae9:	4c 8b 8c 24 40 01 00 	mov    0x140(%rsp),%r9
  1a0af0:	00 
  1a0af1:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0af7:	c5 c4 c6 c7 d8       	vshufps $0xd8,%ymm7,%ymm7,%ymm0
  1a0afc:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0b02:	4c 8b 8c 24 38 01 00 	mov    0x138(%rsp),%r9
  1a0b09:	00 
  1a0b0a:	c4 a1 7c 11 04 8e    	vmovups %ymm0,(%rsi,%r9,4)
  1a0b10:	c5 cc c6 c6 d8       	vshufps $0xd8,%ymm6,%ymm6,%ymm0
  1a0b15:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0b1b:	c5 fc 11 04 86       	vmovups %ymm0,(%rsi,%rax,4)
  1a0b20:	c5 d4 c6 c5 d8       	vshufps $0xd8,%ymm5,%ymm5,%ymm0
  1a0b25:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0b2b:	c4 a1 7c 11 04 96    	vmovups %ymm0,(%rsi,%r10,4)
  1a0b31:	c5 f4 c6 c1 d8       	vshufps $0xd8,%ymm1,%ymm1,%ymm0
  1a0b36:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a0b3c:	c5 fc 11 04 8e       	vmovups %ymm0,(%rsi,%rcx,4)
  1a0b41:	48 ff c3             	inc    %rbx
  1a0b44:	48 3b 5c 24 28       	cmp    0x28(%rsp),%rbx
  1a0b49:	0f 85 31 f3 ff ff    	jne    19fe80 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1f0>
  1a0b4f:	4c 8b ac 24 30 01 00 	mov    0x130(%rsp),%r13
  1a0b56:	00 
  1a0b57:	49 83 c5 04          	add    $0x4,%r13
  1a0b5b:	48 8b 84 24 28 01 00 	mov    0x128(%rsp),%rax
  1a0b62:	00 
  1a0b63:	48 01 44 24 38       	add    %rax,0x38(%rsp)
  1a0b68:	4c 3b 6c 24 20       	cmp    0x20(%rsp),%r13
  1a0b6d:	0f 82 cd f1 ff ff    	jb     19fd40 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xb0>
  1a0b73:	e9 f5 00 00 00       	jmp    1a0c6d <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xfdd>
  1a0b78:	45 31 ed             	xor    %r13d,%r13d
  1a0b7b:	e9 ed 00 00 00       	jmp    1a0c6d <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xfdd>
  1a0b80:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
  1a0b85:	49 ff cd             	dec    %r13
  1a0b88:	49 83 e5 fc          	and    $0xfffffffffffffffc,%r13
  1a0b8c:	49 83 c5 04          	add    $0x4,%r13
  1a0b90:	e9 d8 00 00 00       	jmp    1a0c6d <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xfdd>
  1a0b95:	48 89 d0             	mov    %rdx,%rax
  1a0b98:	48 c1 e0 06          	shl    $0x6,%rax
  1a0b9c:	48 8d 0c 95 00 00 00 	lea    0x0(,%rdx,4),%rcx
  1a0ba3:	00 
  1a0ba4:	45 31 ed             	xor    %r13d,%r13d
  1a0ba7:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a0bab:	48 8d 34 09          	lea    (%rcx,%rcx,1),%rsi
  1a0baf:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
  1a0bb4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  1a0bbb:	00 00 00 00 00 
  1a0bc0:	4d 89 ca             	mov    %r9,%r10
  1a0bc3:	4c 8b 5c 24 28       	mov    0x28(%rsp),%r11
  1a0bc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  1a0bcf:	00 
  1a0bd0:	c4 c1 7c 11 02       	vmovups %ymm0,(%r10)
  1a0bd5:	4d 8d 34 0a          	lea    (%r10,%rcx,1),%r14
  1a0bd9:	c4 c1 7c 11 04 92    	vmovups %ymm0,(%r10,%rdx,4)
  1a0bdf:	49 01 ce             	add    %rcx,%r14
  1a0be2:	c4 c1 7c 11 04 d2    	vmovups %ymm0,(%r10,%rdx,8)
  1a0be8:	c4 c1 7c 11 04 96    	vmovups %ymm0,(%r14,%rdx,4)
  1a0bee:	c4 c1 7c 11 04 d6    	vmovups %ymm0,(%r14,%rdx,8)
  1a0bf4:	49 01 f6             	add    %rsi,%r14
  1a0bf7:	49 8d 1c 96          	lea    (%r14,%rdx,4),%rbx
  1a0bfb:	c4 c1 7c 11 04 96    	vmovups %ymm0,(%r14,%rdx,4)
  1a0c01:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c06:	48 01 cb             	add    %rcx,%rbx
  1a0c09:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c0e:	48 01 cb             	add    %rcx,%rbx
  1a0c11:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c16:	48 01 cb             	add    %rcx,%rbx
  1a0c19:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c1e:	48 01 cb             	add    %rcx,%rbx
  1a0c21:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c26:	48 01 cb             	add    %rcx,%rbx
  1a0c29:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c2e:	48 01 cb             	add    %rcx,%rbx
  1a0c31:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c36:	48 01 cb             	add    %rcx,%rbx
  1a0c39:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c3e:	48 01 cb             	add    %rcx,%rbx
  1a0c41:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c46:	48 01 cb             	add    %rcx,%rbx
  1a0c49:	c5 fc 11 04 19       	vmovups %ymm0,(%rcx,%rbx,1)
  1a0c4e:	49 83 c2 20          	add    $0x20,%r10
  1a0c52:	49 ff cb             	dec    %r11
  1a0c55:	0f 85 75 ff ff ff    	jne    1a0bd0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xf40>
  1a0c5b:	49 83 c5 04          	add    $0x4,%r13
  1a0c5f:	49 01 c1             	add    %rax,%r9
  1a0c62:	4c 3b 6c 24 20       	cmp    0x20(%rsp),%r13
  1a0c67:	0f 82 53 ff ff ff    	jb     1a0bc0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0xf30>
  1a0c6d:	4c 3b 6c 24 18       	cmp    0x18(%rsp),%r13
  1a0c72:	4c 8b 65 10          	mov    0x10(%rbp),%r12
  1a0c76:	0f 83 96 08 00 00    	jae    1a1512 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1882>
  1a0c7c:	49 c1 ec 03          	shr    $0x3,%r12
  1a0c80:	0f 84 8c 08 00 00    	je     1a1512 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1882>
  1a0c86:	48 85 ff             	test   %rdi,%rdi
  1a0c89:	0f 84 95 08 00 00    	je     1a1524 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1894>
  1a0c8f:	48 89 f8             	mov    %rdi,%rax
  1a0c92:	48 c1 e0 07          	shl    $0x7,%rax
  1a0c96:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  1a0c9a:	4c 89 e9             	mov    %r13,%rcx
  1a0c9d:	48 0f af cf          	imul   %rdi,%rcx
  1a0ca1:	48 89 ce             	mov    %rcx,%rsi
  1a0ca4:	48 c1 e6 07          	shl    $0x7,%rsi
  1a0ca8:	48 8d 0c ce          	lea    (%rsi,%rcx,8),%rcx
  1a0cac:	49 01 c8             	add    %rcx,%r8
  1a0caf:	c4 e2 7d 78 15 68 78 	vpbroadcastb -0x168798(%rip),%ymm2        # 38520 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x368>
  1a0cb6:	e9 ff 
  1a0cb8:	c5 7d 6f 15 c0 45 e9 	vmovdqa -0x16ba40(%rip),%ymm10        # 35280 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  1a0cbf:	ff 
  1a0cc0:	c4 62 7d 79 25 55 78 	vpbroadcastw -0x1687ab(%rip),%ymm12        # 3851e <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x366>
  1a0cc7:	e9 ff 
  1a0cc9:	c4 e2 7d 18 05 3e 49 	vbroadcastss -0x16b6c2(%rip),%ymm0        # 35610 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x3d0>
  1a0cd0:	e9 ff 
  1a0cd2:	c5 fc 29 84 24 20 02 	vmovaps %ymm0,0x220(%rsp)
  1a0cd9:	00 00 
  1a0cdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  1a0ce0:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
  1a0ce7:	00 
  1a0ce8:	48 0f af ca          	imul   %rdx,%rcx
  1a0cec:	4a 8d 34 ad 01 00 00 	lea    0x1(,%r13,4),%rsi
  1a0cf3:	00 
  1a0cf4:	48 0f af f2          	imul   %rdx,%rsi
  1a0cf8:	4e 8d 0c ad 02 00 00 	lea    0x2(,%r13,4),%r9
  1a0cff:	00 
  1a0d00:	4c 0f af ca          	imul   %rdx,%r9
  1a0d04:	4e 8d 14 ad 03 00 00 	lea    0x3(,%r13,4),%r10
  1a0d0b:	00 
  1a0d0c:	4c 0f af d2          	imul   %rdx,%r10
  1a0d10:	4c 8b 5c 24 30       	mov    0x30(%rsp),%r11
  1a0d15:	31 db                	xor    %ebx,%ebx
  1a0d17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  1a0d1e:	00 00 
  1a0d20:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  1a0d24:	45 31 f6             	xor    %r14d,%r14d
  1a0d27:	49 89 ff             	mov    %rdi,%r15
  1a0d2a:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a0d2e:	c5 fc 29 84 24 e0 01 	vmovaps %ymm0,0x1e0(%rsp)
  1a0d35:	00 00 
  1a0d37:	c5 fc 29 84 24 c0 01 	vmovaps %ymm0,0x1c0(%rsp)
  1a0d3e:	00 00 
  1a0d40:	c5 fc 29 84 24 a0 01 	vmovaps %ymm0,0x1a0(%rsp)
  1a0d47:	00 00 
  1a0d49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  1a0d50:	c5 fc 29 4c 24 40    	vmovaps %ymm1,0x40(%rsp)
  1a0d56:	c4 81 7e 6f 44 33 08 	vmovdqu 0x8(%r11,%r14,1),%ymm0
  1a0d5d:	c4 81 7e 6f 4c 33 28 	vmovdqu 0x28(%r11,%r14,1),%ymm1
  1a0d64:	c4 81 7e 6f 5c 33 48 	vmovdqu 0x48(%r11,%r14,1),%ymm3
  1a0d6b:	c4 81 7e 6f 64 33 68 	vmovdqu 0x68(%r11,%r14,1),%ymm4
  1a0d72:	c5 fd db ea          	vpand  %ymm2,%ymm0,%ymm5
  1a0d76:	c4 e2 2d 00 ed       	vpshufb %ymm5,%ymm10,%ymm5
  1a0d7b:	c5 f5 db f2          	vpand  %ymm2,%ymm1,%ymm6
  1a0d7f:	c4 e2 2d 00 f6       	vpshufb %ymm6,%ymm10,%ymm6
  1a0d84:	c5 e5 db fa          	vpand  %ymm2,%ymm3,%ymm7
  1a0d88:	c4 62 2d 00 c7       	vpshufb %ymm7,%ymm10,%ymm8
  1a0d8d:	c5 dd db fa          	vpand  %ymm2,%ymm4,%ymm7
  1a0d91:	c4 62 2d 00 cf       	vpshufb %ymm7,%ymm10,%ymm9
  1a0d96:	c5 fd 71 d0 04       	vpsrlw $0x4,%ymm0,%ymm0
  1a0d9b:	c5 fd db c2          	vpand  %ymm2,%ymm0,%ymm0
  1a0d9f:	c4 e2 2d 00 c0       	vpshufb %ymm0,%ymm10,%ymm0
  1a0da4:	c5 f5 71 d1 04       	vpsrlw $0x4,%ymm1,%ymm1
  1a0da9:	c5 f5 db ca          	vpand  %ymm2,%ymm1,%ymm1
  1a0dad:	c4 e2 2d 00 c9       	vpshufb %ymm1,%ymm10,%ymm1
  1a0db2:	c5 e5 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm3
  1a0db7:	c5 e5 db da          	vpand  %ymm2,%ymm3,%ymm3
  1a0dbb:	c4 62 2d 00 d3       	vpshufb %ymm3,%ymm10,%ymm10
  1a0dc0:	c5 e5 71 d4 04       	vpsrlw $0x4,%ymm4,%ymm3
  1a0dc5:	c5 e5 db da          	vpand  %ymm2,%ymm3,%ymm3
  1a0dc9:	c5 fd 6f 15 af 44 e9 	vmovdqa -0x16bb51(%rip),%ymm2        # 35280 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  1a0dd0:	ff 
  1a0dd1:	c4 62 6d 00 db       	vpshufb %ymm3,%ymm2,%ymm11
  1a0dd6:	c5 fd 70 de a0       	vpshufd $0xa0,%ymm6,%ymm3
  1a0ddb:	c4 63 55 02 fb aa    	vpblendd $0xaa,%ymm3,%ymm5,%ymm15
  1a0de1:	c4 42 05 08 f7       	vpsignb %ymm15,%ymm15,%ymm14
  1a0de6:	c5 fd 70 e5 f5       	vpshufd $0xf5,%ymm5,%ymm4
  1a0deb:	c4 e3 5d 02 fe aa    	vpblendd $0xaa,%ymm6,%ymm4,%ymm7
  1a0df1:	c4 e2 45 08 ef       	vpsignb %ymm7,%ymm7,%ymm5
  1a0df6:	c5 fd 7f ac 24 a0 02 	vmovdqa %ymm5,0x2a0(%rsp)
  1a0dfd:	00 00 
  1a0dff:	c4 c1 7d 70 f1 a0    	vpshufd $0xa0,%ymm9,%ymm6
  1a0e05:	c4 e3 3d 02 f6 aa    	vpblendd $0xaa,%ymm6,%ymm8,%ymm6
  1a0e0b:	c4 62 4d 08 ee       	vpsignb %ymm6,%ymm6,%ymm13
  1a0e10:	c4 41 7d 70 c0 f5    	vpshufd $0xf5,%ymm8,%ymm8
  1a0e16:	c4 c3 3d 02 d1 aa    	vpblendd $0xaa,%ymm9,%ymm8,%ymm2
  1a0e1c:	c5 fd 7f 94 24 c0 00 	vmovdqa %ymm2,0xc0(%rsp)
  1a0e23:	00 00 
  1a0e25:	c4 e2 6d 08 da       	vpsignb %ymm2,%ymm2,%ymm3
  1a0e2a:	c5 fd 7f 9c 24 e0 00 	vmovdqa %ymm3,0xe0(%rsp)
  1a0e31:	00 00 
  1a0e33:	c5 7d 70 c1 a0       	vpshufd $0xa0,%ymm1,%ymm8
  1a0e38:	c4 c3 7d 02 d0 aa    	vpblendd $0xaa,%ymm8,%ymm0,%ymm2
  1a0e3e:	c5 fd 7f 94 24 00 01 	vmovdqa %ymm2,0x100(%rsp)
  1a0e45:	00 00 
  1a0e47:	c5 fd 70 c0 f5       	vpshufd $0xf5,%ymm0,%ymm0
  1a0e4c:	c4 e3 7d 02 c1 aa    	vpblendd $0xaa,%ymm1,%ymm0,%ymm0
  1a0e52:	c5 fd 7f 84 24 40 03 	vmovdqa %ymm0,0x340(%rsp)
  1a0e59:	00 00 
  1a0e5b:	c4 c1 7d 70 c3 a0    	vpshufd $0xa0,%ymm11,%ymm0
  1a0e61:	c4 e3 2d 02 c0 aa    	vpblendd $0xaa,%ymm0,%ymm10,%ymm0
  1a0e67:	c5 fd 7f 84 24 c0 02 	vmovdqa %ymm0,0x2c0(%rsp)
  1a0e6e:	00 00 
  1a0e70:	c4 c1 7d 70 c2 f5    	vpshufd $0xf5,%ymm10,%ymm0
  1a0e76:	c4 c3 7d 02 c3 aa    	vpblendd $0xaa,%ymm11,%ymm0,%ymm0
  1a0e7c:	c5 fd 7f 84 24 00 02 	vmovdqa %ymm0,0x200(%rsp)
  1a0e83:	00 00 
  1a0e85:	c4 01 7a 7e 44 30 28 	vmovq  0x28(%r8,%r14,1),%xmm8
  1a0e8c:	c4 81 7a 7e 04 33    	vmovq  (%r11,%r14,1),%xmm0
  1a0e92:	c5 f9 de 0d f6 34 e9 	vpmaxub -0x16cb0a(%rip),%xmm0,%xmm1        # 34390 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x60>
  1a0e99:	ff 
  1a0e9a:	c5 f9 74 c9          	vpcmpeqb %xmm1,%xmm0,%xmm1
  1a0e9e:	c5 e9 76 d2          	vpcmpeqd %xmm2,%xmm2,%xmm2
  1a0ea2:	c5 f1 ef ca          	vpxor  %xmm2,%xmm1,%xmm1
  1a0ea6:	c4 e2 7d 21 c9       	vpmovsxbd %xmm1,%ymm1
  1a0eab:	c4 62 7d 31 c8       	vpmovzxbd %xmm0,%ymm9
  1a0eb0:	c5 fd 6f 9c 24 20 02 	vmovdqa 0x220(%rsp),%ymm3
  1a0eb7:	00 00 
  1a0eb9:	c4 42 65 47 c9       	vpsllvd %ymm9,%ymm3,%ymm9
  1a0ebe:	c5 f9 fc c2          	vpaddb %xmm2,%xmm0,%xmm0
  1a0ec2:	c4 e2 7d 31 c0       	vpmovzxbd %xmm0,%ymm0
  1a0ec7:	c5 fd 72 f0 17       	vpslld $0x17,%ymm0,%ymm0
  1a0ecc:	c4 c3 7d 4a e1 10    	vblendvps %ymm1,%ymm9,%ymm0,%ymm4
  1a0ed2:	c4 81 7a 7e 44 30 08 	vmovq  0x8(%r8,%r14,1),%xmm0
  1a0ed9:	c4 62 7d 58 c8       	vpbroadcastd %xmm0,%ymm9
  1a0ede:	c4 42 35 08 cf       	vpsignb %ymm15,%ymm9,%ymm9
  1a0ee3:	c4 42 0d 04 c9       	vpmaddubsw %ymm9,%ymm14,%ymm9
  1a0ee8:	c4 41 35 f5 cc       	vpmaddwd %ymm12,%ymm9,%ymm9
  1a0eed:	c5 f9 70 c0 55       	vpshufd $0x55,%xmm0,%xmm0
  1a0ef2:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  1a0ef7:	c5 7d 6f df          	vmovdqa %ymm7,%ymm11
  1a0efb:	c5 fd 7f bc 24 60 02 	vmovdqa %ymm7,0x260(%rsp)
  1a0f02:	00 00 
  1a0f04:	c4 e2 7d 08 c7       	vpsignb %ymm7,%ymm0,%ymm0
  1a0f09:	c4 e2 55 04 c0       	vpmaddubsw %ymm0,%ymm5,%ymm0
  1a0f0e:	c5 9d f5 c0          	vpmaddwd %ymm0,%ymm12,%ymm0
  1a0f12:	c5 b5 fe c0          	vpaddd %ymm0,%ymm9,%ymm0
  1a0f16:	c4 42 7d 58 c8       	vpbroadcastd %xmm8,%ymm9
  1a0f1b:	c5 fd 7f b4 24 a0 00 	vmovdqa %ymm6,0xa0(%rsp)
  1a0f22:	00 00 
  1a0f24:	c4 62 35 08 ce       	vpsignb %ymm6,%ymm9,%ymm9
  1a0f29:	c5 7d 7f eb          	vmovdqa %ymm13,%ymm3
  1a0f2d:	c5 7d 7f ac 24 40 02 	vmovdqa %ymm13,0x240(%rsp)
  1a0f34:	00 00 
  1a0f36:	c4 42 15 04 c9       	vpmaddubsw %ymm9,%ymm13,%ymm9
  1a0f3b:	c4 41 35 f5 cc       	vpmaddwd %ymm12,%ymm9,%ymm9
  1a0f40:	c4 41 79 70 c0 55    	vpshufd $0x55,%xmm8,%xmm8
  1a0f46:	c4 42 7d 58 c0       	vpbroadcastd %xmm8,%ymm8
  1a0f4b:	c5 7d 6f ac 24 c0 00 	vmovdqa 0xc0(%rsp),%ymm13
  1a0f52:	00 00 
  1a0f54:	c4 42 3d 08 c5       	vpsignb %ymm13,%ymm8,%ymm8
  1a0f59:	c5 fd 6f bc 24 e0 00 	vmovdqa 0xe0(%rsp),%ymm7
  1a0f60:	00 00 
  1a0f62:	c4 42 45 04 c0       	vpmaddubsw %ymm8,%ymm7,%ymm8
  1a0f67:	c4 41 3d f5 c4       	vpmaddwd %ymm12,%ymm8,%ymm8
  1a0f6c:	c4 41 35 fe c0       	vpaddd %ymm8,%ymm9,%ymm8
  1a0f71:	c4 01 7a 7e 4c 30 48 	vmovq  0x48(%r8,%r14,1),%xmm9
  1a0f78:	c5 bd fe c0          	vpaddd %ymm0,%ymm8,%ymm0
  1a0f7c:	c4 42 7d 58 c1       	vpbroadcastd %xmm9,%ymm8
  1a0f81:	c5 fd 6f 8c 24 00 01 	vmovdqa 0x100(%rsp),%ymm1
  1a0f88:	00 00 
  1a0f8a:	c4 62 3d 08 d1       	vpsignb %ymm1,%ymm8,%ymm10
  1a0f8f:	c4 e2 75 08 c9       	vpsignb %ymm1,%ymm1,%ymm1
  1a0f94:	c5 fd 7f 8c 24 20 03 	vmovdqa %ymm1,0x320(%rsp)
  1a0f9b:	00 00 
  1a0f9d:	c4 42 75 04 d2       	vpmaddubsw %ymm10,%ymm1,%ymm10
  1a0fa2:	c4 41 2d f5 d4       	vpmaddwd %ymm12,%ymm10,%ymm10
  1a0fa7:	c4 41 79 70 c9 55    	vpshufd $0x55,%xmm9,%xmm9
  1a0fad:	c4 42 7d 58 c9       	vpbroadcastd %xmm9,%ymm9
  1a0fb2:	c5 fd 6f ac 24 40 03 	vmovdqa 0x340(%rsp),%ymm5
  1a0fb9:	00 00 
  1a0fbb:	c4 e2 35 08 d5       	vpsignb %ymm5,%ymm9,%ymm2
  1a0fc0:	c4 e2 55 08 cd       	vpsignb %ymm5,%ymm5,%ymm1
  1a0fc5:	c5 fd 7f 8c 24 00 03 	vmovdqa %ymm1,0x300(%rsp)
  1a0fcc:	00 00 
  1a0fce:	c5 7d 6f cd          	vmovdqa %ymm5,%ymm9
  1a0fd2:	c4 e2 75 04 d2       	vpmaddubsw %ymm2,%ymm1,%ymm2
  1a0fd7:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a0fdb:	c5 ad fe d2          	vpaddd %ymm2,%ymm10,%ymm2
  1a0fdf:	c5 fd fe d2          	vpaddd %ymm2,%ymm0,%ymm2
  1a0fe3:	c4 81 7a 7e 44 30 68 	vmovq  0x68(%r8,%r14,1),%xmm0
  1a0fea:	c4 62 7d 58 d0       	vpbroadcastd %xmm0,%ymm10
  1a0fef:	c5 fd 6f ac 24 c0 02 	vmovdqa 0x2c0(%rsp),%ymm5
  1a0ff6:	00 00 
  1a0ff8:	c4 e2 2d 08 cd       	vpsignb %ymm5,%ymm10,%ymm1
  1a0ffd:	c4 62 55 08 d5       	vpsignb %ymm5,%ymm5,%ymm10
  1a1002:	c4 e2 2d 04 c9       	vpmaddubsw %ymm1,%ymm10,%ymm1
  1a1007:	c5 7d 7f 94 24 80 02 	vmovdqa %ymm10,0x280(%rsp)
  1a100e:	00 00 
  1a1010:	c5 9d f5 c9          	vpmaddwd %ymm1,%ymm12,%ymm1
  1a1014:	c5 f9 70 c0 55       	vpshufd $0x55,%xmm0,%xmm0
  1a1019:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  1a101e:	c5 fd 6f ac 24 00 02 	vmovdqa 0x200(%rsp),%ymm5
  1a1025:	00 00 
  1a1027:	c4 e2 7d 08 c5       	vpsignb %ymm5,%ymm0,%ymm0
  1a102c:	c4 e2 55 08 ed       	vpsignb %ymm5,%ymm5,%ymm5
  1a1031:	c5 fd 7f ac 24 e0 02 	vmovdqa %ymm5,0x2e0(%rsp)
  1a1038:	00 00 
  1a103a:	c4 e2 55 04 c0       	vpmaddubsw %ymm0,%ymm5,%ymm0
  1a103f:	c5 9d f5 c0          	vpmaddwd %ymm0,%ymm12,%ymm0
  1a1043:	c5 f5 fe c0          	vpaddd %ymm0,%ymm1,%ymm0
  1a1047:	c5 ed fe c0          	vpaddd %ymm0,%ymm2,%ymm0
  1a104b:	c4 81 79 c4 0c 30 00 	vpinsrw $0x0,(%r8,%r14,1),%xmm0,%xmm1
  1a1052:	c4 e2 79 13 d1       	vcvtph2ps %xmm1,%xmm2
  1a1057:	c5 fc 28 0d 41 3e e9 	vmovaps -0x16c1bf(%rip),%ymm1        # 34ea0 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0xb70>
  1a105e:	ff 
  1a105f:	c4 e2 75 16 cc       	vpermps %ymm4,%ymm1,%ymm1
  1a1064:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  1a1068:	c4 e2 7d 18 d2       	vbroadcastss %xmm2,%ymm2
  1a106d:	c5 ec 59 d1          	vmulps %ymm1,%ymm2,%ymm2
  1a1071:	c5 fc 28 a4 24 a0 01 	vmovaps 0x1a0(%rsp),%ymm4
  1a1078:	00 00 
  1a107a:	c4 e2 7d b8 e2       	vfmadd231ps %ymm2,%ymm0,%ymm4
  1a107f:	c5 fc 29 a4 24 a0 01 	vmovaps %ymm4,0x1a0(%rsp)
  1a1086:	00 00 
  1a1088:	c4 81 7a 7e 44 30 10 	vmovq  0x10(%r8,%r14,1),%xmm0
  1a108f:	c4 e2 7d 58 d0       	vpbroadcastd %xmm0,%ymm2
  1a1094:	c5 7d 7f bc 24 60 03 	vmovdqa %ymm15,0x360(%rsp)
  1a109b:	00 00 
  1a109d:	c4 c2 6d 08 d7       	vpsignb %ymm15,%ymm2,%ymm2
  1a10a2:	c4 e2 0d 04 d2       	vpmaddubsw %ymm2,%ymm14,%ymm2
  1a10a7:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a10ab:	c5 f9 70 c0 55       	vpshufd $0x55,%xmm0,%xmm0
  1a10b0:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  1a10b5:	c4 c2 7d 08 c3       	vpsignb %ymm11,%ymm0,%ymm0
  1a10ba:	c5 7d 6f 84 24 a0 02 	vmovdqa 0x2a0(%rsp),%ymm8
  1a10c1:	00 00 
  1a10c3:	c4 e2 3d 04 c0       	vpmaddubsw %ymm0,%ymm8,%ymm0
  1a10c8:	c5 9d f5 c0          	vpmaddwd %ymm0,%ymm12,%ymm0
  1a10cc:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a10d0:	c4 81 7a 7e 54 30 30 	vmovq  0x30(%r8,%r14,1),%xmm2
  1a10d7:	c4 e2 7d 58 e2       	vpbroadcastd %xmm2,%ymm4
  1a10dc:	c4 e2 5d 08 a4 24 a0 	vpsignb 0xa0(%rsp),%ymm4,%ymm4
  1a10e3:	00 00 00 
  1a10e6:	c4 e2 65 04 e4       	vpmaddubsw %ymm4,%ymm3,%ymm4
  1a10eb:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a10ef:	c5 f9 70 d2 55       	vpshufd $0x55,%xmm2,%xmm2
  1a10f4:	c4 e2 7d 58 d2       	vpbroadcastd %xmm2,%ymm2
  1a10f9:	c4 c2 6d 08 d5       	vpsignb %ymm13,%ymm2,%ymm2
  1a10fe:	c4 e2 45 04 d2       	vpmaddubsw %ymm2,%ymm7,%ymm2
  1a1103:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a1107:	c5 dd fe d2          	vpaddd %ymm2,%ymm4,%ymm2
  1a110b:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a110f:	c4 81 7a 7e 54 30 50 	vmovq  0x50(%r8,%r14,1),%xmm2
  1a1116:	c4 e2 7d 58 e2       	vpbroadcastd %xmm2,%ymm4
  1a111b:	c5 fd 6f 9c 24 00 01 	vmovdqa 0x100(%rsp),%ymm3
  1a1122:	00 00 
  1a1124:	c4 e2 5d 08 e3       	vpsignb %ymm3,%ymm4,%ymm4
  1a1129:	c5 fd 6f b4 24 20 03 	vmovdqa 0x320(%rsp),%ymm6
  1a1130:	00 00 
  1a1132:	c4 e2 4d 04 e4       	vpmaddubsw %ymm4,%ymm6,%ymm4
  1a1137:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a113b:	c5 f9 70 d2 55       	vpshufd $0x55,%xmm2,%xmm2
  1a1140:	c4 e2 7d 58 d2       	vpbroadcastd %xmm2,%ymm2
  1a1145:	c5 7d 7f cf          	vmovdqa %ymm9,%ymm7
  1a1149:	c4 c2 6d 08 d1       	vpsignb %ymm9,%ymm2,%ymm2
  1a114e:	c5 7d 6f 8c 24 00 03 	vmovdqa 0x300(%rsp),%ymm9
  1a1155:	00 00 
  1a1157:	c4 e2 35 04 d2       	vpmaddubsw %ymm2,%ymm9,%ymm2
  1a115c:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a1160:	c5 dd fe d2          	vpaddd %ymm2,%ymm4,%ymm2
  1a1164:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a1168:	c4 81 7a 7e 54 30 70 	vmovq  0x70(%r8,%r14,1),%xmm2
  1a116f:	c4 e2 7d 58 e2       	vpbroadcastd %xmm2,%ymm4
  1a1174:	c5 fd 6f ac 24 c0 02 	vmovdqa 0x2c0(%rsp),%ymm5
  1a117b:	00 00 
  1a117d:	c4 e2 5d 08 e5       	vpsignb %ymm5,%ymm4,%ymm4
  1a1182:	c4 e2 2d 04 e4       	vpmaddubsw %ymm4,%ymm10,%ymm4
  1a1187:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a118b:	c5 f9 70 d2 55       	vpshufd $0x55,%xmm2,%xmm2
  1a1190:	c4 e2 7d 58 d2       	vpbroadcastd %xmm2,%ymm2
  1a1195:	c5 7d 6f 94 24 00 02 	vmovdqa 0x200(%rsp),%ymm10
  1a119c:	00 00 
  1a119e:	c4 c2 6d 08 d2       	vpsignb %ymm10,%ymm2,%ymm2
  1a11a3:	c5 7d 6f 9c 24 e0 02 	vmovdqa 0x2e0(%rsp),%ymm11
  1a11aa:	00 00 
  1a11ac:	c4 e2 25 04 d2       	vpmaddubsw %ymm2,%ymm11,%ymm2
  1a11b1:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a11b5:	c5 dd fe d2          	vpaddd %ymm2,%ymm4,%ymm2
  1a11b9:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a11bd:	c4 81 79 c4 54 30 02 	vpinsrw $0x0,0x2(%r8,%r14,1),%xmm0,%xmm2
  1a11c4:	00 
  1a11c5:	c4 e2 79 13 d2       	vcvtph2ps %xmm2,%xmm2
  1a11ca:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  1a11ce:	c4 e2 7d 18 d2       	vbroadcastss %xmm2,%ymm2
  1a11d3:	c5 ec 59 d1          	vmulps %ymm1,%ymm2,%ymm2
  1a11d7:	c5 fc 28 a4 24 c0 01 	vmovaps 0x1c0(%rsp),%ymm4
  1a11de:	00 00 
  1a11e0:	c4 e2 7d b8 e2       	vfmadd231ps %ymm2,%ymm0,%ymm4
  1a11e5:	c5 fc 29 a4 24 c0 01 	vmovaps %ymm4,0x1c0(%rsp)
  1a11ec:	00 00 
  1a11ee:	c4 81 7a 7e 44 30 18 	vmovq  0x18(%r8,%r14,1),%xmm0
  1a11f5:	c4 e2 7d 58 d0       	vpbroadcastd %xmm0,%ymm2
  1a11fa:	c4 c2 6d 08 d7       	vpsignb %ymm15,%ymm2,%ymm2
  1a11ff:	c4 e2 0d 04 d2       	vpmaddubsw %ymm2,%ymm14,%ymm2
  1a1204:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a1208:	c5 f9 70 c0 55       	vpshufd $0x55,%xmm0,%xmm0
  1a120d:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  1a1212:	c5 7d 6f ac 24 60 02 	vmovdqa 0x260(%rsp),%ymm13
  1a1219:	00 00 
  1a121b:	c4 c2 7d 08 c5       	vpsignb %ymm13,%ymm0,%ymm0
  1a1220:	c4 e2 3d 04 c0       	vpmaddubsw %ymm0,%ymm8,%ymm0
  1a1225:	c5 9d f5 c0          	vpmaddwd %ymm0,%ymm12,%ymm0
  1a1229:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a122d:	c4 81 7a 7e 54 30 38 	vmovq  0x38(%r8,%r14,1),%xmm2
  1a1234:	c4 e2 7d 58 e2       	vpbroadcastd %xmm2,%ymm4
  1a1239:	c4 e2 5d 08 a4 24 a0 	vpsignb 0xa0(%rsp),%ymm4,%ymm4
  1a1240:	00 00 00 
  1a1243:	c5 7d 6f bc 24 40 02 	vmovdqa 0x240(%rsp),%ymm15
  1a124a:	00 00 
  1a124c:	c4 e2 05 04 e4       	vpmaddubsw %ymm4,%ymm15,%ymm4
  1a1251:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a1255:	c5 f9 70 d2 55       	vpshufd $0x55,%xmm2,%xmm2
  1a125a:	c4 e2 7d 58 d2       	vpbroadcastd %xmm2,%ymm2
  1a125f:	c4 e2 6d 08 94 24 c0 	vpsignb 0xc0(%rsp),%ymm2,%ymm2
  1a1266:	00 00 00 
  1a1269:	c5 7d 6f 84 24 e0 00 	vmovdqa 0xe0(%rsp),%ymm8
  1a1270:	00 00 
  1a1272:	c4 e2 3d 04 d2       	vpmaddubsw %ymm2,%ymm8,%ymm2
  1a1277:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a127b:	c5 dd fe d2          	vpaddd %ymm2,%ymm4,%ymm2
  1a127f:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a1283:	c4 81 7a 7e 54 30 58 	vmovq  0x58(%r8,%r14,1),%xmm2
  1a128a:	c4 e2 7d 58 e2       	vpbroadcastd %xmm2,%ymm4
  1a128f:	c4 e2 5d 08 e3       	vpsignb %ymm3,%ymm4,%ymm4
  1a1294:	c4 e2 4d 04 e4       	vpmaddubsw %ymm4,%ymm6,%ymm4
  1a1299:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a129d:	c5 f9 70 d2 55       	vpshufd $0x55,%xmm2,%xmm2
  1a12a2:	c4 e2 7d 58 d2       	vpbroadcastd %xmm2,%ymm2
  1a12a7:	c4 e2 6d 08 d7       	vpsignb %ymm7,%ymm2,%ymm2
  1a12ac:	c4 e2 35 04 d2       	vpmaddubsw %ymm2,%ymm9,%ymm2
  1a12b1:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a12b5:	c5 dd fe d2          	vpaddd %ymm2,%ymm4,%ymm2
  1a12b9:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a12bd:	c4 81 7a 7e 54 30 78 	vmovq  0x78(%r8,%r14,1),%xmm2
  1a12c4:	c4 e2 7d 58 e2       	vpbroadcastd %xmm2,%ymm4
  1a12c9:	c4 e2 5d 08 e5       	vpsignb %ymm5,%ymm4,%ymm4
  1a12ce:	c5 fd 6f 9c 24 80 02 	vmovdqa 0x280(%rsp),%ymm3
  1a12d5:	00 00 
  1a12d7:	c4 e2 65 04 e4       	vpmaddubsw %ymm4,%ymm3,%ymm4
  1a12dc:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a12e0:	c5 f9 70 d2 55       	vpshufd $0x55,%xmm2,%xmm2
  1a12e5:	c4 e2 7d 58 d2       	vpbroadcastd %xmm2,%ymm2
  1a12ea:	c4 c2 6d 08 d2       	vpsignb %ymm10,%ymm2,%ymm2
  1a12ef:	c4 e2 25 04 d2       	vpmaddubsw %ymm2,%ymm11,%ymm2
  1a12f4:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a12f8:	c5 dd fe d2          	vpaddd %ymm2,%ymm4,%ymm2
  1a12fc:	c4 81 79 c4 64 30 04 	vpinsrw $0x0,0x4(%r8,%r14,1),%xmm0,%xmm4
  1a1303:	00 
  1a1304:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a1308:	c4 e2 79 13 d4       	vcvtph2ps %xmm4,%xmm2
  1a130d:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  1a1311:	c4 e2 7d 18 d2       	vbroadcastss %xmm2,%ymm2
  1a1316:	c5 ec 59 d1          	vmulps %ymm1,%ymm2,%ymm2
  1a131a:	c5 fc 28 a4 24 e0 01 	vmovaps 0x1e0(%rsp),%ymm4
  1a1321:	00 00 
  1a1323:	c4 e2 7d b8 e2       	vfmadd231ps %ymm2,%ymm0,%ymm4
  1a1328:	c5 fc 29 a4 24 e0 01 	vmovaps %ymm4,0x1e0(%rsp)
  1a132f:	00 00 
  1a1331:	c4 81 7a 7e 44 30 20 	vmovq  0x20(%r8,%r14,1),%xmm0
  1a1338:	c4 e2 7d 58 d0       	vpbroadcastd %xmm0,%ymm2
  1a133d:	c4 e2 6d 08 94 24 60 	vpsignb 0x360(%rsp),%ymm2,%ymm2
  1a1344:	03 00 00 
  1a1347:	c4 e2 0d 04 d2       	vpmaddubsw %ymm2,%ymm14,%ymm2
  1a134c:	c5 f9 70 c0 55       	vpshufd $0x55,%xmm0,%xmm0
  1a1351:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  1a1356:	c4 c2 7d 08 c5       	vpsignb %ymm13,%ymm0,%ymm0
  1a135b:	c5 fd 6f a4 24 a0 02 	vmovdqa 0x2a0(%rsp),%ymm4
  1a1362:	00 00 
  1a1364:	c4 e2 5d 04 c0       	vpmaddubsw %ymm0,%ymm4,%ymm0
  1a1369:	c4 81 7a 7e 64 30 40 	vmovq  0x40(%r8,%r14,1),%xmm4
  1a1370:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a1374:	c5 9d f5 c0          	vpmaddwd %ymm0,%ymm12,%ymm0
  1a1378:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a137c:	c4 e2 7d 58 d4       	vpbroadcastd %xmm4,%ymm2
  1a1381:	c4 e2 6d 08 94 24 a0 	vpsignb 0xa0(%rsp),%ymm2,%ymm2
  1a1388:	00 00 00 
  1a138b:	c4 e2 05 04 d2       	vpmaddubsw %ymm2,%ymm15,%ymm2
  1a1390:	c5 f9 70 e4 55       	vpshufd $0x55,%xmm4,%xmm4
  1a1395:	c4 e2 7d 58 e4       	vpbroadcastd %xmm4,%ymm4
  1a139a:	c4 e2 5d 08 a4 24 c0 	vpsignb 0xc0(%rsp),%ymm4,%ymm4
  1a13a1:	00 00 00 
  1a13a4:	c5 7d 6f 84 24 e0 00 	vmovdqa 0xe0(%rsp),%ymm8
  1a13ab:	00 00 
  1a13ad:	c4 e2 3d 04 e4       	vpmaddubsw %ymm4,%ymm8,%ymm4
  1a13b2:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a13b6:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a13ba:	c5 ed fe d4          	vpaddd %ymm4,%ymm2,%ymm2
  1a13be:	c4 81 7a 7e 64 30 60 	vmovq  0x60(%r8,%r14,1),%xmm4
  1a13c5:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a13c9:	c4 e2 7d 58 d4       	vpbroadcastd %xmm4,%ymm2
  1a13ce:	c4 e2 6d 08 94 24 00 	vpsignb 0x100(%rsp),%ymm2,%ymm2
  1a13d5:	01 00 00 
  1a13d8:	c4 e2 4d 04 d2       	vpmaddubsw %ymm2,%ymm6,%ymm2
  1a13dd:	c5 f9 70 e4 55       	vpshufd $0x55,%xmm4,%xmm4
  1a13e2:	c4 e2 7d 58 e4       	vpbroadcastd %xmm4,%ymm4
  1a13e7:	c4 e2 5d 08 e7       	vpsignb %ymm7,%ymm4,%ymm4
  1a13ec:	c4 e2 35 04 e4       	vpmaddubsw %ymm4,%ymm9,%ymm4
  1a13f1:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a13f5:	c5 9d f5 e4          	vpmaddwd %ymm4,%ymm12,%ymm4
  1a13f9:	c5 ed fe d4          	vpaddd %ymm4,%ymm2,%ymm2
  1a13fd:	c4 81 7a 7e a4 30 80 	vmovq  0x80(%r8,%r14,1),%xmm4
  1a1404:	00 00 00 
  1a1407:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a140b:	c4 e2 7d 58 d4       	vpbroadcastd %xmm4,%ymm2
  1a1410:	c4 e2 6d 08 d5       	vpsignb %ymm5,%ymm2,%ymm2
  1a1415:	c4 e2 65 04 d2       	vpmaddubsw %ymm2,%ymm3,%ymm2
  1a141a:	c5 f9 70 e4 55       	vpshufd $0x55,%xmm4,%xmm4
  1a141f:	c4 e2 7d 58 e4       	vpbroadcastd %xmm4,%ymm4
  1a1424:	c4 c2 5d 08 e2       	vpsignb %ymm10,%ymm4,%ymm4
  1a1429:	c5 7d 6f 15 4f 3e e9 	vmovdqa -0x16c1b1(%rip),%ymm10        # 35280 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  1a1430:	ff 
  1a1431:	c4 e2 25 04 dc       	vpmaddubsw %ymm4,%ymm11,%ymm3
  1a1436:	c5 9d f5 d2          	vpmaddwd %ymm2,%ymm12,%ymm2
  1a143a:	c5 9d f5 db          	vpmaddwd %ymm3,%ymm12,%ymm3
  1a143e:	c5 ed fe d3          	vpaddd %ymm3,%ymm2,%ymm2
  1a1442:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
  1a1446:	c4 81 79 c4 54 30 06 	vpinsrw $0x0,0x6(%r8,%r14,1),%xmm0,%xmm2
  1a144d:	00 
  1a144e:	c4 e2 79 13 d2       	vcvtph2ps %xmm2,%xmm2
  1a1453:	c4 e2 7d 18 d2       	vbroadcastss %xmm2,%ymm2
  1a1458:	c5 ec 59 c9          	vmulps %ymm1,%ymm2,%ymm1
  1a145c:	c4 e2 7d 78 15 bb 70 	vpbroadcastb -0x168f45(%rip),%ymm2        # 38520 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x368>
  1a1463:	e9 ff 
  1a1465:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  1a1469:	c5 fc 28 5c 24 40    	vmovaps 0x40(%rsp),%ymm3
  1a146f:	c4 e2 7d b8 d9       	vfmadd231ps %ymm1,%ymm0,%ymm3
  1a1474:	c5 fc 29 5c 24 40    	vmovaps %ymm3,0x40(%rsp)
  1a147a:	c5 fc 28 4c 24 40    	vmovaps 0x40(%rsp),%ymm1
  1a1480:	49 81 c6 88 00 00 00 	add    $0x88,%r14
  1a1487:	49 ff cf             	dec    %r15
  1a148a:	0f 85 c0 f8 ff ff    	jne    1a0d50 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x10c0>
  1a1490:	49 89 de             	mov    %rbx,%r14
  1a1493:	49 c1 e6 05          	shl    $0x5,%r14
  1a1497:	4c 03 74 24 10       	add    0x10(%rsp),%r14
  1a149c:	c4 e3 7d 04 84 24 a0 	vpermilps $0xd8,0x1a0(%rsp),%ymm0
  1a14a3:	01 00 00 d8 
  1a14a7:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a14ad:	c4 c1 7c 11 04 8e    	vmovups %ymm0,(%r14,%rcx,4)
  1a14b3:	c4 e3 7d 04 84 24 c0 	vpermilps $0xd8,0x1c0(%rsp),%ymm0
  1a14ba:	01 00 00 d8 
  1a14be:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a14c4:	c4 c1 7c 11 04 b6    	vmovups %ymm0,(%r14,%rsi,4)
  1a14ca:	c4 e3 7d 04 84 24 e0 	vpermilps $0xd8,0x1e0(%rsp),%ymm0
  1a14d1:	01 00 00 d8 
  1a14d5:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a14db:	c4 81 7c 11 04 8e    	vmovups %ymm0,(%r14,%r9,4)
  1a14e1:	c5 f4 c6 c1 d8       	vshufps $0xd8,%ymm1,%ymm1,%ymm0
  1a14e6:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  1a14ec:	c4 81 7c 11 04 96    	vmovups %ymm0,(%r14,%r10,4)
  1a14f2:	48 ff c3             	inc    %rbx
  1a14f5:	49 01 c3             	add    %rax,%r11
  1a14f8:	4c 39 e3             	cmp    %r12,%rbx
  1a14fb:	0f 85 1f f8 ff ff    	jne    1a0d20 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1090>
  1a1501:	49 ff c5             	inc    %r13
  1a1504:	49 01 c0             	add    %rax,%r8
  1a1507:	4c 3b 6c 24 18       	cmp    0x18(%rsp),%r13
  1a150c:	0f 85 ce f7 ff ff    	jne    1a0ce0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1050>
  1a1512:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  1a1516:	5b                   	pop    %rbx
  1a1517:	41 5c                	pop    %r12
  1a1519:	41 5d                	pop    %r13
  1a151b:	41 5e                	pop    %r14
  1a151d:	41 5f                	pop    %r15
  1a151f:	5d                   	pop    %rbp
  1a1520:	c5 f8 77             	vzeroupper
  1a1523:	c3                   	ret
  1a1524:	4c 89 e0             	mov    %r12,%rax
  1a1527:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
  1a152b:	4c 89 e9             	mov    %r13,%rcx
  1a152e:	48 0f af ca          	imul   %rdx,%rcx
  1a1532:	48 c1 e1 04          	shl    $0x4,%rcx
  1a1536:	48 89 d6             	mov    %rdx,%rsi
  1a1539:	48 c1 e6 04          	shl    $0x4,%rsi
  1a153d:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
  1a1542:	4a 8d 3c ad 01 00 00 	lea    0x1(,%r13,4),%rdi
  1a1549:	00 
  1a154a:	48 0f af fa          	imul   %rdx,%rdi
  1a154e:	4e 8d 04 ad 02 00 00 	lea    0x2(,%r13,4),%r8
  1a1555:	00 
  1a1556:	4c 0f af c2          	imul   %rdx,%r8
  1a155a:	4e 8d 0c ad 03 00 00 	lea    0x3(,%r13,4),%r9
  1a1561:	00 
  1a1562:	4c 0f af ca          	imul   %rdx,%r9
  1a1566:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a156a:	4c 8b 54 24 10       	mov    0x10(%rsp),%r10
  1a156f:	eb 7c                	jmp    1a15ed <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x195d>
  1a1571:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1a1578:	0f 1f 84 00 00 00 00 
  1a157f:	00 
  1a1580:	45 31 db             	xor    %r11d,%r11d
  1a1583:	4a 8d 1c ad 00 00 00 	lea    0x0(,%r13,4),%rbx
  1a158a:	00 
  1a158b:	48 0f af da          	imul   %rdx,%rbx
  1a158f:	4e 8d 34 ad 01 00 00 	lea    0x1(,%r13,4),%r14
  1a1596:	00 
  1a1597:	4c 0f af f2          	imul   %rdx,%r14
  1a159b:	4e 8d 3c ad 02 00 00 	lea    0x2(,%r13,4),%r15
  1a15a2:	00 
  1a15a3:	4c 0f af fa          	imul   %rdx,%r15
  1a15a7:	4c 89 e6             	mov    %r12,%rsi
  1a15aa:	4e 8d 24 ad 03 00 00 	lea    0x3(,%r13,4),%r12
  1a15b1:	00 
  1a15b2:	4c 0f af e2          	imul   %rdx,%r12
  1a15b6:	49 c1 e3 05          	shl    $0x5,%r11
  1a15ba:	4c 03 5c 24 10       	add    0x10(%rsp),%r11
  1a15bf:	c4 c1 7c 11 04 9b    	vmovups %ymm0,(%r11,%rbx,4)
  1a15c5:	c4 81 7c 11 04 b3    	vmovups %ymm0,(%r11,%r14,4)
  1a15cb:	c4 81 7c 11 04 bb    	vmovups %ymm0,(%r11,%r15,4)
  1a15d1:	c4 81 7c 11 04 a3    	vmovups %ymm0,(%r11,%r12,4)
  1a15d7:	49 89 f4             	mov    %rsi,%r12
  1a15da:	49 ff c5             	inc    %r13
  1a15dd:	4c 03 54 24 40       	add    0x40(%rsp),%r10
  1a15e2:	4c 3b 6c 24 18       	cmp    0x18(%rsp),%r13
  1a15e7:	0f 84 25 ff ff ff    	je     1a1512 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1882>
  1a15ed:	49 83 fc 01          	cmp    $0x1,%r12
  1a15f1:	74 8d                	je     1a1580 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x18f0>
  1a15f3:	4c 89 d3             	mov    %r10,%rbx
  1a15f6:	45 31 db             	xor    %r11d,%r11d
  1a15f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  1a1600:	c5 fc 11 04 0b       	vmovups %ymm0,(%rbx,%rcx,1)
  1a1605:	c5 fc 11 04 bb       	vmovups %ymm0,(%rbx,%rdi,4)
  1a160a:	c4 a1 7c 11 04 83    	vmovups %ymm0,(%rbx,%r8,4)
  1a1610:	c4 a1 7c 11 04 8b    	vmovups %ymm0,(%rbx,%r9,4)
  1a1616:	c5 fc 11 44 0b 20    	vmovups %ymm0,0x20(%rbx,%rcx,1)
  1a161c:	c5 fc 11 44 bb 20    	vmovups %ymm0,0x20(%rbx,%rdi,4)
  1a1622:	c4 a1 7c 11 44 83 20 	vmovups %ymm0,0x20(%rbx,%r8,4)
  1a1629:	c4 a1 7c 11 44 8b 20 	vmovups %ymm0,0x20(%rbx,%r9,4)
  1a1630:	49 83 c3 02          	add    $0x2,%r11
  1a1634:	48 83 c3 40          	add    $0x40,%rbx
  1a1638:	4c 39 d8             	cmp    %r11,%rax
  1a163b:	75 c3                	jne    1a1600 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x1970>
  1a163d:	41 f6 c4 01          	test   $0x1,%r12b
  1a1641:	0f 85 3c ff ff ff    	jne    1a1583 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x18f3>
  1a1647:	eb 91                	jmp    1a15da <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x868gemm_vex+0x194a>
  1a1649:	cc                   	int3
  1a164a:	cc                   	int3
  1a164b:	cc                   	int3
  1a164c:	cc                   	int3
  1a164d:	cc                   	int3
  1a164e:	cc                   	int3
  1a164f:	cc                   	int3
