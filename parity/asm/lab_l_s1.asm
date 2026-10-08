000000000019cb30 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s1_sc_u1>:
  19cb30:	41 57                	push   %r15
  19cb32:	41 56                	push   %r14
  19cb34:	41 55                	push   %r13
  19cb36:	41 54                	push   %r12
  19cb38:	53                   	push   %rbx
  19cb39:	49 c1 e8 03          	shr    $0x3,%r8
  19cb3d:	0f 84 9d 02 00 00    	je     19cde0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s1_sc_u1+0x2b0>
  19cb43:	48 c1 ef 05          	shr    $0x5,%rdi
  19cb47:	0f 84 a0 02 00 00    	je     19cded <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s1_sc_u1+0x2bd>
  19cb4d:	48 89 f8             	mov    %rdi,%rax
  19cb50:	48 c1 e0 07          	shl    $0x7,%rax
  19cb54:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  19cb58:	45 31 d2             	xor    %r10d,%r10d
  19cb5b:	c5 fd 6f 05 bd 7f e9 	vmovdqa -0x168043(%rip),%ymm0        # 34b20 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x700>
  19cb62:	ff 
  19cb63:	c5 fd 6f 0d 15 88 e9 	vmovdqa -0x1677eb(%rip),%ymm1        # 35380 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  19cb6a:	ff 
  19cb6b:	4c 8d 0d 42 e7 ea ff 	lea    -0x1518be(%rip),%r9        # 4b2b4 <_RNvNtCs96HZWesffxA_4ggml8quants_k9IQ3S_GRID+0x1cac>
  19cb72:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  19cb79:	1f 84 00 00 00 00 00 
  19cb80:	45 31 db             	xor    %r11d,%r11d
  19cb83:	48 89 fb             	mov    %rdi,%rbx
  19cb86:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  19cb8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  19cb90:	c4 a1 7e 6f 5c 9a 28 	vmovdqu 0x28(%rdx,%r11,4),%ymm3
  19cb97:	c4 a1 7e 6f 64 9a 08 	vmovdqu 0x8(%rdx,%r11,4),%ymm4
  19cb9e:	c4 21 7e 6f 4c 9a 48 	vmovdqu 0x48(%rdx,%r11,4),%ymm9
  19cba5:	c4 21 7e 6f 54 9a 68 	vmovdqu 0x68(%rdx,%r11,4),%ymm10
  19cbac:	c5 dd db e8          	vpand  %ymm0,%ymm4,%ymm5
  19cbb0:	c4 62 75 00 dd       	vpshufb %ymm5,%ymm1,%ymm11
  19cbb5:	c5 e5 db e8          	vpand  %ymm0,%ymm3,%ymm5
  19cbb9:	c4 62 75 00 e5       	vpshufb %ymm5,%ymm1,%ymm12
  19cbbe:	c5 b5 db e8          	vpand  %ymm0,%ymm9,%ymm5
  19cbc2:	c4 e2 75 00 fd       	vpshufb %ymm5,%ymm1,%ymm7
  19cbc7:	c5 ad db e8          	vpand  %ymm0,%ymm10,%ymm5
  19cbcb:	c4 62 75 00 c5       	vpshufb %ymm5,%ymm1,%ymm8
  19cbd0:	c5 dd 71 d4 04       	vpsrlw $0x4,%ymm4,%ymm4
  19cbd5:	c5 dd db e0          	vpand  %ymm0,%ymm4,%ymm4
  19cbd9:	c4 e2 75 00 f4       	vpshufb %ymm4,%ymm1,%ymm6
  19cbde:	c5 e5 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm3
  19cbe3:	c5 e5 db d8          	vpand  %ymm0,%ymm3,%ymm3
  19cbe7:	c4 e2 75 00 eb       	vpshufb %ymm3,%ymm1,%ymm5
  19cbec:	c4 c1 65 71 d1 04    	vpsrlw $0x4,%ymm9,%ymm3
  19cbf2:	c5 e5 db d8          	vpand  %ymm0,%ymm3,%ymm3
  19cbf6:	c4 c1 35 71 d2 04    	vpsrlw $0x4,%ymm10,%ymm9
  19cbfc:	c4 e2 75 00 e3       	vpshufb %ymm3,%ymm1,%ymm4
  19cc01:	c5 35 db c8          	vpand  %ymm0,%ymm9,%ymm9
  19cc05:	c4 a2 7d 58 5c 19 02 	vpbroadcastd 0x2(%rcx,%r11,1),%ymm3
  19cc0c:	c4 41 7d 70 d4 a0    	vpshufd $0xa0,%ymm12,%ymm10
  19cc12:	c4 43 25 02 d2 aa    	vpblendd $0xaa,%ymm10,%ymm11,%ymm10
  19cc18:	c4 42 65 08 ea       	vpsignb %ymm10,%ymm3,%ymm13
  19cc1d:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
  19cc21:	c4 42 2d 08 d2       	vpsignb %ymm10,%ymm10,%ymm10
  19cc26:	c4 41 7d 70 db f5    	vpshufd $0xf5,%ymm11,%ymm11
  19cc2c:	c4 43 25 02 dc aa    	vpblendd $0xaa,%ymm12,%ymm11,%ymm11
  19cc32:	c4 42 25 08 e3       	vpsignb %ymm11,%ymm11,%ymm12
  19cc37:	62 d2 2d 28 50 dd    	vpdpbusd %ymm13,%ymm10,%ymm3
  19cc3d:	c4 22 7d 58 54 19 06 	vpbroadcastd 0x6(%rcx,%r11,1),%ymm10
  19cc44:	c4 42 2d 08 d3       	vpsignb %ymm11,%ymm10,%ymm10
  19cc49:	62 d2 1d 28 50 da    	vpdpbusd %ymm10,%ymm12,%ymm3
  19cc4f:	c4 41 7d 70 d0 a0    	vpshufd $0xa0,%ymm8,%ymm10
  19cc55:	c4 22 7d 58 5c 19 0a 	vpbroadcastd 0xa(%rcx,%r11,1),%ymm11
  19cc5c:	c4 43 45 02 d2 aa    	vpblendd $0xaa,%ymm10,%ymm7,%ymm10
  19cc62:	c4 42 2d 08 e2       	vpsignb %ymm10,%ymm10,%ymm12
  19cc67:	c4 42 25 08 d2       	vpsignb %ymm10,%ymm11,%ymm10
  19cc6c:	c5 fd 70 ff f5       	vpshufd $0xf5,%ymm7,%ymm7
  19cc71:	62 d2 1d 28 50 da    	vpdpbusd %ymm10,%ymm12,%ymm3
  19cc77:	c4 c3 45 02 f8 aa    	vpblendd $0xaa,%ymm8,%ymm7,%ymm7
  19cc7d:	c4 22 7d 58 44 19 0e 	vpbroadcastd 0xe(%rcx,%r11,1),%ymm8
  19cc84:	c4 62 45 08 d7       	vpsignb %ymm7,%ymm7,%ymm10
  19cc89:	c4 e2 3d 08 ff       	vpsignb %ymm7,%ymm8,%ymm7
  19cc8e:	c5 7d 70 c5 a0       	vpshufd $0xa0,%ymm5,%ymm8
  19cc93:	c4 43 4d 02 c0 aa    	vpblendd $0xaa,%ymm8,%ymm6,%ymm8
  19cc99:	62 f2 2d 28 50 df    	vpdpbusd %ymm7,%ymm10,%ymm3
  19cc9f:	c4 c2 3d 08 f8       	vpsignb %ymm8,%ymm8,%ymm7
  19cca4:	c4 22 7d 58 54 19 12 	vpbroadcastd 0x12(%rcx,%r11,1),%ymm10
  19ccab:	c4 42 2d 08 c0       	vpsignb %ymm8,%ymm10,%ymm8
  19ccb0:	62 d2 45 28 50 d8    	vpdpbusd %ymm8,%ymm7,%ymm3
  19ccb6:	c4 a2 7d 58 7c 19 16 	vpbroadcastd 0x16(%rcx,%r11,1),%ymm7
  19ccbd:	c4 42 75 00 c1       	vpshufb %ymm9,%ymm1,%ymm8
  19ccc2:	c5 7d 70 ce f5       	vpshufd $0xf5,%ymm6,%ymm9
  19ccc7:	c4 a2 7d 58 74 19 1a 	vpbroadcastd 0x1a(%rcx,%r11,1),%ymm6
  19ccce:	c4 63 35 02 cd aa    	vpblendd $0xaa,%ymm5,%ymm9,%ymm9
  19ccd4:	c4 41 7d 70 d0 a0    	vpshufd $0xa0,%ymm8,%ymm10
  19ccda:	c4 a2 7d 58 6c 19 1e 	vpbroadcastd 0x1e(%rcx,%r11,1),%ymm5
  19cce1:	c4 43 5d 02 d2 aa    	vpblendd $0xaa,%ymm10,%ymm4,%ymm10
  19cce7:	c5 fd 70 e4 f5       	vpshufd $0xf5,%ymm4,%ymm4
  19ccec:	c4 21 79 c4 1c 19 00 	vpinsrw $0x0,(%rcx,%r11,1),%xmm0,%xmm11
  19ccf3:	c4 c3 5d 02 e0 aa    	vpblendd $0xaa,%ymm8,%ymm4,%ymm4
  19ccf9:	c4 42 79 13 c3       	vcvtph2ps %xmm11,%xmm8
  19ccfe:	46 0f b6 74 9a 07    	movzbl 0x7(%rdx,%r11,4),%r14d
  19cd04:	c4 42 35 08 d9       	vpsignb %ymm9,%ymm9,%ymm11
  19cd09:	46 0f b6 7c 9a 03    	movzbl 0x3(%rdx,%r11,4),%r15d
  19cd0f:	46 0f b6 64 9a 06    	movzbl 0x6(%rdx,%r11,4),%r12d
  19cd15:	46 0f b6 6c 9a 02    	movzbl 0x2(%rdx,%r11,4),%r13d
  19cd1b:	c4 c2 45 08 f9       	vpsignb %ymm9,%ymm7,%ymm7
  19cd20:	c4 01 7a 10 0c a9    	vmovss (%r9,%r13,4),%xmm9
  19cd26:	c4 03 31 21 0c a1 10 	vinsertps $0x10,(%r9,%r12,4),%xmm9,%xmm9
  19cd2d:	c4 42 2d 08 e2       	vpsignb %ymm10,%ymm10,%ymm12
  19cd32:	46 0f b6 24 9a       	movzbl (%rdx,%r11,4),%r12d
  19cd37:	c4 03 31 21 0c b9 20 	vinsertps $0x20,(%r9,%r15,4),%xmm9,%xmm9
  19cd3e:	c4 c2 4d 08 f2       	vpsignb %ymm10,%ymm6,%ymm6
  19cd43:	46 0f b6 7c 9a 04    	movzbl 0x4(%rdx,%r11,4),%r15d
  19cd49:	c4 03 31 21 0c b1 30 	vinsertps $0x30,(%r9,%r14,4),%xmm9,%xmm9
  19cd50:	c4 62 5d 08 d4       	vpsignb %ymm4,%ymm4,%ymm10
  19cd55:	c4 01 7a 10 2c a1    	vmovss (%r9,%r12,4),%xmm13
  19cd5b:	c4 03 11 21 2c b9 10 	vinsertps $0x10,(%r9,%r15,4),%xmm13,%xmm13
  19cd62:	62 f2 25 28 50 df    	vpdpbusd %ymm7,%ymm11,%ymm3
  19cd68:	46 0f b6 74 9a 01    	movzbl 0x1(%rdx,%r11,4),%r14d
  19cd6e:	c4 83 11 21 3c b1 20 	vinsertps $0x20,(%r9,%r14,4),%xmm13,%xmm7
  19cd75:	62 f2 1d 28 50 de    	vpdpbusd %ymm6,%ymm12,%ymm3
  19cd7b:	46 0f b6 74 9a 05    	movzbl 0x5(%rdx,%r11,4),%r14d
  19cd81:	c4 83 41 21 34 b1 30 	vinsertps $0x30,(%r9,%r14,4),%xmm7,%xmm6
  19cd88:	c4 e2 55 08 e4       	vpsignb %ymm4,%ymm5,%ymm4
  19cd8d:	62 f2 2d 28 50 dc    	vpdpbusd %ymm4,%ymm10,%ymm3
  19cd93:	c4 c3 4d 18 e1 01    	vinsertf128 $0x1,%xmm9,%ymm6,%ymm4
  19cd99:	c4 c2 7d 18 e8       	vbroadcastss %xmm8,%ymm5
  19cd9e:	c5 fc 5b db          	vcvtdq2ps %ymm3,%ymm3
  19cda2:	c5 d4 59 e4          	vmulps %ymm4,%ymm5,%ymm4
  19cda6:	c4 e2 65 b8 d4       	vfmadd231ps %ymm4,%ymm3,%ymm2
  19cdab:	49 83 c3 22          	add    $0x22,%r11
  19cdaf:	48 ff cb             	dec    %rbx
  19cdb2:	0f 85 d8 fd ff ff    	jne    19cb90 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s1_sc_u1+0x60>
  19cdb8:	4d 8d 5a 01          	lea    0x1(%r10),%r11
  19cdbc:	c5 ec c6 d2 d8       	vshufps $0xd8,%ymm2,%ymm2,%ymm2
  19cdc1:	c4 e3 fd 01 d2 d8    	vpermpd $0xd8,%ymm2,%ymm2
  19cdc7:	49 c1 e2 05          	shl    $0x5,%r10
  19cdcb:	c4 a1 7c 11 14 16    	vmovups %ymm2,(%rsi,%r10,1)
  19cdd1:	48 01 c2             	add    %rax,%rdx
  19cdd4:	4d 89 da             	mov    %r11,%r10
  19cdd7:	4d 39 c3             	cmp    %r8,%r11
  19cdda:	0f 85 a0 fd ff ff    	jne    19cb80 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s1_sc_u1+0x50>
  19cde0:	5b                   	pop    %rbx
  19cde1:	41 5c                	pop    %r12
  19cde3:	41 5d                	pop    %r13
  19cde5:	41 5e                	pop    %r14
  19cde7:	41 5f                	pop    %r15
  19cde9:	c5 f8 77             	vzeroupper
  19cdec:	c3                   	ret
  19cded:	49 c1 e0 05          	shl    $0x5,%r8
  19cdf1:	48 89 f7             	mov    %rsi,%rdi
  19cdf4:	31 f6                	xor    %esi,%esi
  19cdf6:	4c 89 c2             	mov    %r8,%rdx
  19cdf9:	5b                   	pop    %rbx
  19cdfa:	41 5c                	pop    %r12
  19cdfc:	41 5d                	pop    %r13
  19cdfe:	41 5e                	pop    %r14
  19ce00:	41 5f                	pop    %r15
  19ce02:	ff 25 90 a3 18 00    	jmp    *0x18a390(%rip)        # 327198 <memset@GLIBC_2.2.5>
  19ce08:	cc                   	int3
  19ce09:	cc                   	int3
  19ce0a:	cc                   	int3
  19ce0b:	cc                   	int3
  19ce0c:	cc                   	int3
  19ce0d:	cc                   	int3
  19ce0e:	cc                   	int3
  19ce0f:	cc                   	int3

000000000019ce10 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u1>:
  19ce10:	55                   	push   %rbp
  19ce11:	41 57                	push   %r15
  19ce13:	41 56                	push   %r14
  19ce15:	41 55                	push   %r13
  19ce17:	41 54                	push   %r12
  19ce19:	53                   	push   %rbx
  19ce1a:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  19ce1f:	49 c1 e8 03          	shr    $0x3,%r8
  19ce23:	0f 84 a5 02 00 00    	je     19d0ce <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u1+0x2be>
  19ce29:	48 c1 ef 05          	shr    $0x5,%rdi
  19ce2d:	0f 84 a9 02 00 00    	je     19d0dc <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u1+0x2cc>
  19ce33:	48 89 f8             	mov    %rdi,%rax
  19ce36:	48 c1 e0 07          	shl    $0x7,%rax
  19ce3a:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  19ce3e:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  19ce43:	31 f6                	xor    %esi,%esi
  19ce45:	4c 8d 0d 68 e4 ea ff 	lea    -0x151b98(%rip),%r9        # 4b2b4 <_RNvNtCs96HZWesffxA_4ggml8quants_k9IQ3S_GRID+0x1cac>
  19ce4c:	c5 fd 6f 05 cc 7c e9 	vmovdqa -0x168334(%rip),%ymm0        # 34b20 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x700>
  19ce53:	ff 
  19ce54:	c5 fd 6f 0d 24 85 e9 	vmovdqa -0x167adc(%rip),%ymm1        # 35380 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  19ce5b:	ff 
  19ce5c:	0f 1f 40 00          	nopl   0x0(%rax)
  19ce60:	45 31 db             	xor    %r11d,%r11d
  19ce63:	48 89 fb             	mov    %rdi,%rbx
  19ce66:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  19ce6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  19ce70:	c4 a1 7e 6f 74 9a 08 	vmovdqu 0x8(%rdx,%r11,4),%ymm6
  19ce77:	c4 a1 7e 6f 6c 9a 28 	vmovdqu 0x28(%rdx,%r11,4),%ymm5
  19ce7e:	c4 a1 7e 6f 64 9a 48 	vmovdqu 0x48(%rdx,%r11,4),%ymm4
  19ce85:	c4 a1 7e 6f 5c 9a 68 	vmovdqu 0x68(%rdx,%r11,4),%ymm3
  19ce8c:	c5 cd db f8          	vpand  %ymm0,%ymm6,%ymm7
  19ce90:	c4 62 75 00 c7       	vpshufb %ymm7,%ymm1,%ymm8
  19ce95:	c5 d5 db f8          	vpand  %ymm0,%ymm5,%ymm7
  19ce99:	c4 62 75 00 cf       	vpshufb %ymm7,%ymm1,%ymm9
  19ce9e:	c5 5d db e0          	vpand  %ymm0,%ymm4,%ymm12
  19cea2:	c4 c1 7d 70 f9 a0    	vpshufd $0xa0,%ymm9,%ymm7
  19cea8:	c4 22 7d 58 54 19 02 	vpbroadcastd 0x2(%rcx,%r11,1),%ymm10
  19ceaf:	c4 e3 3d 02 ff aa    	vpblendd $0xaa,%ymm7,%ymm8,%ymm7
  19ceb5:	c4 62 2d 08 d7       	vpsignb %ymm7,%ymm10,%ymm10
  19ceba:	c4 62 45 08 df       	vpsignb %ymm7,%ymm7,%ymm11
  19cebf:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  19cec3:	62 d2 25 28 50 fa    	vpdpbusd %ymm10,%ymm11,%ymm7
  19cec9:	c4 42 75 00 d4       	vpshufb %ymm12,%ymm1,%ymm10
  19cece:	c5 65 db d8          	vpand  %ymm0,%ymm3,%ymm11
  19ced2:	c4 42 75 00 db       	vpshufb %ymm11,%ymm1,%ymm11
  19ced7:	c5 cd 71 d6 04       	vpsrlw $0x4,%ymm6,%ymm6
  19cedc:	c5 4d db e0          	vpand  %ymm0,%ymm6,%ymm12
  19cee0:	c4 c1 7d 70 f0 f5    	vpshufd $0xf5,%ymm8,%ymm6
  19cee6:	c4 c3 4d 02 f1 aa    	vpblendd $0xaa,%ymm9,%ymm6,%ymm6
  19ceec:	c4 22 7d 58 44 19 06 	vpbroadcastd 0x6(%rcx,%r11,1),%ymm8
  19cef3:	c4 62 3d 08 c6       	vpsignb %ymm6,%ymm8,%ymm8
  19cef8:	c4 62 4d 08 ce       	vpsignb %ymm6,%ymm6,%ymm9
  19cefd:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  19cf01:	62 d2 35 28 50 f0    	vpdpbusd %ymm8,%ymm9,%ymm6
  19cf07:	c4 41 7d 70 c3 a0    	vpshufd $0xa0,%ymm11,%ymm8
  19cf0d:	c4 43 2d 02 c0 aa    	vpblendd $0xaa,%ymm8,%ymm10,%ymm8
  19cf13:	c4 22 7d 58 4c 19 0a 	vpbroadcastd 0xa(%rcx,%r11,1),%ymm9
  19cf1a:	c4 42 35 08 c8       	vpsignb %ymm8,%ymm9,%ymm9
  19cf1f:	c4 42 3d 08 c0       	vpsignb %ymm8,%ymm8,%ymm8
  19cf24:	62 d2 3d 28 50 f9    	vpdpbusd %ymm9,%ymm8,%ymm7
  19cf2a:	c4 42 75 00 c4       	vpshufb %ymm12,%ymm1,%ymm8
  19cf2f:	c5 d5 71 d5 04       	vpsrlw $0x4,%ymm5,%ymm5
  19cf34:	c5 d5 db e8          	vpand  %ymm0,%ymm5,%ymm5
  19cf38:	c4 e2 75 00 ed       	vpshufb %ymm5,%ymm1,%ymm5
  19cf3d:	c5 dd 71 d4 04       	vpsrlw $0x4,%ymm4,%ymm4
  19cf42:	c5 dd db e0          	vpand  %ymm0,%ymm4,%ymm4
  19cf46:	c4 41 7d 70 ca f5    	vpshufd $0xf5,%ymm10,%ymm9
  19cf4c:	c4 43 35 02 cb aa    	vpblendd $0xaa,%ymm11,%ymm9,%ymm9
  19cf52:	c4 22 7d 58 54 19 0e 	vpbroadcastd 0xe(%rcx,%r11,1),%ymm10
  19cf59:	c4 42 2d 08 d1       	vpsignb %ymm9,%ymm10,%ymm10
  19cf5e:	c4 42 35 08 c9       	vpsignb %ymm9,%ymm9,%ymm9
  19cf63:	62 d2 35 28 50 f2    	vpdpbusd %ymm10,%ymm9,%ymm6
  19cf69:	c4 e2 75 00 e4       	vpshufb %ymm4,%ymm1,%ymm4
  19cf6e:	c5 e5 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm3
  19cf73:	c5 e5 db d8          	vpand  %ymm0,%ymm3,%ymm3
  19cf77:	c5 7d 70 cd a0       	vpshufd $0xa0,%ymm5,%ymm9
  19cf7c:	c4 43 3d 02 c9 aa    	vpblendd $0xaa,%ymm9,%ymm8,%ymm9
  19cf82:	c4 41 7d 70 c0 f5    	vpshufd $0xf5,%ymm8,%ymm8
  19cf88:	c4 e3 3d 02 ed aa    	vpblendd $0xaa,%ymm5,%ymm8,%ymm5
  19cf8e:	c4 22 7d 58 44 19 12 	vpbroadcastd 0x12(%rcx,%r11,1),%ymm8
  19cf95:	c4 42 3d 08 c1       	vpsignb %ymm9,%ymm8,%ymm8
  19cf9a:	c4 42 35 08 c9       	vpsignb %ymm9,%ymm9,%ymm9
  19cf9f:	62 d2 35 28 50 f8    	vpdpbusd %ymm8,%ymm9,%ymm7
  19cfa5:	c4 22 7d 58 44 19 16 	vpbroadcastd 0x16(%rcx,%r11,1),%ymm8
  19cfac:	c4 e2 75 00 db       	vpshufb %ymm3,%ymm1,%ymm3
  19cfb1:	c4 62 3d 08 c5       	vpsignb %ymm5,%ymm8,%ymm8
  19cfb6:	c4 e2 55 08 ed       	vpsignb %ymm5,%ymm5,%ymm5
  19cfbb:	62 d2 55 28 50 f0    	vpdpbusd %ymm8,%ymm5,%ymm6
  19cfc1:	c5 fd 70 eb a0       	vpshufd $0xa0,%ymm3,%ymm5
  19cfc6:	c4 e3 5d 02 ed aa    	vpblendd $0xaa,%ymm5,%ymm4,%ymm5
  19cfcc:	c4 22 7d 58 44 19 1a 	vpbroadcastd 0x1a(%rcx,%r11,1),%ymm8
  19cfd3:	c4 62 3d 08 c5       	vpsignb %ymm5,%ymm8,%ymm8
  19cfd8:	c4 e2 55 08 ed       	vpsignb %ymm5,%ymm5,%ymm5
  19cfdd:	62 d2 55 28 50 f8    	vpdpbusd %ymm8,%ymm5,%ymm7
  19cfe3:	c5 fd 70 e4 f5       	vpshufd $0xf5,%ymm4,%ymm4
  19cfe8:	c4 e3 5d 02 db aa    	vpblendd $0xaa,%ymm3,%ymm4,%ymm3
  19cfee:	c4 a2 7d 58 64 19 1e 	vpbroadcastd 0x1e(%rcx,%r11,1),%ymm4
  19cff5:	c4 e2 5d 08 e3       	vpsignb %ymm3,%ymm4,%ymm4
  19cffa:	c4 e2 65 08 db       	vpsignb %ymm3,%ymm3,%ymm3
  19cfff:	62 f2 65 28 50 f4    	vpdpbusd %ymm4,%ymm3,%ymm6
  19d005:	c4 a1 79 c4 1c 19 00 	vpinsrw $0x0,(%rcx,%r11,1),%xmm0,%xmm3
  19d00c:	46 0f b6 74 9a 07    	movzbl 0x7(%rdx,%r11,4),%r14d
  19d012:	46 0f b6 7c 9a 03    	movzbl 0x3(%rdx,%r11,4),%r15d
  19d018:	46 0f b6 64 9a 06    	movzbl 0x6(%rdx,%r11,4),%r12d
  19d01e:	46 0f b6 6c 9a 02    	movzbl 0x2(%rdx,%r11,4),%r13d
  19d024:	42 0f b6 6c 9a 05    	movzbl 0x5(%rdx,%r11,4),%ebp
  19d02a:	42 0f b6 44 9a 01    	movzbl 0x1(%rdx,%r11,4),%eax
  19d030:	c4 e2 79 13 db       	vcvtph2ps %xmm3,%xmm3
  19d035:	46 0f b6 14 9a       	movzbl (%rdx,%r11,4),%r10d
  19d03a:	c4 81 7a 10 24 a9    	vmovss (%r9,%r13,4),%xmm4
  19d040:	c4 83 59 21 24 a1 10 	vinsertps $0x10,(%r9,%r12,4),%xmm4,%xmm4
  19d047:	c4 83 59 21 24 b9 20 	vinsertps $0x20,(%r9,%r15,4),%xmm4,%xmm4
  19d04e:	c4 83 59 21 24 b1 30 	vinsertps $0x30,(%r9,%r14,4),%xmm4,%xmm4
  19d055:	46 0f b6 74 9a 04    	movzbl 0x4(%rdx,%r11,4),%r14d
  19d05b:	c4 81 7a 10 2c 91    	vmovss (%r9,%r10,4),%xmm5
  19d061:	c4 83 51 21 2c b1 10 	vinsertps $0x10,(%r9,%r14,4),%xmm5,%xmm5
  19d068:	c4 c3 51 21 2c 81 20 	vinsertps $0x20,(%r9,%rax,4),%xmm5,%xmm5
  19d06f:	c4 c3 51 21 2c a9 30 	vinsertps $0x30,(%r9,%rbp,4),%xmm5,%xmm5
  19d076:	c5 cd fe f7          	vpaddd %ymm7,%ymm6,%ymm6
  19d07a:	c4 e3 55 18 e4 01    	vinsertf128 $0x1,%xmm4,%ymm5,%ymm4
  19d080:	c4 e2 7d 18 db       	vbroadcastss %xmm3,%ymm3
  19d085:	c5 e4 59 dc          	vmulps %ymm4,%ymm3,%ymm3
  19d089:	c5 fc 5b e6          	vcvtdq2ps %ymm6,%ymm4
  19d08d:	c4 e2 65 b8 d4       	vfmadd231ps %ymm4,%ymm3,%ymm2
  19d092:	49 83 c3 22          	add    $0x22,%r11
  19d096:	48 ff cb             	dec    %rbx
  19d099:	0f 85 d1 fd ff ff    	jne    19ce70 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u1+0x60>
  19d09f:	48 8d 46 01          	lea    0x1(%rsi),%rax
  19d0a3:	c5 ec c6 d2 d8       	vshufps $0xd8,%ymm2,%ymm2,%ymm2
  19d0a8:	c4 e3 fd 01 d2 d8    	vpermpd $0xd8,%ymm2,%ymm2
  19d0ae:	48 c1 e6 05          	shl    $0x5,%rsi
  19d0b2:	4c 8b 54 24 f0       	mov    -0x10(%rsp),%r10
  19d0b7:	c4 c1 7c 11 14 32    	vmovups %ymm2,(%r10,%rsi,1)
  19d0bd:	48 03 54 24 f8       	add    -0x8(%rsp),%rdx
  19d0c2:	48 89 c6             	mov    %rax,%rsi
  19d0c5:	4c 39 c0             	cmp    %r8,%rax
  19d0c8:	0f 85 92 fd ff ff    	jne    19ce60 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u1+0x50>
  19d0ce:	5b                   	pop    %rbx
  19d0cf:	41 5c                	pop    %r12
  19d0d1:	41 5d                	pop    %r13
  19d0d3:	41 5e                	pop    %r14
  19d0d5:	41 5f                	pop    %r15
  19d0d7:	5d                   	pop    %rbp
  19d0d8:	c5 f8 77             	vzeroupper
  19d0db:	c3                   	ret
  19d0dc:	49 c1 e0 05          	shl    $0x5,%r8
  19d0e0:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  19d0e5:	31 f6                	xor    %esi,%esi
  19d0e7:	4c 89 c2             	mov    %r8,%rdx
  19d0ea:	5b                   	pop    %rbx
  19d0eb:	41 5c                	pop    %r12
  19d0ed:	41 5d                	pop    %r13
  19d0ef:	41 5e                	pop    %r14
  19d0f1:	41 5f                	pop    %r15
  19d0f3:	5d                   	pop    %rbp
  19d0f4:	ff 25 9e a0 18 00    	jmp    *0x18a09e(%rip)        # 327198 <memset@GLIBC_2.2.5>
  19d0fa:	cc                   	int3
  19d0fb:	cc                   	int3
  19d0fc:	cc                   	int3
  19d0fd:	cc                   	int3
  19d0fe:	cc                   	int3
  19d0ff:	cc                   	int3

000000000019d100 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2>:
  19d100:	49 c1 e8 03          	shr    $0x3,%r8
  19d104:	0f 84 31 08 00 00    	je     19d93b <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x83b>
  19d10a:	55                   	push   %rbp
  19d10b:	41 57                	push   %r15
  19d10d:	41 56                	push   %r14
  19d10f:	41 55                	push   %r13
  19d111:	41 54                	push   %r12
  19d113:	53                   	push   %rbx
  19d114:	48 c1 ef 05          	shr    $0x5,%rdi
  19d118:	48 89 f8             	mov    %rdi,%rax
  19d11b:	48 c1 e0 07          	shl    $0x7,%rax
  19d11f:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  19d123:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  19d128:	49 89 f9             	mov    %rdi,%r9
  19d12b:	49 83 e1 fe          	and    $0xfffffffffffffffe,%r9
  19d12f:	49 8d 41 ff          	lea    -0x1(%r9),%rax
  19d133:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
  19d137:	48 83 c0 02          	add    $0x2,%rax
  19d13b:	48 89 44 24 f0       	mov    %rax,-0x10(%rsp)
  19d140:	45 31 db             	xor    %r11d,%r11d
  19d143:	62 e1 fd 28 6f 0d d3 	vmovdqa64 -0x16862d(%rip),%ymm17        # 34b20 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x700>
  19d14a:	79 e9 ff 
  19d14d:	62 e1 fd 28 6f 25 29 	vmovdqa64 -0x167dd7(%rip),%ymm20        # 35380 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  19d154:	82 e9 ff 
  19d157:	48 8d 1d 56 e1 ea ff 	lea    -0x151eaa(%rip),%rbx        # 4b2b4 <_RNvNtCs96HZWesffxA_4ggml8quants_k9IQ3S_GRID+0x1cac>
  19d15e:	eb 2c                	jmp    19d18c <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x8c>
  19d160:	49 8d 43 01          	lea    0x1(%r11),%rax
  19d164:	62 b1 54 20 c6 c5 d8 	vshufps $0xd8,%ymm21,%ymm21,%ymm0
  19d16b:	c4 e3 fd 01 c0 d8    	vpermpd $0xd8,%ymm0,%ymm0
  19d171:	49 c1 e3 05          	shl    $0x5,%r11
  19d175:	c4 a1 7c 11 04 1e    	vmovups %ymm0,(%rsi,%r11,1)
  19d17b:	48 03 54 24 f8       	add    -0x8(%rsp),%rdx
  19d180:	49 89 c3             	mov    %rax,%r11
  19d183:	4c 39 c0             	cmp    %r8,%rax
  19d186:	0f 84 a5 07 00 00    	je     19d931 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x831>
  19d18c:	62 a1 54 00 57 ed    	vxorps %xmm21,%xmm21,%xmm21
  19d192:	4d 85 c9             	test   %r9,%r9
  19d195:	0f 84 35 05 00 00    	je     19d6d0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x5d0>
  19d19b:	45 31 f6             	xor    %r14d,%r14d
  19d19e:	45 31 ff             	xor    %r15d,%r15d
  19d1a1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  19d1a8:	0f 1f 84 00 00 00 00 
  19d1af:	00 
  19d1b0:	c4 a1 7e 6f 5c b2 08 	vmovdqu 0x8(%rdx,%r14,4),%ymm3
  19d1b7:	c4 a1 7e 6f 6c b2 28 	vmovdqu 0x28(%rdx,%r14,4),%ymm5
  19d1be:	c4 a1 7e 6f 74 b2 48 	vmovdqu 0x48(%rdx,%r14,4),%ymm6
  19d1c5:	c4 a1 7e 6f 7c b2 68 	vmovdqu 0x68(%rdx,%r14,4),%ymm7
  19d1cc:	62 b1 e5 28 db c1    	vpandq %ymm17,%ymm3,%ymm0
  19d1d2:	62 b1 d5 28 db c9    	vpandq %ymm17,%ymm5,%ymm1
  19d1d8:	62 f2 5d 20 00 d0    	vpshufb %ymm0,%ymm20,%ymm2
  19d1de:	62 f2 5d 20 00 c1    	vpshufb %ymm1,%ymm20,%ymm0
  19d1e4:	62 b1 cd 28 db c9    	vpandq %ymm17,%ymm6,%ymm1
  19d1ea:	62 b1 c5 28 db e1    	vpandq %ymm17,%ymm7,%ymm4
  19d1f0:	62 f2 5d 20 00 c9    	vpshufb %ymm1,%ymm20,%ymm1
  19d1f6:	62 f2 5d 20 00 e4    	vpshufb %ymm4,%ymm20,%ymm4
  19d1fc:	c5 e5 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm3
  19d201:	c5 d5 71 d5 04       	vpsrlw $0x4,%ymm5,%ymm5
  19d206:	62 b1 e5 28 db d9    	vpandq %ymm17,%ymm3,%ymm3
  19d20c:	62 31 d5 28 db c1    	vpandq %ymm17,%ymm5,%ymm8
  19d212:	c5 d5 71 d6 04       	vpsrlw $0x4,%ymm6,%ymm5
  19d217:	62 b1 d5 28 db f1    	vpandq %ymm17,%ymm5,%ymm6
  19d21d:	62 f2 5d 20 00 eb    	vpshufb %ymm3,%ymm20,%ymm5
  19d223:	c5 e5 71 d7 04       	vpsrlw $0x4,%ymm7,%ymm3
  19d228:	c4 22 7d 58 54 31 02 	vpbroadcastd 0x2(%rcx,%r14,1),%ymm10
  19d22f:	62 52 5d 20 00 f0    	vpshufb %ymm8,%ymm20,%ymm14
  19d235:	62 b1 e5 28 db d9    	vpandq %ymm17,%ymm3,%ymm3
  19d23b:	c4 22 7d 58 64 31 06 	vpbroadcastd 0x6(%rcx,%r14,1),%ymm12
  19d242:	62 72 5d 20 00 de    	vpshufb %ymm6,%ymm20,%ymm11
  19d248:	62 72 5d 20 00 cb    	vpshufb %ymm3,%ymm20,%ymm9
  19d24e:	c4 22 7d 58 6c 31 0a 	vpbroadcastd 0xa(%rcx,%r14,1),%ymm13
  19d255:	c5 fd 70 f0 a0       	vpshufd $0xa0,%ymm0,%ymm6
  19d25a:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
  19d25e:	c4 22 7d 58 44 31 0e 	vpbroadcastd 0xe(%rcx,%r14,1),%ymm8
  19d265:	c4 e3 6d 02 f6 aa    	vpblendd $0xaa,%ymm6,%ymm2,%ymm6
  19d26b:	c5 fd 70 d2 f5       	vpshufd $0xf5,%ymm2,%ymm2
  19d270:	62 22 7d 28 58 a4 31 	vpbroadcastd 0x12(%rcx,%r14,1),%ymm28
  19d277:	12 00 00 00 
  19d27b:	c4 e3 6d 02 f8 aa    	vpblendd $0xaa,%ymm0,%ymm2,%ymm7
  19d281:	c5 fd 70 c4 a0       	vpshufd $0xa0,%ymm4,%ymm0
  19d286:	62 22 7d 28 58 ac 31 	vpbroadcastd 0x16(%rcx,%r14,1),%ymm29
  19d28d:	16 00 00 00 
  19d291:	c4 e3 75 02 d0 aa    	vpblendd $0xaa,%ymm0,%ymm1,%ymm2
  19d297:	c5 fd 70 c9 f5       	vpshufd $0xf5,%ymm1,%ymm1
  19d29c:	62 22 7d 28 58 9c 31 	vpbroadcastd 0x1a(%rcx,%r14,1),%ymm27
  19d2a3:	1a 00 00 00 
  19d2a7:	c4 62 2d 08 fe       	vpsignb %ymm6,%ymm10,%ymm15
  19d2ac:	c4 63 75 02 d4 aa    	vpblendd $0xaa,%ymm4,%ymm1,%ymm10
  19d2b2:	62 22 7d 28 58 94 31 	vpbroadcastd 0x1e(%rcx,%r14,1),%ymm26
  19d2b9:	1e 00 00 00 
  19d2bd:	c4 e2 4d 08 ce       	vpsignb %ymm6,%ymm6,%ymm1
  19d2c2:	c5 d9 ef e4          	vpxor  %xmm4,%xmm4,%xmm4
  19d2c6:	c4 a1 79 c4 34 31 00 	vpinsrw $0x0,(%rcx,%r14,1),%xmm0,%xmm6
  19d2cd:	62 d2 75 28 50 df    	vpdpbusd %ymm15,%ymm1,%ymm3
  19d2d3:	c4 c1 7d 70 ce a0    	vpshufd $0xa0,%ymm14,%ymm1
  19d2d9:	62 e2 7d 08 13 c6    	vcvtph2ps %xmm6,%xmm16
  19d2df:	c4 63 55 02 f9 aa    	vpblendd $0xaa,%ymm1,%ymm5,%ymm15
  19d2e5:	c5 fd 70 ed f5       	vpshufd $0xf5,%ymm5,%ymm5
  19d2ea:	46 0f b6 64 b2 07    	movzbl 0x7(%rdx,%r14,4),%r12d
  19d2f0:	46 0f b6 6c b2 03    	movzbl 0x3(%rdx,%r14,4),%r13d
  19d2f6:	c4 c1 7d 70 c9 a0    	vpshufd $0xa0,%ymm9,%ymm1
  19d2fc:	42 0f b6 6c b2 06    	movzbl 0x6(%rdx,%r14,4),%ebp
  19d302:	46 0f b6 54 b2 02    	movzbl 0x2(%rdx,%r14,4),%r10d
  19d308:	42 0f b6 44 b2 05    	movzbl 0x5(%rdx,%r14,4),%eax
  19d30e:	c4 43 55 02 f6 aa    	vpblendd $0xaa,%ymm14,%ymm5,%ymm14
  19d314:	c4 a1 7a 10 2c 93    	vmovss (%rbx,%r10,4),%xmm5
  19d31a:	c4 e3 51 21 34 ab 10 	vinsertps $0x10,(%rbx,%rbp,4),%xmm5,%xmm6
  19d321:	c4 c1 7d 70 eb f5    	vpshufd $0xf5,%ymm11,%ymm5
  19d327:	46 0f b6 14 b2       	movzbl (%rdx,%r14,4),%r10d
  19d32c:	c4 a3 49 21 34 ab 20 	vinsertps $0x20,(%rbx,%r13,4),%xmm6,%xmm6
  19d333:	c4 e2 1d 08 c7       	vpsignb %ymm7,%ymm12,%ymm0
  19d338:	46 0f b6 6c b2 04    	movzbl 0x4(%rdx,%r14,4),%r13d
  19d33e:	62 a3 4d 08 21 14 a3 	vinsertps $0x30,(%rbx,%r12,4),%xmm6,%xmm18
  19d345:	30 
  19d346:	c4 e2 45 08 ff       	vpsignb %ymm7,%ymm7,%ymm7
  19d34b:	c4 a1 7a 10 34 93    	vmovss (%rbx,%r10,4),%xmm6
  19d351:	c4 a3 49 21 34 ab 10 	vinsertps $0x10,(%rbx,%r13,4),%xmm6,%xmm6
  19d358:	c4 62 15 08 e2       	vpsignb %ymm2,%ymm13,%ymm12
  19d35d:	46 0f b6 54 b2 01    	movzbl 0x1(%rdx,%r14,4),%r10d
  19d363:	c4 23 49 21 2c 93 20 	vinsertps $0x20,(%rbx,%r10,4),%xmm6,%xmm13
  19d36a:	c4 e2 6d 08 f2       	vpsignb %ymm2,%ymm2,%ymm6
  19d36f:	62 e3 15 08 21 1c 83 	vinsertps $0x30,(%rbx,%rax,4),%xmm13,%xmm19
  19d376:	30 
  19d377:	62 a2 7d 28 18 c0    	vbroadcastss %xmm16,%ymm16
  19d37d:	c4 42 3d 08 c2       	vpsignb %ymm10,%ymm8,%ymm8
  19d382:	62 a1 fe 28 6f b4 b2 	vmovdqu64 0x90(%rdx,%r14,4),%ymm22
  19d389:	90 00 00 00 
  19d38d:	62 a1 fe 28 6f bc b2 	vmovdqu64 0xb0(%rdx,%r14,4),%ymm23
  19d394:	b0 00 00 00 
  19d398:	62 21 fe 28 6f 84 b2 	vmovdqu64 0xd0(%rdx,%r14,4),%ymm24
  19d39f:	d0 00 00 00 
  19d3a3:	c4 e3 25 02 d1 aa    	vpblendd $0xaa,%ymm1,%ymm11,%ymm2
  19d3a9:	62 21 fe 28 6f 8c b2 	vmovdqu64 0xf0(%rdx,%r14,4),%ymm25
  19d3b0:	f0 00 00 00 
  19d3b4:	62 b1 cd 20 db c9    	vpandq %ymm17,%ymm22,%ymm1
  19d3ba:	62 72 5d 20 00 d9    	vpshufb %ymm1,%ymm20,%ymm11
  19d3c0:	62 f2 45 28 50 e0    	vpdpbusd %ymm0,%ymm7,%ymm4
  19d3c6:	62 b1 c5 20 db c1    	vpandq %ymm17,%ymm23,%ymm0
  19d3cc:	62 f2 5d 20 00 c0    	vpshufb %ymm0,%ymm20,%ymm0
  19d3d2:	c5 fd 70 c8 a0       	vpshufd $0xa0,%ymm0,%ymm1
  19d3d7:	c4 43 55 02 e9 aa    	vpblendd $0xaa,%ymm9,%ymm5,%ymm13
  19d3dd:	c4 e3 25 02 e9 aa    	vpblendd $0xaa,%ymm1,%ymm11,%ymm5
  19d3e3:	c4 a2 7d 58 7c 31 24 	vpbroadcastd 0x24(%rcx,%r14,1),%ymm7
  19d3ea:	c4 c2 2d 08 ca       	vpsignb %ymm10,%ymm10,%ymm1
  19d3ef:	c4 62 45 08 cd       	vpsignb %ymm5,%ymm7,%ymm9
  19d3f4:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  19d3f8:	c4 41 7d 70 d3 f5    	vpshufd $0xf5,%ymm11,%ymm10
  19d3fe:	c4 e2 55 08 ed       	vpsignb %ymm5,%ymm5,%ymm5
  19d403:	c4 e3 2d 02 c0 aa    	vpblendd $0xaa,%ymm0,%ymm10,%ymm0
  19d409:	c4 22 7d 58 54 31 28 	vpbroadcastd 0x28(%rcx,%r14,1),%ymm10
  19d410:	62 d2 55 28 50 f9    	vpdpbusd %ymm9,%ymm5,%ymm7
  19d416:	c4 e2 7d 08 e8       	vpsignb %ymm0,%ymm0,%ymm5
  19d41b:	c4 e2 2d 08 c0       	vpsignb %ymm0,%ymm10,%ymm0
  19d420:	c4 41 21 ef db       	vpxor  %xmm11,%xmm11,%xmm11
  19d425:	62 72 55 28 50 d8    	vpdpbusd %ymm0,%ymm5,%ymm11
  19d42b:	62 b1 bd 20 db c1    	vpandq %ymm17,%ymm24,%ymm0
  19d431:	62 72 5d 20 00 c8    	vpshufb %ymm0,%ymm20,%ymm9
  19d437:	62 b1 b5 20 db c1    	vpandq %ymm17,%ymm25,%ymm0
  19d43d:	62 91 fd 28 6f ec    	vmovdqa64 %ymm28,%ymm5
  19d443:	c4 c2 55 08 ef       	vpsignb %ymm15,%ymm5,%ymm5
  19d448:	62 61 fd 28 6f e5    	vmovdqa64 %ymm5,%ymm28
  19d44e:	62 f2 5d 20 00 c0    	vpshufb %ymm0,%ymm20,%ymm0
  19d454:	62 b1 55 28 71 d6 04 	vpsrlw $0x4,%ymm22,%ymm5
  19d45b:	62 b1 d5 28 db e9    	vpandq %ymm17,%ymm5,%ymm5
  19d461:	c4 42 05 08 d7       	vpsignb %ymm15,%ymm15,%ymm10
  19d466:	62 41 fd 28 6f f2    	vmovdqa64 %ymm10,%ymm30
  19d46c:	62 f2 5d 20 00 ed    	vpshufb %ymm5,%ymm20,%ymm5
  19d472:	62 b1 05 28 71 d7 04 	vpsrlw $0x4,%ymm23,%ymm15
  19d479:	62 31 85 28 db f9    	vpandq %ymm17,%ymm15,%ymm15
  19d47f:	62 11 fd 28 6f d5    	vmovdqa64 %ymm29,%ymm10
  19d485:	c4 42 2d 08 d6       	vpsignb %ymm14,%ymm10,%ymm10
  19d48a:	62 c1 fd 28 6f fa    	vmovdqa64 %ymm10,%ymm23
  19d490:	62 52 5d 20 00 ff    	vpshufb %ymm15,%ymm20,%ymm15
  19d496:	62 91 4d 20 71 d0 04 	vpsrlw $0x4,%ymm24,%ymm22
  19d49d:	62 a1 cd 20 db f1    	vpandq %ymm17,%ymm22,%ymm22
  19d4a3:	62 d2 4d 28 50 dc    	vpdpbusd %ymm12,%ymm6,%ymm3
  19d4a9:	62 b2 5d 20 00 f6    	vpshufb %ymm22,%ymm20,%ymm6
  19d4af:	62 91 1d 28 71 d1 04 	vpsrlw $0x4,%ymm25,%ymm12
  19d4b6:	62 31 9d 28 db e1    	vpandq %ymm17,%ymm12,%ymm12
  19d4bc:	62 d2 75 28 50 e0    	vpdpbusd %ymm8,%ymm1,%ymm4
  19d4c2:	c5 7d 70 c0 a0       	vpshufd $0xa0,%ymm0,%ymm8
  19d4c7:	c4 a2 7d 58 4c 31 2c 	vpbroadcastd 0x2c(%rcx,%r14,1),%ymm1
  19d4ce:	c4 42 0d 08 d6       	vpsignb %ymm14,%ymm14,%ymm10
  19d4d3:	c4 43 35 02 c0 aa    	vpblendd $0xaa,%ymm8,%ymm9,%ymm8
  19d4d9:	c4 41 7d 70 c9 f5    	vpshufd $0xf5,%ymm9,%ymm9
  19d4df:	c4 e3 35 02 c0 aa    	vpblendd $0xaa,%ymm0,%ymm9,%ymm0
  19d4e5:	c4 62 6d 08 ca       	vpsignb %ymm2,%ymm2,%ymm9
  19d4ea:	62 c1 fd 28 6f f1    	vmovdqa64 %ymm9,%ymm22
  19d4f0:	62 52 5d 20 00 f4    	vpshufb %ymm12,%ymm20,%ymm14
  19d4f6:	c4 42 75 08 e0       	vpsignb %ymm8,%ymm1,%ymm12
  19d4fb:	c4 42 3d 08 c0       	vpsignb %ymm8,%ymm8,%ymm8
  19d500:	62 91 fd 28 6f cb    	vmovdqa64 %ymm27,%ymm1
  19d506:	c4 e2 75 08 ca       	vpsignb %ymm2,%ymm1,%ymm1
  19d50b:	62 61 fd 28 6f c1    	vmovdqa64 %ymm1,%ymm24
  19d511:	c4 a2 7d 58 54 31 30 	vpbroadcastd 0x30(%rcx,%r14,1),%ymm2
  19d518:	c4 e2 6d 08 d0       	vpsignb %ymm0,%ymm2,%ymm2
  19d51d:	62 d2 3d 28 50 fc    	vpdpbusd %ymm12,%ymm8,%ymm7
  19d523:	c4 41 7d 70 c7 a0    	vpshufd $0xa0,%ymm15,%ymm8
  19d529:	c4 43 55 02 c0 aa    	vpblendd $0xaa,%ymm8,%ymm5,%ymm8
  19d52f:	c5 7d 70 e5 f5       	vpshufd $0xf5,%ymm5,%ymm12
  19d534:	62 91 fd 28 6f ca    	vmovdqa64 %ymm26,%ymm1
  19d53a:	c4 c2 75 08 ed       	vpsignb %ymm13,%ymm1,%ymm5
  19d53f:	c4 43 1d 02 ff aa    	vpblendd $0xaa,%ymm15,%ymm12,%ymm15
  19d545:	c4 22 7d 58 64 31 34 	vpbroadcastd 0x34(%rcx,%r14,1),%ymm12
  19d54c:	c4 42 15 08 ed       	vpsignb %ymm13,%ymm13,%ymm13
  19d551:	c4 e2 7d 08 c0       	vpsignb %ymm0,%ymm0,%ymm0
  19d556:	c4 42 1d 08 e0       	vpsignb %ymm8,%ymm12,%ymm12
  19d55b:	c4 c2 3d 08 c8       	vpsignb %ymm8,%ymm8,%ymm1
  19d560:	62 33 65 20 18 c2 01 	vinsertf32x4 $0x1,%xmm18,%ymm19,%ymm8
  19d567:	c4 22 7d 58 4c 31 38 	vpbroadcastd 0x38(%rcx,%r14,1),%ymm9
  19d56e:	c4 42 35 08 cf       	vpsignb %ymm15,%ymm9,%ymm9
  19d573:	62 72 7d 28 50 da    	vpdpbusd %ymm2,%ymm0,%ymm11
  19d579:	c4 c1 7d 70 c6 a0    	vpshufd $0xa0,%ymm14,%ymm0
  19d57f:	c4 e3 4d 02 c0 aa    	vpblendd $0xaa,%ymm0,%ymm6,%ymm0
  19d585:	c5 fd 70 d6 f5       	vpshufd $0xf5,%ymm6,%ymm2
  19d58a:	62 c1 7c 20 59 c0    	vmulps %ymm8,%ymm16,%ymm16
  19d590:	c4 43 6d 02 f6 aa    	vpblendd $0xaa,%ymm14,%ymm2,%ymm14
  19d596:	c4 a2 7d 58 54 31 3c 	vpbroadcastd 0x3c(%rcx,%r14,1),%ymm2
  19d59d:	62 92 0d 20 50 dc    	vpdpbusd %ymm28,%ymm30,%ymm3
  19d5a3:	c4 42 05 08 ff       	vpsignb %ymm15,%ymm15,%ymm15
  19d5a8:	c4 e2 6d 08 f0       	vpsignb %ymm0,%ymm2,%ymm6
  19d5ad:	c4 62 7d 08 c0       	vpsignb %ymm0,%ymm0,%ymm8
  19d5b2:	62 b2 2d 28 50 e7    	vpdpbusd %ymm23,%ymm10,%ymm4
  19d5b8:	c4 c2 0d 08 d6       	vpsignb %ymm14,%ymm14,%ymm2
  19d5bd:	c4 a2 7d 58 44 31 40 	vpbroadcastd 0x40(%rcx,%r14,1),%ymm0
  19d5c4:	62 d2 75 28 50 fc    	vpdpbusd %ymm12,%ymm1,%ymm7
  19d5ca:	c4 c2 7d 08 c6       	vpsignb %ymm14,%ymm0,%ymm0
  19d5cf:	c4 a1 79 c4 4c 31 22 	vpinsrw $0x0,0x22(%rcx,%r14,1),%xmm0,%xmm1
  19d5d6:	00 
  19d5d7:	62 52 05 28 50 d9    	vpdpbusd %ymm9,%ymm15,%ymm11
  19d5dd:	c4 e2 79 13 c9       	vcvtph2ps %xmm1,%xmm1
  19d5e2:	42 0f b6 84 b2 8f 00 	movzbl 0x8f(%rdx,%r14,4),%eax
  19d5e9:	00 00 
  19d5eb:	62 92 4d 20 50 d8    	vpdpbusd %ymm24,%ymm22,%ymm3
  19d5f1:	46 0f b6 94 b2 8b 00 	movzbl 0x8b(%rdx,%r14,4),%r10d
  19d5f8:	00 00 
  19d5fa:	46 0f b6 a4 b2 8e 00 	movzbl 0x8e(%rdx,%r14,4),%r12d
  19d601:	00 00 
  19d603:	46 0f b6 ac b2 8a 00 	movzbl 0x8a(%rdx,%r14,4),%r13d
  19d60a:	00 00 
  19d60c:	62 f2 15 28 50 e5    	vpdpbusd %ymm5,%ymm13,%ymm4
  19d612:	c4 a1 7a 10 2c ab    	vmovss (%rbx,%r13,4),%xmm5
  19d618:	c4 a3 51 21 2c a3 10 	vinsertps $0x10,(%rbx,%r12,4),%xmm5,%xmm5
  19d61f:	62 f2 3d 28 50 fe    	vpdpbusd %ymm6,%ymm8,%ymm7
  19d625:	46 0f b6 a4 b2 88 00 	movzbl 0x88(%rdx,%r14,4),%r12d
  19d62c:	00 00 
  19d62e:	c4 a3 51 21 2c 93 20 	vinsertps $0x20,(%rbx,%r10,4),%xmm5,%xmm5
  19d635:	62 72 6d 28 50 d8    	vpdpbusd %ymm0,%ymm2,%ymm11
  19d63b:	46 0f b6 94 b2 8c 00 	movzbl 0x8c(%rdx,%r14,4),%r10d
  19d642:	00 00 
  19d644:	c4 e3 51 21 04 83 30 	vinsertps $0x30,(%rbx,%rax,4),%xmm5,%xmm0
  19d64b:	c5 dd fe d3          	vpaddd %ymm3,%ymm4,%ymm2
  19d64f:	c4 a1 7a 10 1c a3    	vmovss (%rbx,%r12,4),%xmm3
  19d655:	c4 a3 61 21 1c 93 10 	vinsertps $0x10,(%rbx,%r10,4),%xmm3,%xmm3
  19d65c:	c5 a5 fe e7          	vpaddd %ymm7,%ymm11,%ymm4
  19d660:	42 0f b6 84 b2 89 00 	movzbl 0x89(%rdx,%r14,4),%eax
  19d667:	00 00 
  19d669:	c4 e3 61 21 1c 83 20 	vinsertps $0x20,(%rbx,%rax,4),%xmm3,%xmm3
  19d670:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  19d674:	42 0f b6 84 b2 8d 00 	movzbl 0x8d(%rdx,%r14,4),%eax
  19d67b:	00 00 
  19d67d:	c4 e3 61 21 1c 83 30 	vinsertps $0x30,(%rbx,%rax,4),%xmm3,%xmm3
  19d684:	62 a2 6d 28 a8 c5    	vfmadd213ps %ymm21,%ymm2,%ymm16
  19d68a:	c4 e3 65 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm3,%ymm0
  19d690:	c4 e2 7d 18 c9       	vbroadcastss %xmm1,%ymm1
  19d695:	62 e1 74 28 59 e8    	vmulps %ymm0,%ymm1,%ymm21
  19d69b:	c5 fc 5b c4          	vcvtdq2ps %ymm4,%ymm0
  19d69f:	62 a2 7d 28 a8 e8    	vfmadd213ps %ymm16,%ymm0,%ymm21
  19d6a5:	49 83 c7 02          	add    $0x2,%r15
  19d6a9:	49 83 c6 44          	add    $0x44,%r14
  19d6ad:	4d 39 cf             	cmp    %r9,%r15
  19d6b0:	0f 82 fa fa ff ff    	jb     19d1b0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0xb0>
  19d6b6:	4c 8b 7c 24 f0       	mov    -0x10(%rsp),%r15
  19d6bb:	49 89 fe             	mov    %rdi,%r14
  19d6be:	4d 29 fe             	sub    %r15,%r14
  19d6c1:	0f 86 99 fa ff ff    	jbe    19d160 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x60>
  19d6c7:	eb 16                	jmp    19d6df <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x5df>
  19d6c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  19d6d0:	45 31 ff             	xor    %r15d,%r15d
  19d6d3:	49 89 fe             	mov    %rdi,%r14
  19d6d6:	4d 29 fe             	sub    %r15,%r14
  19d6d9:	0f 86 81 fa ff ff    	jbe    19d160 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x60>
  19d6df:	4d 89 fc             	mov    %r15,%r12
  19d6e2:	49 c1 e4 05          	shl    $0x5,%r12
  19d6e6:	4f 8d 3c 7c          	lea    (%r12,%r15,2),%r15
  19d6ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  19d6f0:	c4 a1 7e 6f 44 ba 28 	vmovdqu 0x28(%rdx,%r15,4),%ymm0
  19d6f7:	c4 a1 7e 6f 4c ba 08 	vmovdqu 0x8(%rdx,%r15,4),%ymm1
  19d6fe:	c4 a1 7e 6f 5c ba 48 	vmovdqu 0x48(%rdx,%r15,4),%ymm3
  19d705:	c4 a1 7e 6f 74 ba 68 	vmovdqu 0x68(%rdx,%r15,4),%ymm6
  19d70c:	62 b1 f5 28 db d1    	vpandq %ymm17,%ymm1,%ymm2
  19d712:	62 f2 5d 20 00 fa    	vpshufb %ymm2,%ymm20,%ymm7
  19d718:	62 b1 fd 28 db d1    	vpandq %ymm17,%ymm0,%ymm2
  19d71e:	62 72 5d 20 00 c2    	vpshufb %ymm2,%ymm20,%ymm8
  19d724:	62 b1 e5 28 db d1    	vpandq %ymm17,%ymm3,%ymm2
  19d72a:	62 f2 5d 20 00 e2    	vpshufb %ymm2,%ymm20,%ymm4
  19d730:	62 b1 cd 28 db d1    	vpandq %ymm17,%ymm6,%ymm2
  19d736:	62 f2 5d 20 00 ea    	vpshufb %ymm2,%ymm20,%ymm5
  19d73c:	c5 f5 71 d1 04       	vpsrlw $0x4,%ymm1,%ymm1
  19d741:	62 b1 f5 28 db c9    	vpandq %ymm17,%ymm1,%ymm1
  19d747:	62 f2 5d 20 00 d1    	vpshufb %ymm1,%ymm20,%ymm2
  19d74d:	c5 fd 71 d0 04       	vpsrlw $0x4,%ymm0,%ymm0
  19d752:	62 b1 fd 28 db c1    	vpandq %ymm17,%ymm0,%ymm0
  19d758:	62 f2 5d 20 00 c8    	vpshufb %ymm0,%ymm20,%ymm1
  19d75e:	c5 fd 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm0
  19d763:	62 b1 fd 28 db c1    	vpandq %ymm17,%ymm0,%ymm0
  19d769:	c5 e5 71 d6 04       	vpsrlw $0x4,%ymm6,%ymm3
  19d76e:	62 f2 5d 20 00 c0    	vpshufb %ymm0,%ymm20,%ymm0
  19d774:	62 b1 e5 28 db f1    	vpandq %ymm17,%ymm3,%ymm6
  19d77a:	c4 a2 7d 58 5c 39 02 	vpbroadcastd 0x2(%rcx,%r15,1),%ymm3
  19d781:	c4 41 7d 70 c8 a0    	vpshufd $0xa0,%ymm8,%ymm9
  19d787:	c4 43 45 02 c9 aa    	vpblendd $0xaa,%ymm9,%ymm7,%ymm9
  19d78d:	c4 42 65 08 d1       	vpsignb %ymm9,%ymm3,%ymm10
  19d792:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
  19d796:	c4 42 35 08 c9       	vpsignb %ymm9,%ymm9,%ymm9
  19d79b:	c5 fd 70 ff f5       	vpshufd $0xf5,%ymm7,%ymm7
  19d7a0:	c4 c3 45 02 f8 aa    	vpblendd $0xaa,%ymm8,%ymm7,%ymm7
  19d7a6:	c4 62 45 08 c7       	vpsignb %ymm7,%ymm7,%ymm8
  19d7ab:	62 d2 35 28 50 da    	vpdpbusd %ymm10,%ymm9,%ymm3
  19d7b1:	c4 22 7d 58 4c 39 06 	vpbroadcastd 0x6(%rcx,%r15,1),%ymm9
  19d7b8:	c4 e2 35 08 ff       	vpsignb %ymm7,%ymm9,%ymm7
  19d7bd:	62 f2 3d 28 50 df    	vpdpbusd %ymm7,%ymm8,%ymm3
  19d7c3:	c5 fd 70 fd a0       	vpshufd $0xa0,%ymm5,%ymm7
  19d7c8:	c4 22 7d 58 44 39 0a 	vpbroadcastd 0xa(%rcx,%r15,1),%ymm8
  19d7cf:	c4 e3 5d 02 ff aa    	vpblendd $0xaa,%ymm7,%ymm4,%ymm7
  19d7d5:	c4 62 45 08 cf       	vpsignb %ymm7,%ymm7,%ymm9
  19d7da:	c4 e2 3d 08 ff       	vpsignb %ymm7,%ymm8,%ymm7
  19d7df:	c5 fd 70 e4 f5       	vpshufd $0xf5,%ymm4,%ymm4
  19d7e4:	62 f2 35 28 50 df    	vpdpbusd %ymm7,%ymm9,%ymm3
  19d7ea:	c4 e3 5d 02 e5 aa    	vpblendd $0xaa,%ymm5,%ymm4,%ymm4
  19d7f0:	c4 a2 7d 58 6c 39 0e 	vpbroadcastd 0xe(%rcx,%r15,1),%ymm5
  19d7f7:	c4 e2 5d 08 fc       	vpsignb %ymm4,%ymm4,%ymm7
  19d7fc:	c4 e2 55 08 e4       	vpsignb %ymm4,%ymm5,%ymm4
  19d801:	c5 fd 70 e9 a0       	vpshufd $0xa0,%ymm1,%ymm5
  19d806:	c4 e3 6d 02 ed aa    	vpblendd $0xaa,%ymm5,%ymm2,%ymm5
  19d80c:	62 f2 45 28 50 dc    	vpdpbusd %ymm4,%ymm7,%ymm3
  19d812:	c4 e2 55 08 e5       	vpsignb %ymm5,%ymm5,%ymm4
  19d817:	c4 a2 7d 58 7c 39 12 	vpbroadcastd 0x12(%rcx,%r15,1),%ymm7
  19d81e:	c4 e2 45 08 ed       	vpsignb %ymm5,%ymm7,%ymm5
  19d823:	62 f2 5d 28 50 dd    	vpdpbusd %ymm5,%ymm4,%ymm3
  19d829:	c4 a2 7d 58 64 39 16 	vpbroadcastd 0x16(%rcx,%r15,1),%ymm4
  19d830:	62 f2 5d 20 00 ee    	vpshufb %ymm6,%ymm20,%ymm5
  19d836:	c5 fd 70 f2 f5       	vpshufd $0xf5,%ymm2,%ymm6
  19d83b:	c4 a2 7d 58 54 39 1a 	vpbroadcastd 0x1a(%rcx,%r15,1),%ymm2
  19d842:	c4 e3 4d 02 f1 aa    	vpblendd $0xaa,%ymm1,%ymm6,%ymm6
  19d848:	c5 fd 70 fd a0       	vpshufd $0xa0,%ymm5,%ymm7
  19d84d:	c4 a2 7d 58 4c 39 1e 	vpbroadcastd 0x1e(%rcx,%r15,1),%ymm1
  19d854:	c4 e3 7d 02 ff aa    	vpblendd $0xaa,%ymm7,%ymm0,%ymm7
  19d85a:	c5 fd 70 c0 f5       	vpshufd $0xf5,%ymm0,%ymm0
  19d85f:	c4 21 79 c4 04 39 00 	vpinsrw $0x0,(%rcx,%r15,1),%xmm0,%xmm8
  19d866:	c4 e3 7d 02 c5 aa    	vpblendd $0xaa,%ymm5,%ymm0,%ymm0
  19d86c:	c4 c2 79 13 e8       	vcvtph2ps %xmm8,%xmm5
  19d871:	42 0f b6 44 ba 07    	movzbl 0x7(%rdx,%r15,4),%eax
  19d877:	c4 62 4d 08 c6       	vpsignb %ymm6,%ymm6,%ymm8
  19d87c:	46 0f b6 54 ba 03    	movzbl 0x3(%rdx,%r15,4),%r10d
  19d882:	46 0f b6 64 ba 06    	movzbl 0x6(%rdx,%r15,4),%r12d
  19d888:	46 0f b6 6c ba 02    	movzbl 0x2(%rdx,%r15,4),%r13d
  19d88e:	c4 e2 5d 08 e6       	vpsignb %ymm6,%ymm4,%ymm4
  19d893:	c4 a1 7a 10 34 ab    	vmovss (%rbx,%r13,4),%xmm6
  19d899:	c4 a3 49 21 34 a3 10 	vinsertps $0x10,(%rbx,%r12,4),%xmm6,%xmm6
  19d8a0:	c4 62 45 08 cf       	vpsignb %ymm7,%ymm7,%ymm9
  19d8a5:	46 0f b6 24 ba       	movzbl (%rdx,%r15,4),%r12d
  19d8aa:	c4 a3 49 21 34 93 20 	vinsertps $0x20,(%rbx,%r10,4),%xmm6,%xmm6
  19d8b1:	c4 e2 6d 08 d7       	vpsignb %ymm7,%ymm2,%ymm2
  19d8b6:	46 0f b6 54 ba 04    	movzbl 0x4(%rdx,%r15,4),%r10d
  19d8bc:	c4 e3 49 21 34 83 30 	vinsertps $0x30,(%rbx,%rax,4),%xmm6,%xmm6
  19d8c3:	c4 e2 7d 08 f8       	vpsignb %ymm0,%ymm0,%ymm7
  19d8c8:	c4 21 7a 10 14 a3    	vmovss (%rbx,%r12,4),%xmm10
  19d8ce:	c4 23 29 21 14 93 10 	vinsertps $0x10,(%rbx,%r10,4),%xmm10,%xmm10
  19d8d5:	62 f2 3d 28 50 dc    	vpdpbusd %ymm4,%ymm8,%ymm3
  19d8db:	42 0f b6 44 ba 01    	movzbl 0x1(%rdx,%r15,4),%eax
  19d8e1:	c4 e3 29 21 24 83 20 	vinsertps $0x20,(%rbx,%rax,4),%xmm10,%xmm4
  19d8e8:	62 f2 35 28 50 da    	vpdpbusd %ymm2,%ymm9,%ymm3
  19d8ee:	42 0f b6 44 ba 05    	movzbl 0x5(%rdx,%r15,4),%eax
  19d8f4:	c4 e3 59 21 14 83 30 	vinsertps $0x30,(%rbx,%rax,4),%xmm4,%xmm2
  19d8fb:	c4 e2 75 08 c0       	vpsignb %ymm0,%ymm1,%ymm0
  19d900:	62 f2 45 28 50 d8    	vpdpbusd %ymm0,%ymm7,%ymm3
  19d906:	c4 e3 6d 18 c6 01    	vinsertf128 $0x1,%xmm6,%ymm2,%ymm0
  19d90c:	c4 e2 7d 18 cd       	vbroadcastss %xmm5,%ymm1
  19d911:	c5 fc 5b d3          	vcvtdq2ps %ymm3,%ymm2
  19d915:	c5 f4 59 c0          	vmulps %ymm0,%ymm1,%ymm0
  19d919:	62 e2 6d 28 b8 e8    	vfmadd231ps %ymm0,%ymm2,%ymm21
  19d91f:	49 83 c7 22          	add    $0x22,%r15
  19d923:	49 ff ce             	dec    %r14
  19d926:	0f 85 c4 fd ff ff    	jne    19d6f0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x5f0>
  19d92c:	e9 2f f8 ff ff       	jmp    19d160 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s2_sc_u2+0x60>
  19d931:	5b                   	pop    %rbx
  19d932:	41 5c                	pop    %r12
  19d934:	41 5d                	pop    %r13
  19d936:	41 5e                	pop    %r14
  19d938:	41 5f                	pop    %r15
  19d93a:	5d                   	pop    %rbp
  19d93b:	c5 f8 77             	vzeroupper
  19d93e:	c3                   	ret
  19d93f:	cc                   	int3

000000000019d940 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s4_sc_u1>:
  19d940:	55                   	push   %rbp
  19d941:	41 57                	push   %r15
  19d943:	41 56                	push   %r14
  19d945:	41 55                	push   %r13
  19d947:	41 54                	push   %r12
  19d949:	53                   	push   %rbx
  19d94a:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  19d94f:	49 c1 e8 03          	shr    $0x3,%r8
  19d953:	0f 84 b8 02 00 00    	je     19dc11 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s4_sc_u1+0x2d1>
  19d959:	48 c1 ef 05          	shr    $0x5,%rdi
  19d95d:	0f 84 bc 02 00 00    	je     19dc1f <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s4_sc_u1+0x2df>
  19d963:	48 89 f8             	mov    %rdi,%rax
  19d966:	48 c1 e0 07          	shl    $0x7,%rax
  19d96a:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  19d96e:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  19d973:	45 31 c9             	xor    %r9d,%r9d
  19d976:	c5 fd 6f 05 a2 71 e9 	vmovdqa -0x168e5e(%rip),%ymm0        # 34b20 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x700>
  19d97d:	ff 
  19d97e:	c5 fd 6f 0d fa 79 e9 	vmovdqa -0x168606(%rip),%ymm1        # 35380 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  19d985:	ff 
  19d986:	4c 8d 15 27 d9 ea ff 	lea    -0x1526d9(%rip),%r10        # 4b2b4 <_RNvNtCs96HZWesffxA_4ggml8quants_k9IQ3S_GRID+0x1cac>
  19d98d:	0f 1f 00             	nopl   (%rax)
  19d990:	45 31 db             	xor    %r11d,%r11d
  19d993:	48 89 fb             	mov    %rdi,%rbx
  19d996:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  19d99a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  19d9a0:	c4 a1 7e 6f 6c 9a 08 	vmovdqu 0x8(%rdx,%r11,4),%ymm5
  19d9a7:	c4 a1 7e 6f 74 9a 28 	vmovdqu 0x28(%rdx,%r11,4),%ymm6
  19d9ae:	c4 a1 7e 6f 7c 9a 48 	vmovdqu 0x48(%rdx,%r11,4),%ymm7
  19d9b5:	c4 21 7e 6f 44 9a 68 	vmovdqu 0x68(%rdx,%r11,4),%ymm8
  19d9bc:	c5 d5 db d8          	vpand  %ymm0,%ymm5,%ymm3
  19d9c0:	c4 e2 75 00 db       	vpshufb %ymm3,%ymm1,%ymm3
  19d9c5:	c5 cd db e0          	vpand  %ymm0,%ymm6,%ymm4
  19d9c9:	c4 e2 75 00 e4       	vpshufb %ymm4,%ymm1,%ymm4
  19d9ce:	c5 7d 70 cb f5       	vpshufd $0xf5,%ymm3,%ymm9
  19d9d3:	c4 63 35 02 cc aa    	vpblendd $0xaa,%ymm4,%ymm9,%ymm9
  19d9d9:	c4 22 7d 58 54 19 06 	vpbroadcastd 0x6(%rcx,%r11,1),%ymm10
  19d9e0:	c4 42 35 08 d9       	vpsignb %ymm9,%ymm9,%ymm11
  19d9e5:	c4 42 2d 08 c9       	vpsignb %ymm9,%ymm10,%ymm9
  19d9ea:	c4 41 29 ef d2       	vpxor  %xmm10,%xmm10,%xmm10
  19d9ef:	62 52 25 28 50 d1    	vpdpbusd %ymm9,%ymm11,%ymm10
  19d9f5:	c5 d5 71 d5 04       	vpsrlw $0x4,%ymm5,%ymm5
  19d9fa:	c5 d5 db e8          	vpand  %ymm0,%ymm5,%ymm5
  19d9fe:	c4 e2 75 00 ed       	vpshufb %ymm5,%ymm1,%ymm5
  19da03:	c5 cd 71 d6 04       	vpsrlw $0x4,%ymm6,%ymm6
  19da08:	c5 cd db f0          	vpand  %ymm0,%ymm6,%ymm6
  19da0c:	c4 e2 75 00 f6       	vpshufb %ymm6,%ymm1,%ymm6
  19da11:	c5 7d 70 cd f5       	vpshufd $0xf5,%ymm5,%ymm9
  19da16:	c4 63 35 02 ce aa    	vpblendd $0xaa,%ymm6,%ymm9,%ymm9
  19da1c:	c4 42 35 08 d9       	vpsignb %ymm9,%ymm9,%ymm11
  19da21:	c4 22 7d 58 64 19 16 	vpbroadcastd 0x16(%rcx,%r11,1),%ymm12
  19da28:	c4 42 1d 08 c9       	vpsignb %ymm9,%ymm12,%ymm9
  19da2d:	62 52 25 28 50 d1    	vpdpbusd %ymm9,%ymm11,%ymm10
  19da33:	c5 45 db c8          	vpand  %ymm0,%ymm7,%ymm9
  19da37:	c4 42 75 00 c9       	vpshufb %ymm9,%ymm1,%ymm9
  19da3c:	c5 3d db d8          	vpand  %ymm0,%ymm8,%ymm11
  19da40:	c4 42 75 00 db       	vpshufb %ymm11,%ymm1,%ymm11
  19da45:	c4 41 7d 70 e3 a0    	vpshufd $0xa0,%ymm11,%ymm12
  19da4b:	c4 43 35 02 e4 aa    	vpblendd $0xaa,%ymm12,%ymm9,%ymm12
  19da51:	c4 42 1d 08 ec       	vpsignb %ymm12,%ymm12,%ymm13
  19da56:	c4 22 7d 58 74 19 0a 	vpbroadcastd 0xa(%rcx,%r11,1),%ymm14
  19da5d:	c4 42 0d 08 e4       	vpsignb %ymm12,%ymm14,%ymm12
  19da62:	c4 41 09 ef f6       	vpxor  %xmm14,%xmm14,%xmm14
  19da67:	62 52 15 28 50 f4    	vpdpbusd %ymm12,%ymm13,%ymm14
  19da6d:	c5 c5 71 d7 04       	vpsrlw $0x4,%ymm7,%ymm7
  19da72:	c5 c5 db f8          	vpand  %ymm0,%ymm7,%ymm7
  19da76:	c4 e2 75 00 ff       	vpshufb %ymm7,%ymm1,%ymm7
  19da7b:	c4 c1 3d 71 d0 04    	vpsrlw $0x4,%ymm8,%ymm8
  19da81:	c5 3d db c0          	vpand  %ymm0,%ymm8,%ymm8
  19da85:	c4 42 75 00 c0       	vpshufb %ymm8,%ymm1,%ymm8
  19da8a:	c4 41 7d 70 e0 a0    	vpshufd $0xa0,%ymm8,%ymm12
  19da90:	c4 43 45 02 e4 aa    	vpblendd $0xaa,%ymm12,%ymm7,%ymm12
  19da96:	c4 42 1d 08 ec       	vpsignb %ymm12,%ymm12,%ymm13
  19da9b:	c4 22 7d 58 7c 19 1a 	vpbroadcastd 0x1a(%rcx,%r11,1),%ymm15
  19daa2:	c4 42 05 08 e4       	vpsignb %ymm12,%ymm15,%ymm12
  19daa7:	62 52 15 28 50 f4    	vpdpbusd %ymm12,%ymm13,%ymm14
  19daad:	c4 41 0d fe d2       	vpaddd %ymm10,%ymm14,%ymm10
  19dab2:	c4 41 7d 70 c9 f5    	vpshufd $0xf5,%ymm9,%ymm9
  19dab8:	c4 43 35 02 cb aa    	vpblendd $0xaa,%ymm11,%ymm9,%ymm9
  19dabe:	c4 42 35 08 d9       	vpsignb %ymm9,%ymm9,%ymm11
  19dac3:	c4 22 7d 58 64 19 0e 	vpbroadcastd 0xe(%rcx,%r11,1),%ymm12
  19daca:	c4 42 1d 08 c9       	vpsignb %ymm9,%ymm12,%ymm9
  19dacf:	c4 41 19 ef e4       	vpxor  %xmm12,%xmm12,%xmm12
  19dad4:	62 52 25 28 50 e1    	vpdpbusd %ymm9,%ymm11,%ymm12
  19dada:	c5 fd 70 ff f5       	vpshufd $0xf5,%ymm7,%ymm7
  19dadf:	c4 c3 45 02 f8 aa    	vpblendd $0xaa,%ymm8,%ymm7,%ymm7
  19dae5:	c4 62 45 08 c7       	vpsignb %ymm7,%ymm7,%ymm8
  19daea:	c4 22 7d 58 4c 19 1e 	vpbroadcastd 0x1e(%rcx,%r11,1),%ymm9
  19daf1:	c4 e2 35 08 ff       	vpsignb %ymm7,%ymm9,%ymm7
  19daf6:	62 72 3d 28 50 e7    	vpdpbusd %ymm7,%ymm8,%ymm12
  19dafc:	c5 fd 70 e4 a0       	vpshufd $0xa0,%ymm4,%ymm4
  19db01:	c4 e3 65 02 dc aa    	vpblendd $0xaa,%ymm4,%ymm3,%ymm3
  19db07:	c4 e2 65 08 e3       	vpsignb %ymm3,%ymm3,%ymm4
  19db0c:	c4 a2 7d 58 7c 19 02 	vpbroadcastd 0x2(%rcx,%r11,1),%ymm7
  19db13:	c4 e2 45 08 db       	vpsignb %ymm3,%ymm7,%ymm3
  19db18:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  19db1c:	62 f2 5d 28 50 fb    	vpdpbusd %ymm3,%ymm4,%ymm7
  19db22:	c5 fd 70 de a0       	vpshufd $0xa0,%ymm6,%ymm3
  19db27:	c4 e3 55 02 db aa    	vpblendd $0xaa,%ymm3,%ymm5,%ymm3
  19db2d:	c4 e2 65 08 e3       	vpsignb %ymm3,%ymm3,%ymm4
  19db32:	c4 a2 7d 58 6c 19 12 	vpbroadcastd 0x12(%rcx,%r11,1),%ymm5
  19db39:	c4 e2 55 08 db       	vpsignb %ymm3,%ymm5,%ymm3
  19db3e:	62 f2 5d 28 50 fb    	vpdpbusd %ymm3,%ymm4,%ymm7
  19db44:	c5 9d fe df          	vpaddd %ymm7,%ymm12,%ymm3
  19db48:	c4 a1 79 c4 24 19 00 	vpinsrw $0x0,(%rcx,%r11,1),%xmm0,%xmm4
  19db4f:	c5 ad fe db          	vpaddd %ymm3,%ymm10,%ymm3
  19db53:	c4 e2 79 13 e4       	vcvtph2ps %xmm4,%xmm4
  19db58:	c5 fc 5b db          	vcvtdq2ps %ymm3,%ymm3
  19db5c:	46 0f b6 74 9a 07    	movzbl 0x7(%rdx,%r11,4),%r14d
  19db62:	46 0f b6 7c 9a 03    	movzbl 0x3(%rdx,%r11,4),%r15d
  19db68:	46 0f b6 64 9a 06    	movzbl 0x6(%rdx,%r11,4),%r12d
  19db6e:	46 0f b6 6c 9a 02    	movzbl 0x2(%rdx,%r11,4),%r13d
  19db74:	42 0f b6 6c 9a 05    	movzbl 0x5(%rdx,%r11,4),%ebp
  19db7a:	42 0f b6 44 9a 01    	movzbl 0x1(%rdx,%r11,4),%eax
  19db80:	42 0f b6 34 9a       	movzbl (%rdx,%r11,4),%esi
  19db85:	c4 81 7a 10 2c aa    	vmovss (%r10,%r13,4),%xmm5
  19db8b:	c4 83 51 21 2c a2 10 	vinsertps $0x10,(%r10,%r12,4),%xmm5,%xmm5
  19db92:	c4 83 51 21 2c ba 20 	vinsertps $0x20,(%r10,%r15,4),%xmm5,%xmm5
  19db99:	46 0f b6 7c 9a 04    	movzbl 0x4(%rdx,%r11,4),%r15d
  19db9f:	c4 83 51 21 2c b2 30 	vinsertps $0x30,(%r10,%r14,4),%xmm5,%xmm5
  19dba6:	c4 c1 7a 10 34 b2    	vmovss (%r10,%rsi,4),%xmm6
  19dbac:	c4 83 49 21 34 ba 10 	vinsertps $0x10,(%r10,%r15,4),%xmm6,%xmm6
  19dbb3:	c4 c3 49 21 34 82 20 	vinsertps $0x20,(%r10,%rax,4),%xmm6,%xmm6
  19dbba:	c4 c3 49 21 34 aa 30 	vinsertps $0x30,(%r10,%rbp,4),%xmm6,%xmm6
  19dbc1:	c4 e3 4d 18 ed 01    	vinsertf128 $0x1,%xmm5,%ymm6,%ymm5
  19dbc7:	c4 e2 7d 18 e4       	vbroadcastss %xmm4,%ymm4
  19dbcc:	c5 dc 59 e5          	vmulps %ymm5,%ymm4,%ymm4
  19dbd0:	c4 e2 65 b8 d4       	vfmadd231ps %ymm4,%ymm3,%ymm2
  19dbd5:	49 83 c3 22          	add    $0x22,%r11
  19dbd9:	48 ff cb             	dec    %rbx
  19dbdc:	0f 85 be fd ff ff    	jne    19d9a0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s4_sc_u1+0x60>
  19dbe2:	4d 8d 59 01          	lea    0x1(%r9),%r11
  19dbe6:	c5 ec c6 d2 d8       	vshufps $0xd8,%ymm2,%ymm2,%ymm2
  19dbeb:	c4 e3 fd 01 d2 d8    	vpermpd $0xd8,%ymm2,%ymm2
  19dbf1:	49 c1 e1 05          	shl    $0x5,%r9
  19dbf5:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  19dbfa:	c4 a1 7c 11 14 08    	vmovups %ymm2,(%rax,%r9,1)
  19dc00:	48 03 54 24 f8       	add    -0x8(%rsp),%rdx
  19dc05:	4d 89 d9             	mov    %r11,%r9
  19dc08:	4d 39 c3             	cmp    %r8,%r11
  19dc0b:	0f 85 7f fd ff ff    	jne    19d990 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_l_s4_sc_u1+0x50>
  19dc11:	5b                   	pop    %rbx
  19dc12:	41 5c                	pop    %r12
  19dc14:	41 5d                	pop    %r13
  19dc16:	41 5e                	pop    %r14
  19dc18:	41 5f                	pop    %r15
  19dc1a:	5d                   	pop    %rbp
  19dc1b:	c5 f8 77             	vzeroupper
  19dc1e:	c3                   	ret
  19dc1f:	49 c1 e0 05          	shl    $0x5,%r8
  19dc23:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  19dc28:	31 f6                	xor    %esi,%esi
  19dc2a:	4c 89 c2             	mov    %r8,%rdx
  19dc2d:	5b                   	pop    %rbx
  19dc2e:	41 5c                	pop    %r12
  19dc30:	41 5d                	pop    %r13
  19dc32:	41 5e                	pop    %r14
  19dc34:	41 5f                	pop    %r15
  19dc36:	5d                   	pop    %rbp
  19dc37:	ff 25 5b 95 18 00    	jmp    *0x18955b(%rip)        # 327198 <memset@GLIBC_2.2.5>
  19dc3d:	cc                   	int3
  19dc3e:	cc                   	int3
  19dc3f:	cc                   	int3

000000000019dc40 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_v_s1_sc_u1>:
  19dc40:	49 c1 e8 03          	shr    $0x3,%r8
  19dc44:	0f 84 8f 02 00 00    	je     19ded9 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_v_s1_sc_u1+0x299>
  19dc4a:	48 c1 ef 05          	shr    $0x5,%rdi
  19dc4e:	0f 84 89 02 00 00    	je     19dedd <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_v_s1_sc_u1+0x29d>
  19dc54:	48 89 f8             	mov    %rdi,%rax
  19dc57:	48 c1 e0 07          	shl    $0x7,%rax
  19dc5b:	48 8d 04 f8          	lea    (%rax,%rdi,8),%rax
  19dc5f:	45 31 c9             	xor    %r9d,%r9d
  19dc62:	c5 fd 6f 05 b6 6e e9 	vmovdqa -0x16914a(%rip),%ymm0        # 34b20 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x700>
  19dc69:	ff 
  19dc6a:	c5 fd 6f 0d 0e 77 e9 	vmovdqa -0x1688f2(%rip),%ymm1        # 35380 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x40>
  19dc71:	ff 
  19dc72:	62 e2 7d 28 58 05 e4 	vpbroadcastd -0x16861c(%rip),%ymm16        # 35660 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x320>
  19dc79:	79 e9 ff 
  19dc7c:	62 e1 fe 08 7e 0d 8a 	vmovq  -0x16ab76(%rip),%xmm17        # 33110 <anon.a243a2cefe40099c5384ecefc2bb7996.415.llvm.6308452637081725772+0x400>
  19dc83:	54 e9 ff 
  19dc86:	c4 e2 7d 58 25 81 7a 	vpbroadcastd -0x16857f(%rip),%ymm4        # 35710 <anon.07bf7b68542f29918c2fe01e2698508b.1589.llvm.13623703653819427234+0x3d0>
  19dc8d:	e9 ff 
  19dc8f:	c5 f9 6f 2d e9 67 e9 	vmovdqa -0x169817(%rip),%xmm5        # 34480 <anon.a243a2cefe40099c5384ecefc2bb7996.1.llvm.6308452637081725772+0x60>
  19dc96:	ff 
  19dc97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  19dc9e:	00 00 
  19dca0:	45 31 d2             	xor    %r10d,%r10d
  19dca3:	49 89 fb             	mov    %rdi,%r11
  19dca6:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
  19dcaa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  19dcb0:	c4 a1 7e 6f 7c 92 28 	vmovdqu 0x28(%rdx,%r10,4),%ymm7
  19dcb7:	c4 21 7e 6f 44 92 08 	vmovdqu 0x8(%rdx,%r10,4),%ymm8
  19dcbe:	c4 21 7e 6f 4c 92 48 	vmovdqu 0x48(%rdx,%r10,4),%ymm9
  19dcc5:	c4 21 7e 6f 74 92 68 	vmovdqu 0x68(%rdx,%r10,4),%ymm14
  19dccc:	c5 3d db d0          	vpand  %ymm0,%ymm8,%ymm10
  19dcd0:	c4 42 75 00 fa       	vpshufb %ymm10,%ymm1,%ymm15
  19dcd5:	c5 45 db d0          	vpand  %ymm0,%ymm7,%ymm10
  19dcd9:	c4 c2 75 00 d2       	vpshufb %ymm10,%ymm1,%ymm2
  19dcde:	c5 35 db d0          	vpand  %ymm0,%ymm9,%ymm10
  19dce2:	c4 42 75 00 ea       	vpshufb %ymm10,%ymm1,%ymm13
  19dce7:	c5 0d db d0          	vpand  %ymm0,%ymm14,%ymm10
  19dceb:	c4 42 75 00 e2       	vpshufb %ymm10,%ymm1,%ymm12
  19dcf0:	c4 c1 3d 71 d0 04    	vpsrlw $0x4,%ymm8,%ymm8
  19dcf6:	c5 3d db c0          	vpand  %ymm0,%ymm8,%ymm8
  19dcfa:	c4 42 75 00 d0       	vpshufb %ymm8,%ymm1,%ymm10
  19dcff:	c5 c5 71 d7 04       	vpsrlw $0x4,%ymm7,%ymm7
  19dd04:	c5 c5 db f8          	vpand  %ymm0,%ymm7,%ymm7
  19dd08:	c4 62 75 00 df       	vpshufb %ymm7,%ymm1,%ymm11
  19dd0d:	c4 c1 45 71 d1 04    	vpsrlw $0x4,%ymm9,%ymm7
  19dd13:	c5 c5 db f8          	vpand  %ymm0,%ymm7,%ymm7
  19dd17:	c4 c1 3d 71 d6 04    	vpsrlw $0x4,%ymm14,%ymm8
  19dd1d:	c4 62 75 00 cf       	vpshufb %ymm7,%ymm1,%ymm9
  19dd22:	c5 bd db f8          	vpand  %ymm0,%ymm8,%ymm7
  19dd26:	c4 62 75 00 c7       	vpshufb %ymm7,%ymm1,%ymm8
  19dd2b:	c5 fd 70 fa a0       	vpshufd $0xa0,%ymm2,%ymm7
  19dd30:	c4 e3 05 02 ff aa    	vpblendd $0xaa,%ymm7,%ymm15,%ymm7
  19dd36:	c4 62 45 08 f7       	vpsignb %ymm7,%ymm7,%ymm14
  19dd3b:	c4 a2 7d 58 5c 11 02 	vpbroadcastd 0x2(%rcx,%r10,1),%ymm3
  19dd42:	c4 e2 65 08 df       	vpsignb %ymm7,%ymm3,%ymm3
  19dd47:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  19dd4b:	c4 41 7d 70 ff f5    	vpshufd $0xf5,%ymm15,%ymm15
  19dd51:	c4 e3 05 02 d2 aa    	vpblendd $0xaa,%ymm2,%ymm15,%ymm2
  19dd57:	62 f2 0d 28 50 fb    	vpdpbusd %ymm3,%ymm14,%ymm7
  19dd5d:	c4 e2 6d 08 da       	vpsignb %ymm2,%ymm2,%ymm3
  19dd62:	c4 22 7d 58 74 11 06 	vpbroadcastd 0x6(%rcx,%r10,1),%ymm14
  19dd69:	c4 e2 0d 08 d2       	vpsignb %ymm2,%ymm14,%ymm2
  19dd6e:	c4 41 7d 70 f4 a0    	vpshufd $0xa0,%ymm12,%ymm14
  19dd74:	c4 43 15 02 f6 aa    	vpblendd $0xaa,%ymm14,%ymm13,%ymm14
  19dd7a:	c4 42 0d 08 fe       	vpsignb %ymm14,%ymm14,%ymm15
  19dd7f:	62 f2 65 28 50 fa    	vpdpbusd %ymm2,%ymm3,%ymm7
  19dd85:	c4 a2 7d 58 54 11 0a 	vpbroadcastd 0xa(%rcx,%r10,1),%ymm2
  19dd8c:	c4 c2 6d 08 d6       	vpsignb %ymm14,%ymm2,%ymm2
  19dd91:	62 f2 05 28 50 fa    	vpdpbusd %ymm2,%ymm15,%ymm7
  19dd97:	c4 c1 7d 70 d5 f5    	vpshufd $0xf5,%ymm13,%ymm2
  19dd9d:	c4 a2 7d 58 5c 11 0e 	vpbroadcastd 0xe(%rcx,%r10,1),%ymm3
  19dda4:	c4 c3 6d 02 d4 aa    	vpblendd $0xaa,%ymm12,%ymm2,%ymm2
  19ddaa:	c4 62 6d 08 e2       	vpsignb %ymm2,%ymm2,%ymm12
  19ddaf:	c4 e2 65 08 d2       	vpsignb %ymm2,%ymm3,%ymm2
  19ddb4:	c4 c1 7d 70 db a0    	vpshufd $0xa0,%ymm11,%ymm3
  19ddba:	62 f2 1d 28 50 fa    	vpdpbusd %ymm2,%ymm12,%ymm7
  19ddc0:	c4 e3 2d 02 d3 aa    	vpblendd $0xaa,%ymm3,%ymm10,%ymm2
  19ddc6:	c4 a2 7d 58 5c 11 12 	vpbroadcastd 0x12(%rcx,%r10,1),%ymm3
  19ddcd:	c4 62 6d 08 e2       	vpsignb %ymm2,%ymm2,%ymm12
  19ddd2:	c4 e2 65 08 d2       	vpsignb %ymm2,%ymm3,%ymm2
  19ddd7:	c4 c1 7d 70 da f5    	vpshufd $0xf5,%ymm10,%ymm3
  19dddd:	c4 c3 65 02 db aa    	vpblendd $0xaa,%ymm11,%ymm3,%ymm3
  19dde3:	62 f2 1d 28 50 fa    	vpdpbusd %ymm2,%ymm12,%ymm7
  19dde9:	c4 e2 65 08 d3       	vpsignb %ymm3,%ymm3,%ymm2
  19ddee:	c4 22 7d 58 54 11 16 	vpbroadcastd 0x16(%rcx,%r10,1),%ymm10
  19ddf5:	c4 e2 2d 08 db       	vpsignb %ymm3,%ymm10,%ymm3
  19ddfa:	62 f2 6d 28 50 fb    	vpdpbusd %ymm3,%ymm2,%ymm7
  19de00:	c4 a2 7d 58 54 11 1a 	vpbroadcastd 0x1a(%rcx,%r10,1),%ymm2
  19de07:	c4 c1 7d 70 d8 a0    	vpshufd $0xa0,%ymm8,%ymm3
  19de0d:	c4 e3 35 02 db aa    	vpblendd $0xaa,%ymm3,%ymm9,%ymm3
  19de13:	c4 22 7d 58 54 11 1e 	vpbroadcastd 0x1e(%rcx,%r10,1),%ymm10
  19de1a:	c4 62 65 08 db       	vpsignb %ymm3,%ymm3,%ymm11
  19de1f:	c4 41 7d 70 c9 f5    	vpshufd $0xf5,%ymm9,%ymm9
  19de25:	c4 21 79 c4 24 11 00 	vpinsrw $0x0,(%rcx,%r10,1),%xmm0,%xmm12
  19de2c:	c4 e2 6d 08 d3       	vpsignb %ymm3,%ymm2,%ymm2
  19de31:	c4 c3 35 02 d8 aa    	vpblendd $0xaa,%ymm8,%ymm9,%ymm3
  19de37:	c4 42 79 13 c4       	vcvtph2ps %xmm12,%xmm8
  19de3c:	62 f2 25 28 50 fa    	vpdpbusd %ymm2,%ymm11,%ymm7
  19de42:	c4 e2 65 08 d3       	vpsignb %ymm3,%ymm3,%ymm2
  19de47:	c4 21 7a 7e 0c 92    	vmovq  (%rdx,%r10,4),%xmm9
  19de4d:	62 32 35 08 00 c9    	vpshufb %xmm17,%xmm9,%xmm9
  19de53:	c4 e2 2d 08 db       	vpsignb %ymm3,%ymm10,%ymm3
  19de58:	c4 42 7d 31 d1       	vpmovzxbd %xmm9,%ymm10
  19de5d:	c4 c1 25 72 f2 17    	vpslld $0x17,%ymm10,%ymm11
  19de63:	62 31 25 28 fe d8    	vpaddd %ymm16,%ymm11,%ymm11
  19de69:	62 f2 6d 28 50 fb    	vpdpbusd %ymm3,%ymm2,%ymm7
  19de6f:	c4 c1 6d 72 f2 15    	vpslld $0x15,%ymm10,%ymm2
  19de75:	62 f3 35 08 3e c5 01 	vpcmpltub %xmm5,%xmm9,%k0
  19de7c:	62 f2 7e 28 38 d8    	vpmovm2d %k0,%ymm3
  19de82:	c5 ed fe d4          	vpaddd %ymm4,%ymm2,%ymm2
  19de86:	62 f2 7e 28 29 cb    	vpmovb2m %ymm3,%k1
  19de8c:	62 71 7f 29 6f da    	vmovdqu8 %ymm2,%ymm11{%k1}
  19de92:	c4 c2 7d 18 d0       	vbroadcastss %xmm8,%ymm2
  19de97:	c5 fc 5b df          	vcvtdq2ps %ymm7,%ymm3
  19de9b:	c5 a4 59 d2          	vmulps %ymm2,%ymm11,%ymm2
  19de9f:	c4 e2 65 b8 f2       	vfmadd231ps %ymm2,%ymm3,%ymm6
  19dea4:	49 83 c2 22          	add    $0x22,%r10
  19dea8:	49 ff cb             	dec    %r11
  19deab:	0f 85 ff fd ff ff    	jne    19dcb0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_v_s1_sc_u1+0x70>
  19deb1:	4d 8d 51 01          	lea    0x1(%r9),%r10
  19deb5:	c5 cc c6 d6 d8       	vshufps $0xd8,%ymm6,%ymm6,%ymm2
  19deba:	c4 e3 fd 01 d2 d8    	vpermpd $0xd8,%ymm2,%ymm2
  19dec0:	49 c1 e1 05          	shl    $0x5,%r9
  19dec4:	c4 a1 7c 11 14 0e    	vmovups %ymm2,(%rsi,%r9,1)
  19deca:	48 01 c2             	add    %rax,%rdx
  19decd:	4d 89 d1             	mov    %r10,%r9
  19ded0:	4d 39 c2             	cmp    %r8,%r10
  19ded3:	0f 85 c7 fd ff ff    	jne    19dca0 <_RNvNtNtCs96HZWesffxA_4ggml6repack14mxfp4_gemv_lab14lab_v_s1_sc_u1+0x60>
  19ded9:	c5 f8 77             	vzeroupper
  19dedc:	c3                   	ret
  19dedd:	49 c1 e0 05          	shl    $0x5,%r8
  19dee1:	48 89 f7             	mov    %rsi,%rdi
  19dee4:	31 f6                	xor    %esi,%esi
  19dee6:	4c 89 c2             	mov    %r8,%rdx
  19dee9:	ff 25 a9 92 18 00    	jmp    *0x1892a9(%rip)        # 327198 <memset@GLIBC_2.2.5>
  19deef:	cc                   	int3
