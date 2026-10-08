
target/release/deps/ggml-c87c5157cc8a9cd3:     file format elf64-x86-64


Disassembly of section .text:

00000000001a6890 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512>:
  1a6890:	55                   	push   %rbp
  1a6891:	41 57                	push   %r15
  1a6893:	41 56                	push   %r14
  1a6895:	41 55                	push   %r13
  1a6897:	41 54                	push   %r12
  1a6899:	53                   	push   %rbx
  1a689a:	48 81 ec 00 10 00 00 	sub    $0x1000,%rsp
  1a68a1:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
  1a68a8:	00 
  1a68a9:	48 81 ec 58 02 00 00 	sub    $0x258,%rsp
  1a68b0:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  1a68b5:	49 89 d5             	mov    %rdx,%r13
  1a68b8:	48 89 34 24          	mov    %rsi,(%rsp)
  1a68bc:	48 8b 84 24 90 12 00 	mov    0x1290(%rsp),%rax
  1a68c3:	00 
  1a68c4:	45 31 d2             	xor    %r10d,%r10d
  1a68c7:	4c 89 c9             	mov    %r9,%rcx
  1a68ca:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
  1a68ce:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
  1a68d3:	0f 84 95 2b 00 00    	je     1a946e <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bde>
  1a68d9:	48 89 c1             	mov    %rax,%rcx
  1a68dc:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
  1a68e0:	0f 84 88 2b 00 00    	je     1a946e <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bde>
  1a68e6:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  1a68eb:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
  1a68f0:	4c 89 4c 24 38       	mov    %r9,0x38(%rsp)
  1a68f5:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  1a68fa:	48 c1 e9 03          	shr    $0x3,%rcx
  1a68fe:	48 c1 e8 04          	shr    $0x4,%rax
  1a6902:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  1a6907:	48 29 c1             	sub    %rax,%rcx
  1a690a:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
  1a690f:	0f 84 2d 2b 00 00    	je     1a9442 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bb2>
  1a6915:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  1a691a:	4c 8b 5c 24 30       	mov    0x30(%rsp),%r11
  1a691f:	49 c1 eb 05          	shr    $0x5,%r11
  1a6923:	48 c1 6c 24 18 02    	shrq   $0x2,0x18(%rsp)
  1a6929:	4c 89 d8             	mov    %r11,%rax
  1a692c:	48 c1 e0 07          	shl    $0x7,%rax
  1a6930:	4a 8d 14 d8          	lea    (%rax,%r11,8),%rdx
  1a6934:	4d 85 db             	test   %r11,%r11
  1a6937:	4c 89 6c 24 48       	mov    %r13,0x48(%rsp)
  1a693c:	4c 89 5c 24 28       	mov    %r11,0x28(%rsp)
  1a6941:	48 89 54 24 40       	mov    %rdx,0x40(%rsp)
  1a6946:	0f 84 b8 20 00 00    	je     1a8a04 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2174>
  1a694c:	49 69 c3 98 01 00 00 	imul   $0x198,%r11,%rax
  1a6953:	4c 8b 54 24 10       	mov    0x10(%rsp),%r10
  1a6958:	4c 01 d0             	add    %r10,%rax
  1a695b:	49 69 cb 20 02 00 00 	imul   $0x220,%r11,%rcx
  1a6962:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
  1a6967:	49 69 cb 10 01 00 00 	imul   $0x110,%r11,%rcx
  1a696e:	48 89 8c 24 80 00 00 	mov    %rcx,0x80(%rsp)
  1a6975:	00 
  1a6976:	4d 8d 04 0a          	lea    (%r10,%rcx,1),%r8
  1a697a:	4d 8d 0c 12          	lea    (%r10,%rdx,1),%r9
  1a697e:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  1a6983:	48 01 d1             	add    %rdx,%rcx
  1a6986:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  1a698b:	31 ed                	xor    %ebp,%ebp
  1a698d:	62 71 fd 48 6f 05 a9 	vmovdqa64 -0x16da57(%rip),%zmm8        # 38f40 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x398>
  1a6994:	25 e9 ff 
  1a6997:	62 e1 fd 48 6f 0d df 	vmovdqa64 -0x16da21(%rip),%zmm17        # 38f80 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x3d8>
  1a699e:	25 e9 ff 
  1a69a1:	4c 8d 25 54 54 ea ff 	lea    -0x15abac(%rip),%r12        # 4bdfc <_RNvNtCs96HZWesffxA_4ggml8quants_k9IQ3S_GRID+0x1cac>
  1a69a8:	62 a1 5d 00 ef e4    	vpxord %xmm20,%xmm20,%xmm20
  1a69ae:	b1 03                	mov    $0x3,%cl
  1a69b0:	c5 fb 92 c9          	kmovd  %ecx,%k1
  1a69b4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  1a69bb:	00 00 00 00 00 
  1a69c0:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
  1a69c7:	00 
  1a69c8:	49 0f af cd          	imul   %r13,%rcx
  1a69cc:	48 89 8c 24 00 01 00 	mov    %rcx,0x100(%rsp)
  1a69d3:	00 
  1a69d4:	48 8d 0c ad 01 00 00 	lea    0x1(,%rbp,4),%rcx
  1a69db:	00 
  1a69dc:	49 0f af cd          	imul   %r13,%rcx
  1a69e0:	48 89 8c 24 f8 00 00 	mov    %rcx,0xf8(%rsp)
  1a69e7:	00 
  1a69e8:	48 8d 0c ad 02 00 00 	lea    0x2(,%rbp,4),%rcx
  1a69ef:	00 
  1a69f0:	49 0f af cd          	imul   %r13,%rcx
  1a69f4:	48 89 8c 24 f0 00 00 	mov    %rcx,0xf0(%rsp)
  1a69fb:	00 
  1a69fc:	48 8d 0c ad 03 00 00 	lea    0x3(,%rbp,4),%rcx
  1a6a03:	00 
  1a6a04:	49 0f af cd          	imul   %r13,%rcx
  1a6a08:	48 89 8c 24 e8 00 00 	mov    %rcx,0xe8(%rsp)
  1a6a0f:	00 
  1a6a10:	48 8d 0c ad 04 00 00 	lea    0x4(,%rbp,4),%rcx
  1a6a17:	00 
  1a6a18:	49 0f af cd          	imul   %r13,%rcx
  1a6a1c:	48 89 8c 24 e0 00 00 	mov    %rcx,0xe0(%rsp)
  1a6a23:	00 
  1a6a24:	48 8d 0c ad 05 00 00 	lea    0x5(,%rbp,4),%rcx
  1a6a2b:	00 
  1a6a2c:	49 0f af cd          	imul   %r13,%rcx
  1a6a30:	48 89 8c 24 d8 00 00 	mov    %rcx,0xd8(%rsp)
  1a6a37:	00 
  1a6a38:	48 8d 0c ad 06 00 00 	lea    0x6(,%rbp,4),%rcx
  1a6a3f:	00 
  1a6a40:	49 0f af cd          	imul   %r13,%rcx
  1a6a44:	48 89 8c 24 d0 00 00 	mov    %rcx,0xd0(%rsp)
  1a6a4b:	00 
  1a6a4c:	48 8d 0c ad 07 00 00 	lea    0x7(,%rbp,4),%rcx
  1a6a53:	00 
  1a6a54:	49 0f af cd          	imul   %r13,%rcx
  1a6a58:	48 89 8c 24 c8 00 00 	mov    %rcx,0xc8(%rsp)
  1a6a5f:	00 
  1a6a60:	48 8d 0c ad 08 00 00 	lea    0x8(,%rbp,4),%rcx
  1a6a67:	00 
  1a6a68:	49 0f af cd          	imul   %r13,%rcx
  1a6a6c:	48 89 8c 24 c0 00 00 	mov    %rcx,0xc0(%rsp)
  1a6a73:	00 
  1a6a74:	48 8d 0c ad 09 00 00 	lea    0x9(,%rbp,4),%rcx
  1a6a7b:	00 
  1a6a7c:	49 0f af cd          	imul   %r13,%rcx
  1a6a80:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
  1a6a87:	00 
  1a6a88:	48 8d 0c ad 0a 00 00 	lea    0xa(,%rbp,4),%rcx
  1a6a8f:	00 
  1a6a90:	49 0f af cd          	imul   %r13,%rcx
  1a6a94:	48 89 8c 24 b0 00 00 	mov    %rcx,0xb0(%rsp)
  1a6a9b:	00 
  1a6a9c:	48 8d 0c ad 0b 00 00 	lea    0xb(,%rbp,4),%rcx
  1a6aa3:	00 
  1a6aa4:	49 0f af cd          	imul   %r13,%rcx
  1a6aa8:	48 89 8c 24 a8 00 00 	mov    %rcx,0xa8(%rsp)
  1a6aaf:	00 
  1a6ab0:	48 8d 0c ad 0c 00 00 	lea    0xc(,%rbp,4),%rcx
  1a6ab7:	00 
  1a6ab8:	49 0f af cd          	imul   %r13,%rcx
  1a6abc:	48 89 8c 24 a0 00 00 	mov    %rcx,0xa0(%rsp)
  1a6ac3:	00 
  1a6ac4:	48 8d 0c ad 0d 00 00 	lea    0xd(,%rbp,4),%rcx
  1a6acb:	00 
  1a6acc:	49 0f af cd          	imul   %r13,%rcx
  1a6ad0:	48 89 8c 24 98 00 00 	mov    %rcx,0x98(%rsp)
  1a6ad7:	00 
  1a6ad8:	48 8d 0c ad 0e 00 00 	lea    0xe(,%rbp,4),%rcx
  1a6adf:	00 
  1a6ae0:	49 0f af cd          	imul   %r13,%rcx
  1a6ae4:	48 89 8c 24 90 00 00 	mov    %rcx,0x90(%rsp)
  1a6aeb:	00 
  1a6aec:	48 89 6c 24 78       	mov    %rbp,0x78(%rsp)
  1a6af1:	48 8d 0c ad 0f 00 00 	lea    0xf(,%rbp,4),%rcx
  1a6af8:	00 
  1a6af9:	49 0f af cd          	imul   %r13,%rcx
  1a6afd:	48 89 8c 24 88 00 00 	mov    %rcx,0x88(%rsp)
  1a6b04:	00 
  1a6b05:	4c 8b 6c 24 08       	mov    0x8(%rsp),%r13
  1a6b0a:	4c 8b 74 24 68       	mov    0x68(%rsp),%r14
  1a6b0f:	31 c9                	xor    %ecx,%ecx
  1a6b11:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  1a6b16:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  1a6b1d:	00 00 00 
  1a6b20:	48 89 94 24 08 01 00 	mov    %rdx,0x108(%rsp)
  1a6b27:	00 
  1a6b28:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a6b2c:	31 db                	xor    %ebx,%ebx
  1a6b2e:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
  1a6b32:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
  1a6b36:	c4 41 28 57 d2       	vxorps %xmm10,%xmm10,%xmm10
  1a6b3b:	c4 41 20 57 db       	vxorps %xmm11,%xmm11,%xmm11
  1a6b40:	c4 41 18 57 e4       	vxorps %xmm12,%xmm12,%xmm12
  1a6b45:	c4 41 10 57 ed       	vxorps %xmm13,%xmm13,%xmm13
  1a6b4a:	62 01 3c 00 57 c0    	vxorps %xmm24,%xmm24,%xmm24
  1a6b50:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  1a6b54:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x850(%rsp)
  1a6b5b:	50 08 00 00 
  1a6b5f:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x810(%rsp)
  1a6b66:	10 08 00 00 
  1a6b6a:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x7d0(%rsp)
  1a6b71:	d0 07 00 00 
  1a6b75:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x790(%rsp)
  1a6b7c:	90 07 00 00 
  1a6b80:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x710(%rsp)
  1a6b87:	10 07 00 00 
  1a6b8b:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x6d0(%rsp)
  1a6b92:	d0 06 00 00 
  1a6b96:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x690(%rsp)
  1a6b9d:	90 06 00 00 
  1a6ba1:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x750(%rsp)
  1a6ba8:	50 07 00 00 
  1a6bac:	0f 1f 40 00          	nopl   0x0(%rax)
  1a6bb0:	62 61 7c 48 11 84 24 	vmovups %zmm24,0x1110(%rsp)
  1a6bb7:	10 11 00 00 
  1a6bbb:	62 71 7c 48 11 ac 24 	vmovups %zmm13,0x1150(%rsp)
  1a6bc2:	50 11 00 00 
  1a6bc6:	62 71 7c 48 11 a4 24 	vmovups %zmm12,0x1190(%rsp)
  1a6bcd:	90 11 00 00 
  1a6bd1:	62 71 7c 48 11 9c 24 	vmovups %zmm11,0x11d0(%rsp)
  1a6bd8:	d0 11 00 00 
  1a6bdc:	62 71 7c 48 11 94 24 	vmovups %zmm10,0x1210(%rsp)
  1a6be3:	10 12 00 00 
  1a6be7:	62 f1 7c 48 11 a4 24 	vmovups %zmm4,0xa90(%rsp)
  1a6bee:	90 0a 00 00 
  1a6bf2:	62 f1 7c 48 11 bc 24 	vmovups %zmm7,0xad0(%rsp)
  1a6bf9:	d0 0a 00 00 
  1a6bfd:	62 f1 7c 48 11 84 24 	vmovups %zmm0,0xb10(%rsp)
  1a6c04:	10 0b 00 00 
  1a6c08:	62 d1 fe 48 6f 84 1e 	vmovdqu64 0x8(%r14,%rbx,1),%zmm0
  1a6c0f:	08 00 00 00 
  1a6c13:	62 d1 fe 48 6f 8c 1e 	vmovdqu64 0x48(%r14,%rbx,1),%zmm1
  1a6c1a:	48 00 00 00 
  1a6c1e:	62 d1 fe 48 6f 94 1d 	vmovdqu64 0x8(%r13,%rbx,1),%zmm2
  1a6c25:	08 00 00 00 
  1a6c29:	62 d1 fe 48 6f 9c 1d 	vmovdqu64 0x48(%r13,%rbx,1),%zmm3
  1a6c30:	48 00 00 00 
  1a6c34:	62 f3 ed 48 43 e8 88 	vshufi64x2 $0x88,%zmm0,%zmm2,%zmm5
  1a6c3b:	62 f3 e5 48 43 f9 88 	vshufi64x2 $0x88,%zmm1,%zmm3,%zmm7
  1a6c42:	62 f3 ed 48 43 c0 dd 	vshufi64x2 $0xdd,%zmm0,%zmm2,%zmm0
  1a6c49:	62 f3 e5 48 43 c9 dd 	vshufi64x2 $0xdd,%zmm1,%zmm3,%zmm1
  1a6c50:	62 d1 d5 48 db d0    	vpandq %zmm8,%zmm5,%zmm2
  1a6c56:	62 62 75 40 00 c2    	vpshufb %zmm2,%zmm17,%zmm24
  1a6c5c:	62 d1 c5 48 db d0    	vpandq %zmm8,%zmm7,%zmm2
  1a6c62:	62 f1 65 48 71 d5 04 	vpsrlw $0x4,%zmm5,%zmm3
  1a6c69:	62 f2 75 40 00 f2    	vpshufb %zmm2,%zmm17,%zmm6
  1a6c6f:	62 d1 e5 48 db d0    	vpandq %zmm8,%zmm3,%zmm2
  1a6c75:	62 f1 65 48 71 d7 04 	vpsrlw $0x4,%zmm7,%zmm3
  1a6c7c:	62 d1 fd 48 db e8    	vpandq %zmm8,%zmm0,%zmm5
  1a6c82:	62 d1 e5 48 db d8    	vpandq %zmm8,%zmm3,%zmm3
  1a6c88:	62 f2 75 40 00 e5    	vpshufb %zmm5,%zmm17,%zmm4
  1a6c8e:	62 f1 fe 48 7f a4 24 	vmovdqu64 %zmm4,0x1d0(%rsp)
  1a6c95:	d0 01 00 00 
  1a6c99:	62 d1 f5 48 db f8    	vpandq %zmm8,%zmm1,%zmm7
  1a6c9f:	62 f1 7d 48 71 d0 04 	vpsrlw $0x4,%zmm0,%zmm0
  1a6ca6:	62 f2 75 40 00 ff    	vpshufb %zmm7,%zmm17,%zmm7
  1a6cac:	62 51 fd 48 db c8    	vpandq %zmm8,%zmm0,%zmm9
  1a6cb2:	62 f1 7d 48 71 d1 04 	vpsrlw $0x4,%zmm1,%zmm0
  1a6cb9:	62 51 fd 48 db d8    	vpandq %zmm8,%zmm0,%zmm11
  1a6cbf:	62 72 75 40 00 c2    	vpshufb %zmm2,%zmm17,%zmm8
  1a6cc5:	c4 c1 7e 6f 64 1a 08 	vmovdqu 0x8(%r10,%rbx,1),%ymm4
  1a6ccc:	62 41 fe 28 6f ac 1a 	vmovdqu64 0x28(%r10,%rbx,1),%ymm29
  1a6cd3:	28 00 00 00 
  1a6cd7:	62 c1 fe 28 6f bc 1a 	vmovdqu64 0x48(%r10,%rbx,1),%ymm23
  1a6cde:	48 00 00 00 
  1a6ce2:	62 72 75 40 00 d3    	vpshufb %zmm3,%zmm17,%zmm10
  1a6ce8:	c4 c1 7e 6f 6c 1a 68 	vmovdqu 0x68(%r10,%rbx,1),%ymm5
  1a6cef:	c5 79 70 e5 a0       	vpshufd $0xa0,%xmm5,%xmm12
  1a6cf4:	62 53 9d 48 43 ec 00 	vshufi64x2 $0x0,%zmm12,%zmm12,%zmm13
  1a6cfb:	62 52 75 40 00 db    	vpshufb %zmm11,%zmm17,%zmm11
  1a6d01:	62 52 7d 48 1c e5    	vpabsb %zmm13,%zmm12
  1a6d07:	62 d2 7e 48 29 d5    	vpmovb2m %zmm13,%k2
  1a6d0d:	62 41 7d 48 70 e2 88 	vpshufd $0x88,%zmm10,%zmm28
  1a6d14:	62 31 7d 08 70 ef a0 	vpshufd $0xa0,%xmm23,%xmm13
  1a6d1b:	62 53 95 48 43 f5 00 	vshufi64x2 $0x0,%zmm13,%zmm13,%zmm14
  1a6d22:	62 52 7d 48 1c ee    	vpabsb %zmm14,%zmm13
  1a6d28:	62 01 5d 40 f8 cc    	vpsubb %zmm28,%zmm20,%zmm25
  1a6d2e:	62 d1 7d 48 70 cb 88 	vpshufd $0x88,%zmm11,%zmm1
  1a6d35:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0x2d0(%rsp)
  1a6d3c:	d0 02 00 00 
  1a6d40:	62 f1 5d 40 f8 c1    	vpsubb %zmm1,%zmm20,%zmm0
  1a6d46:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x110(%rsp)
  1a6d4d:	10 01 00 00 
  1a6d51:	62 12 1d 42 66 f9    	vpblendmb %zmm25,%zmm28,%zmm15{%k2}
  1a6d57:	62 e2 75 4a 66 c0    	vpblendmb %zmm0,%zmm1,%zmm16{%k2}
  1a6d5d:	62 d2 7e 48 29 d6    	vpmovb2m %zmm14,%k2
  1a6d63:	62 52 75 40 00 c9    	vpshufb %zmm9,%zmm17,%zmm9
  1a6d69:	62 41 7d 48 70 d8 88 	vpshufd $0x88,%zmm8,%zmm27
  1a6d70:	62 11 7d 08 70 f5 a0 	vpshufd $0xa0,%xmm29,%xmm14
  1a6d77:	62 53 8d 48 43 f6 00 	vshufi64x2 $0x0,%zmm14,%zmm14,%zmm14
  1a6d7e:	62 01 5d 40 f8 f3    	vpsubb %zmm27,%zmm20,%zmm30
  1a6d84:	62 c2 7d 48 1c ce    	vpabsb %zmm14,%zmm17
  1a6d8a:	62 41 7d 48 70 f9 88 	vpshufd $0x88,%zmm9,%zmm31
  1a6d91:	62 82 25 42 66 d6    	vpblendmb %zmm30,%zmm27,%zmm18{%k2}
  1a6d97:	62 91 5d 40 f8 c7    	vpsubb %zmm31,%zmm20,%zmm0
  1a6d9d:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x250(%rsp)
  1a6da4:	50 02 00 00 
  1a6da8:	62 e2 05 42 66 d8    	vpblendmb %zmm0,%zmm31,%zmm19{%k2}
  1a6dae:	62 d2 7e 48 29 d6    	vpmovb2m %zmm14,%k2
  1a6db4:	62 f1 7d 48 70 ce 88 	vpshufd $0x88,%zmm6,%zmm1
  1a6dbb:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0xa50(%rsp)
  1a6dc2:	50 0a 00 00 
  1a6dc6:	62 e1 5d 40 f8 f1    	vpsubb %zmm1,%zmm20,%zmm22
  1a6dcc:	c5 79 70 f4 a0       	vpshufd $0xa0,%xmm4,%xmm14
  1a6dd1:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  1a6dd5:	62 d2 1d 48 50 c7    	vpdpbusd %zmm15,%zmm12,%zmm0
  1a6ddb:	62 53 8d 48 43 f6 00 	vshufi64x2 $0x0,%zmm14,%zmm14,%zmm14
  1a6de2:	62 01 2d 00 ef d2    	vpxord %xmm26,%xmm26,%xmm26
  1a6de8:	62 32 75 4a 66 fe    	vpblendmb %zmm22,%zmm1,%zmm15{%k2}
  1a6dee:	62 22 1d 48 50 d0    	vpdpbusd %zmm16,%zmm12,%zmm26
  1a6df4:	62 f1 7d 48 70 cf 88 	vpshufd $0x88,%zmm7,%zmm1
  1a6dfb:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0x150(%rsp)
  1a6e02:	50 01 00 00 
  1a6e06:	62 f1 5d 40 f8 d1    	vpsubb %zmm1,%zmm20,%zmm2
  1a6e0c:	62 f1 fe 48 7f 94 24 	vmovdqu64 %zmm2,0x590(%rsp)
  1a6e13:	90 05 00 00 
  1a6e17:	62 b2 15 48 50 c2    	vpdpbusd %zmm18,%zmm13,%zmm0
  1a6e1d:	62 72 75 4a 66 e2    	vpblendmb %zmm2,%zmm1,%zmm12{%k2}
  1a6e23:	62 d2 7e 48 29 d6    	vpmovb2m %zmm14,%k2
  1a6e29:	62 52 7d 48 1c f6    	vpabsb %zmm14,%zmm14
  1a6e2f:	62 22 15 48 50 d3    	vpdpbusd %zmm19,%zmm13,%zmm26
  1a6e35:	62 81 7d 48 70 e8 88 	vpshufd $0x88,%zmm24,%zmm21
  1a6e3c:	62 b1 5d 40 f8 cd    	vpsubb %zmm21,%zmm20,%zmm1
  1a6e42:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0x310(%rsp)
  1a6e49:	10 03 00 00 
  1a6e4d:	62 d2 75 40 50 c7    	vpdpbusd %zmm15,%zmm17,%zmm0
  1a6e53:	62 72 55 42 66 e9    	vpblendmb %zmm1,%zmm21,%zmm13{%k2}
  1a6e59:	62 e1 fe 48 7f ac 24 	vmovdqu64 %zmm21,0x410(%rsp)
  1a6e60:	10 04 00 00 
  1a6e64:	62 42 75 40 50 d4    	vpdpbusd %zmm12,%zmm17,%zmm26
  1a6e6a:	62 d2 0d 48 50 c5    	vpdpbusd %zmm13,%zmm14,%zmm0
  1a6e70:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x10d0(%rsp)
  1a6e77:	d0 10 00 00 
  1a6e7b:	62 f1 fe 48 6f 94 24 	vmovdqu64 0x1d0(%rsp),%zmm2
  1a6e82:	d0 01 00 00 
  1a6e86:	62 e1 7d 48 70 d2 88 	vpshufd $0x88,%zmm2,%zmm18
  1a6e8d:	62 b1 5d 40 f8 da    	vpsubb %zmm18,%zmm20,%zmm3
  1a6e93:	62 72 6d 42 66 e3    	vpblendmb %zmm3,%zmm18,%zmm12{%k2}
  1a6e99:	62 f1 fe 48 7f 9c 24 	vmovdqu64 %zmm3,0x890(%rsp)
  1a6ea0:	90 08 00 00 
  1a6ea4:	62 42 0d 48 50 d4    	vpdpbusd %zmm12,%zmm14,%zmm26
  1a6eaa:	62 61 fe 48 7f 94 24 	vmovdqu64 %zmm26,0x1090(%rsp)
  1a6eb1:	90 10 00 00 
  1a6eb5:	c5 7d 6f fd          	vmovdqa %ymm5,%ymm15
  1a6eb9:	c4 41 79 70 e7 f5    	vpshufd $0xf5,%xmm15,%xmm12
  1a6ebf:	62 53 9d 48 43 e4 00 	vshufi64x2 $0x0,%zmm12,%zmm12,%zmm12
  1a6ec6:	62 d2 7e 48 29 d4    	vpmovb2m %zmm12,%k2
  1a6ecc:	62 d1 7d 48 70 c2 dd 	vpshufd $0xdd,%zmm10,%zmm0
  1a6ed3:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x490(%rsp)
  1a6eda:	90 04 00 00 
  1a6ede:	62 e1 5d 40 f8 d8    	vpsubb %zmm0,%zmm20,%zmm19
  1a6ee4:	62 32 7d 4a 66 d3    	vpblendmb %zmm19,%zmm0,%zmm10{%k2}
  1a6eea:	62 52 7d 48 1c e4    	vpabsb %zmm12,%zmm12
  1a6ef0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  1a6ef4:	62 d2 1d 48 50 c2    	vpdpbusd %zmm10,%zmm12,%zmm0
  1a6efa:	62 71 fd 48 6f f0    	vmovdqa64 %zmm0,%zmm14
  1a6f00:	62 d1 7d 48 70 eb dd 	vpshufd $0xdd,%zmm11,%zmm5
  1a6f07:	62 f1 fe 48 7f ac 24 	vmovdqu64 %zmm5,0x4d0(%rsp)
  1a6f0e:	d0 04 00 00 
  1a6f12:	62 f1 5d 40 f8 cd    	vpsubb %zmm5,%zmm20,%zmm1
  1a6f18:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0x3d0(%rsp)
  1a6f1f:	d0 03 00 00 
  1a6f23:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  1a6f27:	62 72 55 4a 66 d1    	vpblendmb %zmm1,%zmm5,%zmm10{%k2}
  1a6f2d:	62 d2 1d 48 50 c2    	vpdpbusd %zmm10,%zmm12,%zmm0
  1a6f33:	62 31 7d 08 70 d7 f5 	vpshufd $0xf5,%xmm23,%xmm10
  1a6f3a:	62 53 ad 48 43 d2 00 	vshufi64x2 $0x0,%zmm10,%zmm10,%zmm10
  1a6f41:	62 d2 7e 48 29 d2    	vpmovb2m %zmm10,%k2
  1a6f47:	62 c1 7d 48 70 c8 dd 	vpshufd $0xdd,%zmm8,%zmm17
  1a6f4e:	62 b1 5d 40 f8 c9    	vpsubb %zmm17,%zmm20,%zmm1
  1a6f54:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0x190(%rsp)
  1a6f5b:	90 01 00 00 
  1a6f5f:	62 72 75 42 66 c1    	vpblendmb %zmm1,%zmm17,%zmm8{%k2}
  1a6f65:	62 e1 fe 48 7f 8c 24 	vmovdqu64 %zmm17,0x650(%rsp)
  1a6f6c:	50 06 00 00 
  1a6f70:	62 52 7d 48 1c d2    	vpabsb %zmm10,%zmm10
  1a6f76:	62 52 2d 48 50 f0    	vpdpbusd %zmm8,%zmm10,%zmm14
  1a6f7c:	62 d1 7d 48 70 c9 dd 	vpshufd $0xdd,%zmm9,%zmm1
  1a6f83:	62 71 5d 40 f8 e1    	vpsubb %zmm1,%zmm20,%zmm12
  1a6f89:	62 52 75 4a 66 c4    	vpblendmb %zmm12,%zmm1,%zmm8{%k2}
  1a6f8f:	62 71 fe 48 7f a4 24 	vmovdqu64 %zmm12,0x5d0(%rsp)
  1a6f96:	d0 05 00 00 
  1a6f9a:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0x610(%rsp)
  1a6fa1:	10 06 00 00 
  1a6fa5:	62 71 fd 48 6f e8    	vmovdqa64 %zmm0,%zmm13
  1a6fab:	62 52 2d 48 50 e8    	vpdpbusd %zmm8,%zmm10,%zmm13
  1a6fb1:	62 11 7d 08 70 c5 f5 	vpshufd $0xf5,%xmm29,%xmm8
  1a6fb8:	62 53 bd 48 43 c0 00 	vshufi64x2 $0x0,%zmm8,%zmm8,%zmm8
  1a6fbf:	62 d2 7e 48 29 d0    	vpmovb2m %zmm8,%k2
  1a6fc5:	62 52 7d 48 1c c0    	vpabsb %zmm8,%zmm8
  1a6fcb:	62 e1 7d 48 70 c6 dd 	vpshufd $0xdd,%zmm6,%zmm16
  1a6fd2:	62 b1 5d 40 f8 c0    	vpsubb %zmm16,%zmm20,%zmm0
  1a6fd8:	62 f2 7d 42 66 f0    	vpblendmb %zmm0,%zmm16,%zmm6{%k2}
  1a6fde:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x350(%rsp)
  1a6fe5:	50 03 00 00 
  1a6fe9:	62 e1 fe 48 7f 84 24 	vmovdqu64 %zmm16,0x8d0(%rsp)
  1a6ff0:	d0 08 00 00 
  1a6ff4:	62 72 3d 48 50 f6    	vpdpbusd %zmm6,%zmm8,%zmm14
  1a6ffa:	62 71 7d 48 70 d7 dd 	vpshufd $0xdd,%zmm7,%zmm10
  1a7001:	62 51 5d 40 f8 ca    	vpsubb %zmm10,%zmm20,%zmm9
  1a7007:	62 d2 2d 4a 66 f1    	vpblendmb %zmm9,%zmm10,%zmm6{%k2}
  1a700d:	62 71 fe 48 7f 8c 24 	vmovdqu64 %zmm9,0x910(%rsp)
  1a7014:	10 09 00 00 
  1a7018:	62 71 fe 48 7f 94 24 	vmovdqu64 %zmm10,0x290(%rsp)
  1a701f:	90 02 00 00 
  1a7023:	62 72 3d 48 50 ee    	vpdpbusd %zmm6,%zmm8,%zmm13
  1a7029:	c5 7d 6f dc          	vmovdqa %ymm4,%ymm11
  1a702d:	c4 c1 79 70 f3 f5    	vpshufd $0xf5,%xmm11,%xmm6
  1a7033:	62 f3 cd 48 43 f6 00 	vshufi64x2 $0x0,%zmm6,%zmm6,%zmm6
  1a703a:	62 f2 7e 48 29 d6    	vpmovb2m %zmm6,%k2
  1a7040:	62 91 7d 48 70 e8 dd 	vpshufd $0xdd,%zmm24,%zmm5
  1a7047:	62 f1 fe 48 7f ac 24 	vmovdqu64 %zmm5,0x510(%rsp)
  1a704e:	10 05 00 00 
  1a7052:	62 f1 5d 40 f8 e5    	vpsubb %zmm5,%zmm20,%zmm4
  1a7058:	62 f1 fe 48 7f a4 24 	vmovdqu64 %zmm4,0x550(%rsp)
  1a705f:	50 05 00 00 
  1a7063:	62 f2 55 4a 66 e4    	vpblendmb %zmm4,%zmm5,%zmm4{%k2}
  1a7069:	62 f2 7d 48 1c f6    	vpabsb %zmm6,%zmm6
  1a706f:	62 72 4d 48 50 f4    	vpdpbusd %zmm4,%zmm6,%zmm14
  1a7075:	62 71 fe 48 7f b4 24 	vmovdqu64 %zmm14,0xed0(%rsp)
  1a707c:	d0 0e 00 00 
  1a7080:	62 f1 7d 48 70 e2 dd 	vpshufd $0xdd,%zmm2,%zmm4
  1a7087:	62 f1 fe 48 7f a4 24 	vmovdqu64 %zmm4,0x390(%rsp)
  1a708e:	90 03 00 00 
  1a7092:	62 f1 5d 40 f8 d4    	vpsubb %zmm4,%zmm20,%zmm2
  1a7098:	62 f1 fe 48 7f 94 24 	vmovdqu64 %zmm2,0x1d0(%rsp)
  1a709f:	d0 01 00 00 
  1a70a3:	62 f2 5d 4a 66 e2    	vpblendmb %zmm2,%zmm4,%zmm4{%k2}
  1a70a9:	62 72 4d 48 50 ec    	vpdpbusd %zmm4,%zmm6,%zmm13
  1a70af:	62 71 fe 48 7f ac 24 	vmovdqu64 %zmm13,0xe50(%rsp)
  1a70b6:	50 0e 00 00 
  1a70ba:	c4 c1 7d 70 e7 a0    	vpshufd $0xa0,%ymm15,%ymm4
  1a70c0:	c4 41 7d 6f c7       	vmovdqa %ymm15,%ymm8
  1a70c5:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a70cc:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a70d2:	62 61 fe 48 7f a4 24 	vmovdqu64 %zmm28,0x950(%rsp)
  1a70d9:	50 09 00 00 
  1a70dd:	62 92 1d 42 66 e9    	vpblendmb %zmm25,%zmm28,%zmm5{%k2}
  1a70e3:	62 81 fd 48 6f e1    	vmovdqa64 %zmm25,%zmm20
  1a70e9:	62 61 fe 48 7f 8c 24 	vmovdqu64 %zmm25,0x990(%rsp)
  1a70f0:	90 09 00 00 
  1a70f4:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a70fa:	c4 41 09 ef f6       	vpxor  %xmm14,%xmm14,%xmm14
  1a70ff:	62 71 fe 48 6f ac 24 	vmovdqu64 0x110(%rsp),%zmm13
  1a7106:	10 01 00 00 
  1a710a:	62 f1 fe 48 6f bc 24 	vmovdqu64 0x2d0(%rsp),%zmm7
  1a7111:	d0 02 00 00 
  1a7115:	62 d2 45 4a 66 f5    	vpblendmb %zmm13,%zmm7,%zmm6{%k2}
  1a711b:	62 72 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm14
  1a7121:	c4 41 01 ef ff       	vpxor  %xmm15,%xmm15,%xmm15
  1a7126:	62 72 5d 48 50 fe    	vpdpbusd %zmm6,%zmm4,%zmm15
  1a712c:	62 b1 7d 28 70 e7 a0 	vpshufd $0xa0,%ymm23,%ymm4
  1a7133:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a713a:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a7140:	62 01 fd 48 6f ce    	vmovdqa64 %zmm30,%zmm25
  1a7146:	62 61 fe 48 7f 9c 24 	vmovdqu64 %zmm27,0x210(%rsp)
  1a714d:	10 02 00 00 
  1a7151:	62 92 25 42 66 ee    	vpblendmb %zmm30,%zmm27,%zmm5{%k2}
  1a7157:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a715d:	62 72 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm14
  1a7163:	62 61 fe 48 6f b4 24 	vmovdqu64 0x250(%rsp),%zmm30
  1a716a:	50 02 00 00 
  1a716e:	62 61 fe 48 7f bc 24 	vmovdqu64 %zmm31,0x450(%rsp)
  1a7175:	50 04 00 00 
  1a7179:	62 92 05 42 66 ee    	vpblendmb %zmm30,%zmm31,%zmm5{%k2}
  1a717f:	62 72 5d 48 50 fd    	vpdpbusd %zmm5,%zmm4,%zmm15
  1a7185:	62 01 fd 28 6f d5    	vmovdqa64 %ymm29,%ymm26
  1a718b:	62 91 7d 28 70 e5 a0 	vpshufd $0xa0,%ymm29,%ymm4
  1a7192:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a7199:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a719f:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a71a5:	62 61 fe 48 6f ac 24 	vmovdqu64 0xa50(%rsp),%zmm29
  1a71ac:	50 0a 00 00 
  1a71b0:	62 b2 15 42 66 ee    	vpblendmb %zmm22,%zmm29,%zmm5{%k2}
  1a71b6:	62 72 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm14
  1a71bc:	62 f1 fe 48 6f ac 24 	vmovdqu64 0x150(%rsp),%zmm5
  1a71c3:	50 01 00 00 
  1a71c7:	62 f2 55 4a 66 ac 24 	vpblendmb 0x590(%rsp),%zmm5,%zmm5{%k2}
  1a71ce:	90 05 00 00 
  1a71d2:	62 72 5d 48 50 fd    	vpdpbusd %zmm5,%zmm4,%zmm15
  1a71d8:	c4 c1 7d 70 e3 a0    	vpshufd $0xa0,%ymm11,%ymm4
  1a71de:	62 41 fd 28 6f c3    	vmovdqa64 %ymm11,%ymm24
  1a71e4:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a71eb:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a71f1:	62 f2 55 42 66 ac 24 	vpblendmb 0x310(%rsp),%zmm21,%zmm5{%k2}
  1a71f8:	10 03 00 00 
  1a71fc:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a7202:	62 72 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm14
  1a7208:	62 71 fe 48 7f b4 24 	vmovdqu64 %zmm14,0x1050(%rsp)
  1a720f:	50 10 00 00 
  1a7213:	62 f2 6d 42 66 eb    	vpblendmb %zmm3,%zmm18,%zmm5{%k2}
  1a7219:	62 72 5d 48 50 fd    	vpdpbusd %zmm5,%zmm4,%zmm15
  1a721f:	62 71 fe 48 7f bc 24 	vmovdqu64 %zmm15,0x1010(%rsp)
  1a7226:	10 10 00 00 
  1a722a:	c4 c1 7d 70 d8 f5    	vpshufd $0xf5,%ymm8,%ymm3
  1a7230:	62 f3 e5 48 43 db 55 	vshufi64x2 $0x55,%zmm3,%zmm3,%zmm3
  1a7237:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a723d:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a7243:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
  1a7247:	62 71 fe 48 6f 9c 24 	vmovdqu64 0x490(%rsp),%zmm11
  1a724e:	90 04 00 00 
  1a7252:	62 b2 25 4a 66 e3    	vpblendmb %zmm19,%zmm11,%zmm4{%k2}
  1a7258:	62 31 fd 48 6f fb    	vmovdqa64 %zmm19,%zmm15
  1a725e:	62 f2 65 48 50 ec    	vpdpbusd %zmm4,%zmm3,%zmm5
  1a7264:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  1a7268:	62 71 fe 48 6f b4 24 	vmovdqu64 0x3d0(%rsp),%zmm14
  1a726f:	d0 03 00 00 
  1a7273:	62 e1 fe 48 6f ac 24 	vmovdqu64 0x4d0(%rsp),%zmm21
  1a727a:	d0 04 00 00 
  1a727e:	62 d2 55 42 66 e6    	vpblendmb %zmm14,%zmm21,%zmm4{%k2}
  1a7284:	62 f2 65 48 50 f4    	vpdpbusd %zmm4,%zmm3,%zmm6
  1a728a:	62 b1 7d 28 70 d7 f5 	vpshufd $0xf5,%ymm23,%ymm2
  1a7291:	62 f3 ed 48 43 d2 55 	vshufi64x2 $0x55,%zmm2,%zmm2,%zmm2
  1a7298:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a729e:	62 f2 7d 48 1c d2    	vpabsb %zmm2,%zmm2
  1a72a4:	62 f2 75 42 66 9c 24 	vpblendmb 0x190(%rsp),%zmm17,%zmm3{%k2}
  1a72ab:	90 01 00 00 
  1a72af:	62 f2 6d 48 50 eb    	vpdpbusd %zmm3,%zmm2,%zmm5
  1a72b5:	62 d2 75 4a 66 dc    	vpblendmb %zmm12,%zmm1,%zmm3{%k2}
  1a72bb:	62 f2 6d 48 50 f3    	vpdpbusd %zmm3,%zmm2,%zmm6
  1a72c1:	62 91 7d 28 70 ca f5 	vpshufd $0xf5,%ymm26,%ymm1
  1a72c8:	62 f3 f5 48 43 c9 55 	vshufi64x2 $0x55,%zmm1,%zmm1,%zmm1
  1a72cf:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a72d5:	62 f2 7d 42 66 d0    	vpblendmb %zmm0,%zmm16,%zmm2{%k2}
  1a72db:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a72e1:	62 f2 75 48 50 ea    	vpdpbusd %zmm2,%zmm1,%zmm5
  1a72e7:	62 d2 2d 4a 66 d1    	vpblendmb %zmm9,%zmm10,%zmm2{%k2}
  1a72ed:	62 f2 75 48 50 f2    	vpdpbusd %zmm2,%zmm1,%zmm6
  1a72f3:	62 f1 fd 48 6f e6    	vmovdqa64 %zmm6,%zmm4
  1a72f9:	c4 41 7e 6f 64 19 68 	vmovdqu 0x68(%r9,%rbx,1),%ymm12
  1a7300:	c4 c1 79 70 d4 a0    	vpshufd $0xa0,%xmm12,%xmm2
  1a7306:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a730d:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a7313:	62 f2 7d 48 1c d2    	vpabsb %zmm2,%zmm2
  1a7319:	62 b2 1d 42 66 dc    	vpblendmb %zmm20,%zmm28,%zmm3{%k2}
  1a731f:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  1a7323:	62 f2 6d 48 50 c3    	vpdpbusd %zmm3,%zmm2,%zmm0
  1a7329:	62 71 fd 48 6f d0    	vmovdqa64 %zmm0,%zmm10
  1a732f:	62 d2 45 4a 66 dd    	vpblendmb %zmm13,%zmm7,%zmm3{%k2}
  1a7335:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  1a7339:	62 f2 6d 48 50 f3    	vpdpbusd %zmm3,%zmm2,%zmm6
  1a733f:	62 91 7d 28 70 c0 f5 	vpshufd $0xf5,%ymm24,%ymm0
  1a7346:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a734d:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a7353:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7359:	62 f1 fe 48 6f bc 24 	vmovdqu64 0x510(%rsp),%zmm7
  1a7360:	10 05 00 00 
  1a7364:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x550(%rsp),%zmm1
  1a736b:	50 05 00 00 
  1a736f:	62 f2 45 4a 66 d1    	vpblendmb %zmm1,%zmm7,%zmm2{%k2}
  1a7375:	62 f2 7d 48 50 ea    	vpdpbusd %zmm2,%zmm0,%zmm5
  1a737b:	62 f1 fe 48 7f ac 24 	vmovdqu64 %zmm5,0xe90(%rsp)
  1a7382:	90 0e 00 00 
  1a7386:	62 71 fe 48 6f 84 24 	vmovdqu64 0x390(%rsp),%zmm8
  1a738d:	90 03 00 00 
  1a7391:	62 e1 fe 48 6f a4 24 	vmovdqu64 0x1d0(%rsp),%zmm20
  1a7398:	d0 01 00 00 
  1a739c:	62 b2 3d 4a 66 d4    	vpblendmb %zmm20,%zmm8,%zmm2{%k2}
  1a73a2:	62 f2 7d 48 50 e2    	vpdpbusd %zmm2,%zmm0,%zmm4
  1a73a8:	62 f1 fe 48 7f a4 24 	vmovdqu64 %zmm4,0xe10(%rsp)
  1a73af:	10 0e 00 00 
  1a73b3:	c4 c1 7e 6f 54 19 48 	vmovdqu 0x48(%r9,%rbx,1),%ymm2
  1a73ba:	c5 f9 70 c2 a0       	vpshufd $0xa0,%xmm2,%xmm0
  1a73bf:	62 f3 fd 48 43 c0 00 	vshufi64x2 $0x0,%zmm0,%zmm0,%zmm0
  1a73c6:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a73cc:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a73d2:	62 92 25 42 66 d9    	vpblendmb %zmm25,%zmm27,%zmm3{%k2}
  1a73d8:	62 81 fd 48 6f f9    	vmovdqa64 %zmm25,%zmm23
  1a73de:	62 61 fe 48 7f 8c 24 	vmovdqu64 %zmm25,0xb50(%rsp)
  1a73e5:	50 0b 00 00 
  1a73e9:	62 72 7d 48 50 d3    	vpdpbusd %zmm3,%zmm0,%zmm10
  1a73ef:	62 92 05 42 66 de    	vpblendmb %zmm30,%zmm31,%zmm3{%k2}
  1a73f5:	62 01 fd 48 6f d6    	vmovdqa64 %zmm30,%zmm26
  1a73fb:	62 f2 7d 48 50 f3    	vpdpbusd %zmm3,%zmm0,%zmm6
  1a7401:	c4 c1 7e 6f 5c 19 28 	vmovdqu 0x28(%r9,%rbx,1),%ymm3
  1a7408:	c5 f9 70 c3 a0       	vpshufd $0xa0,%xmm3,%xmm0
  1a740d:	62 f3 fd 48 43 c0 00 	vshufi64x2 $0x0,%zmm0,%zmm0,%zmm0
  1a7414:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a741a:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7420:	62 b2 15 42 66 e6    	vpblendmb %zmm22,%zmm29,%zmm4{%k2}
  1a7426:	62 81 fd 48 6f c5    	vmovdqa64 %zmm29,%zmm16
  1a742c:	62 a1 fd 48 6f ce    	vmovdqa64 %zmm22,%zmm17
  1a7432:	62 71 fe 48 6f 8c 24 	vmovdqu64 0x590(%rsp),%zmm9
  1a7439:	90 05 00 00 
  1a743d:	62 61 fe 48 6f a4 24 	vmovdqu64 0x150(%rsp),%zmm28
  1a7444:	50 01 00 00 
  1a7448:	62 d2 1d 42 66 e9    	vpblendmb %zmm9,%zmm28,%zmm5{%k2}
  1a744e:	62 72 7d 48 50 d4    	vpdpbusd %zmm4,%zmm0,%zmm10
  1a7454:	62 f2 7d 48 50 f5    	vpdpbusd %zmm5,%zmm0,%zmm6
  1a745a:	62 41 fe 28 6f bc 19 	vmovdqu64 0x8(%r9,%rbx,1),%ymm31
  1a7461:	08 00 00 00 
  1a7465:	62 91 7d 08 70 e7 a0 	vpshufd $0xa0,%xmm31,%xmm4
  1a746c:	62 f3 dd 48 43 e4 00 	vshufi64x2 $0x0,%zmm4,%zmm4,%zmm4
  1a7473:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a7479:	62 61 fe 48 6f 9c 24 	vmovdqu64 0x310(%rsp),%zmm27
  1a7480:	10 03 00 00 
  1a7484:	62 61 fe 48 6f b4 24 	vmovdqu64 0x410(%rsp),%zmm30
  1a748b:	10 04 00 00 
  1a748f:	62 92 0d 42 66 eb    	vpblendmb %zmm27,%zmm30,%zmm5{%k2}
  1a7495:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a749b:	62 72 5d 48 50 d5    	vpdpbusd %zmm5,%zmm4,%zmm10
  1a74a1:	62 71 fe 48 7f 94 24 	vmovdqu64 %zmm10,0xfd0(%rsp)
  1a74a8:	d0 0f 00 00 
  1a74ac:	62 61 fe 48 6f ac 24 	vmovdqu64 0x890(%rsp),%zmm29
  1a74b3:	90 08 00 00 
  1a74b7:	62 21 fd 48 6f c2    	vmovdqa64 %zmm18,%zmm24
  1a74bd:	62 e1 fe 48 7f 94 24 	vmovdqu64 %zmm18,0xa10(%rsp)
  1a74c4:	10 0a 00 00 
  1a74c8:	62 92 6d 42 66 ed    	vpblendmb %zmm29,%zmm18,%zmm5{%k2}
  1a74ce:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  1a74d4:	62 f1 fe 48 7f b4 24 	vmovdqu64 %zmm6,0xf90(%rsp)
  1a74db:	90 0f 00 00 
  1a74df:	c4 c1 79 70 e4 f5    	vpshufd $0xf5,%xmm12,%xmm4
  1a74e5:	62 f3 dd 48 43 e4 00 	vshufi64x2 $0x0,%zmm4,%zmm4,%zmm4
  1a74ec:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a74f2:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a74f8:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  1a74fc:	62 b2 25 4a 66 eb    	vpblendmb %zmm19,%zmm11,%zmm5{%k2}
  1a7502:	62 e1 fe 48 7f 9c 24 	vmovdqu64 %zmm19,0x9d0(%rsp)
  1a7509:	d0 09 00 00 
  1a750d:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  1a7513:	62 01 35 00 ef c9    	vpxord %xmm25,%xmm25,%xmm25
  1a7519:	62 d2 55 42 66 ee    	vpblendmb %zmm14,%zmm21,%zmm5{%k2}
  1a751f:	62 62 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm25
  1a7525:	c5 f9 70 e2 f5       	vpshufd $0xf5,%xmm2,%xmm4
  1a752a:	62 f3 dd 48 43 e4 00 	vshufi64x2 $0x0,%zmm4,%zmm4,%zmm4
  1a7531:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a7537:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a753d:	62 71 fe 48 6f 9c 24 	vmovdqu64 0x190(%rsp),%zmm11
  1a7544:	90 01 00 00 
  1a7548:	62 f1 fe 48 6f 84 24 	vmovdqu64 0x650(%rsp),%zmm0
  1a754f:	50 06 00 00 
  1a7553:	62 d2 7d 4a 66 eb    	vpblendmb %zmm11,%zmm0,%zmm5{%k2}
  1a7559:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  1a755f:	62 71 fe 48 6f 94 24 	vmovdqu64 0x5d0(%rsp),%zmm10
  1a7566:	d0 05 00 00 
  1a756a:	62 e1 fe 48 6f 9c 24 	vmovdqu64 0x610(%rsp),%zmm19
  1a7571:	10 06 00 00 
  1a7575:	62 d2 65 42 66 ea    	vpblendmb %zmm10,%zmm19,%zmm5{%k2}
  1a757b:	62 62 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm25
  1a7581:	c5 f9 70 e3 f5       	vpshufd $0xf5,%xmm3,%xmm4
  1a7586:	62 f3 dd 48 43 e4 00 	vshufi64x2 $0x0,%zmm4,%zmm4,%zmm4
  1a758d:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a7593:	62 e1 fe 48 6f ac 24 	vmovdqu64 0x8d0(%rsp),%zmm21
  1a759a:	d0 08 00 00 
  1a759e:	62 e1 fe 48 6f b4 24 	vmovdqu64 0x350(%rsp),%zmm22
  1a75a5:	50 03 00 00 
  1a75a9:	62 b2 55 42 66 ee    	vpblendmb %zmm22,%zmm21,%zmm5{%k2}
  1a75af:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a75b5:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  1a75bb:	62 e1 fe 48 6f 94 24 	vmovdqu64 0x290(%rsp),%zmm18
  1a75c2:	90 02 00 00 
  1a75c6:	62 71 fe 48 6f ac 24 	vmovdqu64 0x910(%rsp),%zmm13
  1a75cd:	10 09 00 00 
  1a75d1:	62 d2 6d 42 66 ed    	vpblendmb %zmm13,%zmm18,%zmm5{%k2}
  1a75d7:	62 62 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm25
  1a75dd:	62 91 7d 08 70 e7 f5 	vpshufd $0xf5,%xmm31,%xmm4
  1a75e4:	62 f3 dd 48 43 e4 00 	vshufi64x2 $0x0,%zmm4,%zmm4,%zmm4
  1a75eb:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a75f1:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a75f7:	62 f2 45 4a 66 e9    	vpblendmb %zmm1,%zmm7,%zmm5{%k2}
  1a75fd:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  1a7603:	62 f1 fe 48 7f b4 24 	vmovdqu64 %zmm6,0xdd0(%rsp)
  1a760a:	d0 0d 00 00 
  1a760e:	62 b2 3d 4a 66 ec    	vpblendmb %zmm20,%zmm8,%zmm5{%k2}
  1a7614:	62 62 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm25
  1a761a:	62 61 fe 48 7f 8c 24 	vmovdqu64 %zmm25,0xbd0(%rsp)
  1a7621:	d0 0b 00 00 
  1a7625:	c4 c1 7d 70 e4 a0    	vpshufd $0xa0,%ymm12,%ymm4
  1a762b:	62 c1 fd 28 6f e4    	vmovdqa64 %ymm12,%ymm20
  1a7631:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a7638:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a763e:	62 71 fe 48 6f a4 24 	vmovdqu64 0x950(%rsp),%zmm12
  1a7645:	50 09 00 00 
  1a7649:	62 61 fe 48 6f 8c 24 	vmovdqu64 0x990(%rsp),%zmm25
  1a7650:	90 09 00 00 
  1a7654:	62 92 1d 4a 66 e9    	vpblendmb %zmm25,%zmm12,%zmm5{%k2}
  1a765a:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a7660:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
  1a7664:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x2d0(%rsp),%zmm6
  1a766b:	d0 02 00 00 
  1a766f:	62 f2 4d 4a 66 b4 24 	vpblendmb 0x110(%rsp),%zmm6,%zmm6{%k2}
  1a7676:	10 01 00 00 
  1a767a:	62 f2 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm1
  1a7680:	c4 41 39 ef c0       	vpxor  %xmm8,%xmm8,%xmm8
  1a7685:	62 72 5d 48 50 c6    	vpdpbusd %zmm6,%zmm4,%zmm8
  1a768b:	c5 fd 70 e2 a0       	vpshufd $0xa0,%ymm2,%ymm4
  1a7690:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a7697:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a769d:	62 f1 fe 48 6f ac 24 	vmovdqu64 0x210(%rsp),%zmm5
  1a76a4:	10 02 00 00 
  1a76a8:	62 b2 55 4a 66 ef    	vpblendmb %zmm23,%zmm5,%zmm5{%k2}
  1a76ae:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a76b4:	62 f2 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm1
  1a76ba:	62 f1 fe 48 6f ac 24 	vmovdqu64 0x450(%rsp),%zmm5
  1a76c1:	50 04 00 00 
  1a76c5:	62 92 55 4a 66 ea    	vpblendmb %zmm26,%zmm5,%zmm5{%k2}
  1a76cb:	62 72 5d 48 50 c5    	vpdpbusd %zmm5,%zmm4,%zmm8
  1a76d1:	c5 fd 70 e3 a0       	vpshufd $0xa0,%ymm3,%ymm4
  1a76d6:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a76dd:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a76e3:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a76e9:	62 b2 7d 42 66 e9    	vpblendmb %zmm17,%zmm16,%zmm5{%k2}
  1a76ef:	62 21 fd 48 6f d0    	vmovdqa64 %zmm16,%zmm26
  1a76f5:	62 a1 fd 48 6f c1    	vmovdqa64 %zmm17,%zmm16
  1a76fb:	62 f2 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm1
  1a7701:	62 d2 1d 42 66 e9    	vpblendmb %zmm9,%zmm28,%zmm5{%k2}
  1a7707:	62 c1 fd 48 6f c9    	vmovdqa64 %zmm9,%zmm17
  1a770d:	62 72 5d 48 50 c5    	vpdpbusd %zmm5,%zmm4,%zmm8
  1a7713:	62 91 7d 28 70 e7 a0 	vpshufd $0xa0,%ymm31,%ymm4
  1a771a:	62 81 fd 28 6f ff    	vmovdqa64 %ymm31,%ymm23
  1a7720:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a7727:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a772d:	62 92 0d 42 66 eb    	vpblendmb %zmm27,%zmm30,%zmm5{%k2}
  1a7733:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a7739:	62 f2 5d 48 50 cd    	vpdpbusd %zmm5,%zmm4,%zmm1
  1a773f:	62 f1 fe 48 7f 8c 24 	vmovdqu64 %zmm1,0xf50(%rsp)
  1a7746:	50 0f 00 00 
  1a774a:	62 92 3d 42 66 ed    	vpblendmb %zmm29,%zmm24,%zmm5{%k2}
  1a7750:	62 72 5d 48 50 c5    	vpdpbusd %zmm5,%zmm4,%zmm8
  1a7756:	62 71 fe 48 7f 84 24 	vmovdqu64 %zmm8,0xf10(%rsp)
  1a775d:	10 0f 00 00 
  1a7761:	62 b1 7d 28 70 cc f5 	vpshufd $0xf5,%ymm20,%ymm1
  1a7768:	62 f3 f5 48 43 c9 55 	vshufi64x2 $0x55,%zmm1,%zmm1,%zmm1
  1a776f:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a7775:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a777b:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
  1a777f:	62 71 fe 48 6f 84 24 	vmovdqu64 0x490(%rsp),%zmm8
  1a7786:	90 04 00 00 
  1a778a:	62 d2 3d 4a 66 e7    	vpblendmb %zmm15,%zmm8,%zmm4{%k2}
  1a7790:	62 f2 75 48 50 ec    	vpdpbusd %zmm4,%zmm1,%zmm5
  1a7796:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  1a779a:	62 61 fe 48 6f 9c 24 	vmovdqu64 0x4d0(%rsp),%zmm27
  1a77a1:	d0 04 00 00 
  1a77a5:	62 d2 25 42 66 e6    	vpblendmb %zmm14,%zmm27,%zmm4{%k2}
  1a77ab:	62 f2 75 48 50 f4    	vpdpbusd %zmm4,%zmm1,%zmm6
  1a77b1:	c5 fd 70 ca f5       	vpshufd $0xf5,%ymm2,%ymm1
  1a77b6:	62 f3 f5 48 43 c9 55 	vshufi64x2 $0x55,%zmm1,%zmm1,%zmm1
  1a77bd:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a77c3:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a77c9:	62 d2 7d 4a 66 d3    	vpblendmb %zmm11,%zmm0,%zmm2{%k2}
  1a77cf:	62 f2 75 48 50 ea    	vpdpbusd %zmm2,%zmm1,%zmm5
  1a77d5:	62 d2 65 42 66 d2    	vpblendmb %zmm10,%zmm19,%zmm2{%k2}
  1a77db:	62 f2 75 48 50 f2    	vpdpbusd %zmm2,%zmm1,%zmm6
  1a77e1:	c5 fd 70 cb f5       	vpshufd $0xf5,%ymm3,%ymm1
  1a77e6:	62 f3 f5 48 43 c9 55 	vshufi64x2 $0x55,%zmm1,%zmm1,%zmm1
  1a77ed:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a77f3:	62 b2 55 42 66 d6    	vpblendmb %zmm22,%zmm21,%zmm2{%k2}
  1a77f9:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a77ff:	62 f2 75 48 50 ea    	vpdpbusd %zmm2,%zmm1,%zmm5
  1a7805:	62 c1 fd 48 6f e5    	vmovdqa64 %zmm13,%zmm20
  1a780b:	62 d2 6d 42 66 d5    	vpblendmb %zmm13,%zmm18,%zmm2{%k2}
  1a7811:	62 f2 75 48 50 f2    	vpdpbusd %zmm2,%zmm1,%zmm6
  1a7817:	62 71 fd 48 6f ce    	vmovdqa64 %zmm6,%zmm9
  1a781d:	62 41 fe 28 6f ac 18 	vmovdqu64 0x68(%r8,%rbx,1),%ymm29
  1a7824:	68 00 00 00 
  1a7828:	62 91 7d 08 70 cd a0 	vpshufd $0xa0,%xmm29,%xmm1
  1a782f:	62 f3 f5 48 43 c9 00 	vshufi64x2 $0x0,%zmm1,%zmm1,%zmm1
  1a7836:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a783c:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a7842:	62 92 1d 4a 66 d1    	vpblendmb %zmm25,%zmm12,%zmm2{%k2}
  1a7848:	62 41 fd 48 6f c4    	vmovdqa64 %zmm12,%zmm24
  1a784e:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
  1a7852:	62 f2 75 48 50 da    	vpdpbusd %zmm2,%zmm1,%zmm3
  1a7858:	62 71 fd 48 6f d3    	vmovdqa64 %zmm3,%zmm10
  1a785e:	62 61 fe 48 6f a4 24 	vmovdqu64 0x110(%rsp),%zmm28
  1a7865:	10 01 00 00 
  1a7869:	62 61 fe 48 6f bc 24 	vmovdqu64 0x2d0(%rsp),%zmm31
  1a7870:	d0 02 00 00 
  1a7874:	62 92 05 42 66 d4    	vpblendmb %zmm28,%zmm31,%zmm2{%k2}
  1a787a:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
  1a787e:	62 f2 75 48 50 da    	vpdpbusd %zmm2,%zmm1,%zmm3
  1a7884:	62 e1 fd 48 6f db    	vmovdqa64 %zmm3,%zmm19
  1a788a:	62 b1 7d 28 70 c7 f5 	vpshufd $0xf5,%ymm23,%ymm0
  1a7891:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a7898:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a789e:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a78a4:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x510(%rsp),%zmm6
  1a78ab:	10 05 00 00 
  1a78af:	62 f1 fe 48 6f a4 24 	vmovdqu64 0x550(%rsp),%zmm4
  1a78b6:	50 05 00 00 
  1a78ba:	62 f2 4d 4a 66 cc    	vpblendmb %zmm4,%zmm6,%zmm1{%k2}
  1a78c0:	62 f2 7d 48 50 e9    	vpdpbusd %zmm1,%zmm0,%zmm5
  1a78c6:	62 f1 fe 48 7f ac 24 	vmovdqu64 %zmm5,0xcd0(%rsp)
  1a78cd:	d0 0c 00 00 
  1a78d1:	62 f1 fe 48 6f bc 24 	vmovdqu64 0x390(%rsp),%zmm7
  1a78d8:	90 03 00 00 
  1a78dc:	62 f1 fe 48 6f 9c 24 	vmovdqu64 0x1d0(%rsp),%zmm3
  1a78e3:	d0 01 00 00 
  1a78e7:	62 f2 45 4a 66 cb    	vpblendmb %zmm3,%zmm7,%zmm1{%k2}
  1a78ed:	62 72 7d 48 50 c9    	vpdpbusd %zmm1,%zmm0,%zmm9
  1a78f3:	62 71 fe 48 7f 8c 24 	vmovdqu64 %zmm9,0xc50(%rsp)
  1a78fa:	50 0c 00 00 
  1a78fe:	c4 41 7e 6f 5c 18 48 	vmovdqu 0x48(%r8,%rbx,1),%ymm11
  1a7905:	c4 c1 79 70 cb a0    	vpshufd $0xa0,%xmm11,%xmm1
  1a790b:	62 f3 f5 48 43 c9 00 	vshufi64x2 $0x0,%zmm1,%zmm1,%zmm1
  1a7912:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a7918:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a791e:	62 71 fe 48 6f bc 24 	vmovdqu64 0x210(%rsp),%zmm15
  1a7925:	10 02 00 00 
  1a7929:	62 e1 fe 48 6f bc 24 	vmovdqu64 0xb50(%rsp),%zmm23
  1a7930:	50 0b 00 00 
  1a7934:	62 b2 05 4a 66 d7    	vpblendmb %zmm23,%zmm15,%zmm2{%k2}
  1a793a:	62 d1 fd 48 6f c2    	vmovdqa64 %zmm10,%zmm0
  1a7940:	62 f2 75 48 50 c2    	vpdpbusd %zmm2,%zmm1,%zmm0
  1a7946:	62 61 fe 48 6f b4 24 	vmovdqu64 0x250(%rsp),%zmm30
  1a794d:	50 02 00 00 
  1a7951:	62 71 fe 48 6f a4 24 	vmovdqu64 0x450(%rsp),%zmm12
  1a7958:	50 04 00 00 
  1a795c:	62 92 1d 4a 66 d6    	vpblendmb %zmm30,%zmm12,%zmm2{%k2}
  1a7962:	62 e2 75 48 50 da    	vpdpbusd %zmm2,%zmm1,%zmm19
  1a7968:	c4 41 7e 6f 74 18 28 	vmovdqu 0x28(%r8,%rbx,1),%ymm14
  1a796f:	c4 c1 79 70 ce a0    	vpshufd $0xa0,%xmm14,%xmm1
  1a7975:	62 f3 f5 48 43 c9 00 	vshufi64x2 $0x0,%zmm1,%zmm1,%zmm1
  1a797c:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a7982:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a7988:	62 11 fd 48 6f d2    	vmovdqa64 %zmm26,%zmm10
  1a798e:	62 b2 2d 42 66 d0    	vpblendmb %zmm16,%zmm26,%zmm2{%k2}
  1a7994:	62 21 fd 48 6f d0    	vmovdqa64 %zmm16,%zmm26
  1a799a:	62 e1 fe 48 7f 84 24 	vmovdqu64 %zmm16,0xb90(%rsp)
  1a79a1:	90 0b 00 00 
  1a79a5:	62 71 fe 48 6f ac 24 	vmovdqu64 0x150(%rsp),%zmm13
  1a79ac:	50 01 00 00 
  1a79b0:	62 b2 15 4a 66 e9    	vpblendmb %zmm17,%zmm13,%zmm5{%k2}
  1a79b6:	62 f2 75 48 50 c2    	vpdpbusd %zmm2,%zmm1,%zmm0
  1a79bc:	62 e2 75 48 50 dd    	vpdpbusd %zmm5,%zmm1,%zmm19
  1a79c2:	62 b1 fd 48 6f eb    	vmovdqa64 %zmm19,%zmm5
  1a79c8:	62 c1 fe 28 6f 8c 18 	vmovdqu64 0x8(%r8,%rbx,1),%ymm17
  1a79cf:	08 00 00 00 
  1a79d3:	62 b1 7d 08 70 c9 a0 	vpshufd $0xa0,%xmm17,%xmm1
  1a79da:	62 f3 f5 48 43 c9 00 	vshufi64x2 $0x0,%zmm1,%zmm1,%zmm1
  1a79e1:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a79e7:	62 71 fe 48 6f 8c 24 	vmovdqu64 0x410(%rsp),%zmm9
  1a79ee:	10 04 00 00 
  1a79f2:	62 e1 fe 48 6f b4 24 	vmovdqu64 0x310(%rsp),%zmm22
  1a79f9:	10 03 00 00 
  1a79fd:	62 b2 35 4a 66 d6    	vpblendmb %zmm22,%zmm9,%zmm2{%k2}
  1a7a03:	62 f2 7d 48 1c c9    	vpabsb %zmm1,%zmm1
  1a7a09:	62 f2 75 48 50 c2    	vpdpbusd %zmm2,%zmm1,%zmm0
  1a7a0f:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0xd10(%rsp)
  1a7a16:	10 0d 00 00 
  1a7a1a:	62 e1 fe 48 6f 9c 24 	vmovdqu64 0xa10(%rsp),%zmm19
  1a7a21:	10 0a 00 00 
  1a7a25:	62 e1 fe 48 6f 94 24 	vmovdqu64 0x890(%rsp),%zmm18
  1a7a2c:	90 08 00 00 
  1a7a30:	62 b2 65 42 66 d2    	vpblendmb %zmm18,%zmm19,%zmm2{%k2}
  1a7a36:	62 f2 75 48 50 ea    	vpdpbusd %zmm2,%zmm1,%zmm5
  1a7a3c:	62 f1 fe 48 7f ac 24 	vmovdqu64 %zmm5,0xc90(%rsp)
  1a7a43:	90 0c 00 00 
  1a7a47:	62 91 7d 08 70 cd f5 	vpshufd $0xf5,%xmm29,%xmm1
  1a7a4e:	62 f3 f5 48 43 c9 00 	vshufi64x2 $0x0,%zmm1,%zmm1,%zmm1
  1a7a55:	62 f2 7e 48 29 d1    	vpmovb2m %zmm1,%k2
  1a7a5b:	62 f2 7d 48 1c e9    	vpabsb %zmm1,%zmm5
  1a7a61:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  1a7a65:	62 f1 fe 48 6f 94 24 	vmovdqu64 0x9d0(%rsp),%zmm2
  1a7a6c:	d0 09 00 00 
  1a7a70:	62 f2 3d 4a 66 ca    	vpblendmb %zmm2,%zmm8,%zmm1{%k2}
  1a7a76:	62 f2 55 48 50 c1    	vpdpbusd %zmm1,%zmm5,%zmm0
  1a7a7c:	62 a1 7d 00 ef c0    	vpxord %xmm16,%xmm16,%xmm16
  1a7a82:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x3d0(%rsp),%zmm1
  1a7a89:	d0 03 00 00 
  1a7a8d:	62 72 25 42 66 c1    	vpblendmb %zmm1,%zmm27,%zmm8{%k2}
  1a7a93:	62 c2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm16
  1a7a99:	c4 c1 79 70 eb f5    	vpshufd $0xf5,%xmm11,%xmm5
  1a7a9f:	62 f3 d5 48 43 ed 00 	vshufi64x2 $0x0,%zmm5,%zmm5,%zmm5
  1a7aa6:	62 f2 7e 48 29 d5    	vpmovb2m %zmm5,%k2
  1a7aac:	62 f2 7d 48 1c ed    	vpabsb %zmm5,%zmm5
  1a7ab2:	62 61 fe 48 6f 9c 24 	vmovdqu64 0x190(%rsp),%zmm27
  1a7ab9:	90 01 00 00 
  1a7abd:	62 71 fe 48 6f 84 24 	vmovdqu64 0x650(%rsp),%zmm8
  1a7ac4:	50 06 00 00 
  1a7ac8:	62 12 3d 4a 66 c3    	vpblendmb %zmm27,%zmm8,%zmm8{%k2}
  1a7ace:	62 d2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm0
  1a7ad4:	62 71 fe 48 6f 84 24 	vmovdqu64 0x610(%rsp),%zmm8
  1a7adb:	10 06 00 00 
  1a7adf:	62 72 3d 4a 66 84 24 	vpblendmb 0x5d0(%rsp),%zmm8,%zmm8{%k2}
  1a7ae6:	d0 05 00 00 
  1a7aea:	62 c2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm16
  1a7af0:	c4 c1 79 70 ee f5    	vpshufd $0xf5,%xmm14,%xmm5
  1a7af6:	62 f3 d5 48 43 ed 00 	vshufi64x2 $0x0,%zmm5,%zmm5,%zmm5
  1a7afd:	62 f2 7e 48 29 d5    	vpmovb2m %zmm5,%k2
  1a7b03:	62 72 55 42 66 84 24 	vpblendmb 0x350(%rsp),%zmm21,%zmm8{%k2}
  1a7b0a:	50 03 00 00 
  1a7b0e:	62 f2 7d 48 1c ed    	vpabsb %zmm5,%zmm5
  1a7b14:	62 d2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm0
  1a7b1a:	62 71 fe 48 6f 84 24 	vmovdqu64 0x290(%rsp),%zmm8
  1a7b21:	90 02 00 00 
  1a7b25:	62 32 3d 4a 66 c4    	vpblendmb %zmm20,%zmm8,%zmm8{%k2}
  1a7b2b:	62 c2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm16
  1a7b31:	62 b1 7d 08 70 e9 f5 	vpshufd $0xf5,%xmm17,%xmm5
  1a7b38:	62 f3 d5 48 43 ed 00 	vshufi64x2 $0x0,%zmm5,%zmm5,%zmm5
  1a7b3f:	62 f2 7e 48 29 d5    	vpmovb2m %zmm5,%k2
  1a7b45:	62 f2 7d 48 1c ed    	vpabsb %zmm5,%zmm5
  1a7b4b:	62 72 4d 4a 66 c4    	vpblendmb %zmm4,%zmm6,%zmm8{%k2}
  1a7b51:	62 d2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm0
  1a7b57:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0xc10(%rsp)
  1a7b5e:	10 0c 00 00 
  1a7b62:	62 72 45 4a 66 c3    	vpblendmb %zmm3,%zmm7,%zmm8{%k2}
  1a7b68:	62 c2 55 48 50 c0    	vpdpbusd %zmm8,%zmm5,%zmm16
  1a7b6e:	62 91 7d 28 70 ed a0 	vpshufd $0xa0,%ymm29,%ymm5
  1a7b75:	62 f3 d5 48 43 ed 55 	vshufi64x2 $0x55,%zmm5,%zmm5,%zmm5
  1a7b7c:	62 f2 7e 48 29 d5    	vpmovb2m %zmm5,%k2
  1a7b82:	62 12 3d 42 66 c1    	vpblendmb %zmm25,%zmm24,%zmm8{%k2}
  1a7b88:	62 f2 7d 48 1c ed    	vpabsb %zmm5,%zmm5
  1a7b8e:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
  1a7b92:	62 92 05 42 66 c4    	vpblendmb %zmm28,%zmm31,%zmm0{%k2}
  1a7b98:	62 d2 55 48 50 d8    	vpdpbusd %zmm8,%zmm5,%zmm3
  1a7b9e:	c5 d9 ef e4          	vpxor  %xmm4,%xmm4,%xmm4
  1a7ba2:	62 f2 55 48 50 e0    	vpdpbusd %zmm0,%zmm5,%zmm4
  1a7ba8:	c4 c1 7d 70 c3 a0    	vpshufd $0xa0,%ymm11,%ymm0
  1a7bae:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a7bb5:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a7bbb:	62 a1 fd 48 6f e7    	vmovdqa64 %zmm23,%zmm20
  1a7bc1:	62 b2 05 4a 66 ef    	vpblendmb %zmm23,%zmm15,%zmm5{%k2}
  1a7bc7:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7bcd:	62 f2 7d 48 50 dd    	vpdpbusd %zmm5,%zmm0,%zmm3
  1a7bd3:	62 92 1d 4a 66 ee    	vpblendmb %zmm30,%zmm12,%zmm5{%k2}
  1a7bd9:	62 f2 7d 48 50 e5    	vpdpbusd %zmm5,%zmm0,%zmm4
  1a7bdf:	c4 c1 7d 70 c6 a0    	vpshufd $0xa0,%ymm14,%ymm0
  1a7be5:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a7bec:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a7bf2:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7bf8:	62 92 2d 4a 66 ea    	vpblendmb %zmm26,%zmm10,%zmm5{%k2}
  1a7bfe:	62 f2 7d 48 50 dd    	vpdpbusd %zmm5,%zmm0,%zmm3
  1a7c04:	62 61 fe 48 6f 84 24 	vmovdqu64 0x590(%rsp),%zmm24
  1a7c0b:	90 05 00 00 
  1a7c0f:	62 92 15 4a 66 e8    	vpblendmb %zmm24,%zmm13,%zmm5{%k2}
  1a7c15:	62 f2 7d 48 50 e5    	vpdpbusd %zmm5,%zmm0,%zmm4
  1a7c1b:	62 b1 7d 28 70 c1 a0 	vpshufd $0xa0,%ymm17,%ymm0
  1a7c22:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a7c29:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a7c2f:	62 21 fd 48 6f ce    	vmovdqa64 %zmm22,%zmm25
  1a7c35:	62 b2 35 4a 66 ee    	vpblendmb %zmm22,%zmm9,%zmm5{%k2}
  1a7c3b:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7c41:	62 f2 7d 48 50 dd    	vpdpbusd %zmm5,%zmm0,%zmm3
  1a7c47:	62 f1 fe 48 7f 9c 24 	vmovdqu64 %zmm3,0xd90(%rsp)
  1a7c4e:	90 0d 00 00 
  1a7c52:	62 31 fd 48 6f e2    	vmovdqa64 %zmm18,%zmm12
  1a7c58:	62 b2 65 42 66 ea    	vpblendmb %zmm18,%zmm19,%zmm5{%k2}
  1a7c5e:	62 f2 7d 48 50 e5    	vpdpbusd %zmm5,%zmm0,%zmm4
  1a7c64:	62 f1 fe 48 7f a4 24 	vmovdqu64 %zmm4,0xd50(%rsp)
  1a7c6b:	50 0d 00 00 
  1a7c6f:	62 91 7d 28 70 c5 f5 	vpshufd $0xf5,%ymm29,%ymm0
  1a7c76:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a7c7d:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a7c83:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7c89:	c4 41 39 ef c0       	vpxor  %xmm8,%xmm8,%xmm8
  1a7c8e:	62 71 fe 48 6f ac 24 	vmovdqu64 0x490(%rsp),%zmm13
  1a7c95:	90 04 00 00 
  1a7c99:	62 f2 15 4a 66 ea    	vpblendmb %zmm2,%zmm13,%zmm5{%k2}
  1a7c9f:	62 72 7d 48 50 c5    	vpdpbusd %zmm5,%zmm0,%zmm8
  1a7ca5:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
  1a7ca9:	62 e1 fe 48 6f bc 24 	vmovdqu64 0x4d0(%rsp),%zmm23
  1a7cb0:	d0 04 00 00 
  1a7cb4:	62 62 45 42 66 f9    	vpblendmb %zmm1,%zmm23,%zmm31{%k2}
  1a7cba:	62 92 7d 48 50 ef    	vpdpbusd %zmm31,%zmm0,%zmm5
  1a7cc0:	c4 c1 7d 70 c3 f5    	vpshufd $0xf5,%ymm11,%ymm0
  1a7cc6:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a7ccd:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a7cd3:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a7cd9:	62 e1 fe 48 6f 94 24 	vmovdqu64 0x650(%rsp),%zmm18
  1a7ce0:	50 06 00 00 
  1a7ce4:	62 92 6d 42 66 db    	vpblendmb %zmm27,%zmm18,%zmm3{%k2}
  1a7cea:	62 72 7d 48 50 c3    	vpdpbusd %zmm3,%zmm0,%zmm8
  1a7cf0:	45 0f b6 7c 1e 07    	movzbl 0x7(%r14,%rbx,1),%r15d
  1a7cf6:	41 0f b6 74 1e 06    	movzbl 0x6(%r14,%rbx,1),%esi
  1a7cfc:	41 0f b6 7c 1e 05    	movzbl 0x5(%r14,%rbx,1),%edi
  1a7d02:	41 0f b6 6c 1e 04    	movzbl 0x4(%r14,%rbx,1),%ebp
  1a7d08:	41 0f b6 54 1e 02    	movzbl 0x2(%r14,%rbx,1),%edx
  1a7d0e:	c4 c1 7a 10 1c ac    	vmovss (%r12,%rbp,4),%xmm3
  1a7d14:	c4 c3 61 21 1c bc 10 	vinsertps $0x10,(%r12,%rdi,4),%xmm3,%xmm3
  1a7d1b:	41 0f b6 7c 1e 01    	movzbl 0x1(%r14,%rbx,1),%edi
  1a7d21:	c4 c3 61 21 1c b4 20 	vinsertps $0x20,(%r12,%rsi,4),%xmm3,%xmm3
  1a7d28:	41 0f b6 34 1e       	movzbl (%r14,%rbx,1),%esi
  1a7d2d:	62 41 7e 08 10 3c b4 	vmovss (%r12,%rsi,4),%xmm31
  1a7d34:	62 43 05 00 21 3c bc 	vinsertps $0x10,(%r12,%rdi,4),%xmm31,%xmm31
  1a7d3b:	10 
  1a7d3c:	41 0f b6 74 1d 05    	movzbl 0x5(%r13,%rbx,1),%esi
  1a7d42:	62 43 05 00 21 3c 94 	vinsertps $0x20,(%r12,%rdx,4),%xmm31,%xmm31
  1a7d49:	20 
  1a7d4a:	41 0f b6 54 1d 04    	movzbl 0x4(%r13,%rbx,1),%edx
  1a7d50:	62 41 7e 08 10 34 94 	vmovss (%r12,%rdx,4),%xmm30
  1a7d57:	62 43 0d 00 21 34 b4 	vinsertps $0x10,(%r12,%rsi,4),%xmm30,%xmm30
  1a7d5e:	10 
  1a7d5f:	c4 83 61 21 1c bc 30 	vinsertps $0x30,(%r12,%r15,4),%xmm3,%xmm3
  1a7d66:	41 0f b6 54 1e 03    	movzbl 0x3(%r14,%rbx,1),%edx
  1a7d6c:	41 0f b6 74 1d 06    	movzbl 0x6(%r13,%rbx,1),%esi
  1a7d72:	62 43 05 00 21 3c 94 	vinsertps $0x30,(%r12,%rdx,4),%xmm31,%xmm31
  1a7d79:	30 
  1a7d7a:	41 0f b6 54 1d 01    	movzbl 0x1(%r13,%rbx,1),%edx
  1a7d80:	62 43 0d 00 21 34 b4 	vinsertps $0x20,(%r12,%rsi,4),%xmm30,%xmm30
  1a7d87:	20 
  1a7d88:	41 0f b6 74 1d 00    	movzbl 0x0(%r13,%rbx,1),%esi
  1a7d8e:	62 41 7e 08 10 2c b4 	vmovss (%r12,%rsi,4),%xmm29
  1a7d95:	62 43 15 00 21 2c 94 	vinsertps $0x10,(%r12,%rdx,4),%xmm29,%xmm29
  1a7d9c:	10 
  1a7d9d:	41 0f b6 54 1d 02    	movzbl 0x2(%r13,%rbx,1),%edx
  1a7da3:	62 43 15 00 21 2c 94 	vinsertps $0x20,(%r12,%rdx,4),%xmm29,%xmm29
  1a7daa:	20 
  1a7dab:	62 71 fe 48 6f 9c 24 	vmovdqu64 0x5d0(%rsp),%zmm11
  1a7db2:	d0 05 00 00 
  1a7db6:	62 71 fe 48 6f 94 24 	vmovdqu64 0x610(%rsp),%zmm10
  1a7dbd:	10 06 00 00 
  1a7dc1:	62 42 2d 4a 66 e3    	vpblendmb %zmm11,%zmm10,%zmm28{%k2}
  1a7dc7:	62 92 7d 48 50 ec    	vpdpbusd %zmm28,%zmm0,%zmm5
  1a7dcd:	41 0f b6 54 1d 07    	movzbl 0x7(%r13,%rbx,1),%edx
  1a7dd3:	62 d3 0d 00 21 04 94 	vinsertps $0x30,(%r12,%rdx,4),%xmm30,%xmm0
  1a7dda:	30 
  1a7ddb:	62 61 7c 48 10 b4 24 	vmovups 0x6d0(%rsp),%zmm30
  1a7de2:	d0 06 00 00 
  1a7de6:	62 f3 05 20 18 db 01 	vinsertf32x4 $0x1,%xmm3,%ymm31,%ymm3
  1a7ded:	62 61 7c 48 10 bc 24 	vmovups 0x690(%rsp),%zmm31
  1a7df4:	90 06 00 00 
  1a7df8:	41 0f b6 54 1d 03    	movzbl 0x3(%r13,%rbx,1),%edx
  1a7dfe:	62 43 15 00 21 24 94 	vinsertps $0x30,(%r12,%rdx,4),%xmm29,%xmm28
  1a7e05:	30 
  1a7e06:	62 f3 1d 20 18 c0 01 	vinsertf32x4 $0x1,%xmm0,%ymm28,%ymm0
  1a7e0d:	62 f1 7c 48 10 94 24 	vmovups 0x710(%rsp),%zmm2
  1a7e14:	10 07 00 00 
  1a7e18:	62 f3 fd 48 1a c3 01 	vinsertf64x4 $0x1,%ymm3,%zmm0,%zmm0
  1a7e1f:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xed0(%rsp),%zmm1
  1a7e26:	d0 0e 00 00 
  1a7e2a:	62 f1 75 48 fe 9c 24 	vpaddd 0x10d0(%rsp),%zmm1,%zmm3
  1a7e31:	d0 10 00 00 
  1a7e35:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xe50(%rsp),%zmm1
  1a7e3c:	50 0e 00 00 
  1a7e40:	62 e1 75 48 fe b4 24 	vpaddd 0x1090(%rsp),%zmm1,%zmm22
  1a7e47:	90 10 00 00 
  1a7e4b:	62 c1 7e 89 6f 2c 1a 	vmovdqu32 (%r10,%rbx,1),%xmm21{%k1}{z}
  1a7e52:	62 a2 7d 08 13 ed    	vcvtph2ps %xmm21,%xmm21
  1a7e58:	62 21 e5 48 6c d6    	vpunpcklqdq %zmm22,%zmm3,%zmm26
  1a7e5e:	62 01 7c 48 5b d2    	vcvtdq2ps %zmm26,%zmm26
  1a7e64:	62 22 7d 48 18 dd    	vbroadcastss %xmm21,%zmm27
  1a7e6a:	62 01 7c 48 59 db    	vmulps %zmm27,%zmm0,%zmm27
  1a7e70:	62 f1 7c 48 10 8c 24 	vmovups 0x750(%rsp),%zmm1
  1a7e77:	50 07 00 00 
  1a7e7b:	62 92 2d 40 b8 cb    	vfmadd231ps %zmm27,%zmm26,%zmm1
  1a7e81:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x750(%rsp)
  1a7e88:	50 07 00 00 
  1a7e8c:	c4 c1 7d 70 e6 f5    	vpshufd $0xf5,%ymm14,%ymm4
  1a7e92:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a7e99:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a7e9f:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a7ea5:	62 e1 fe 48 6f 9c 24 	vmovdqu64 0x350(%rsp),%zmm19
  1a7eac:	50 03 00 00 
  1a7eb0:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x8d0(%rsp),%zmm1
  1a7eb7:	d0 08 00 00 
  1a7ebb:	62 22 75 4a 66 d3    	vpblendmb %zmm19,%zmm1,%zmm26{%k2}
  1a7ec1:	62 61 fe 48 6f a4 24 	vmovdqu64 0x290(%rsp),%zmm28
  1a7ec8:	90 02 00 00 
  1a7ecc:	62 62 1d 42 66 9c 24 	vpblendmb 0x910(%rsp),%zmm28,%zmm27{%k2}
  1a7ed3:	10 09 00 00 
  1a7ed7:	62 12 5d 48 50 c2    	vpdpbusd %zmm26,%zmm4,%zmm8
  1a7edd:	62 92 5d 48 50 eb    	vpdpbusd %zmm27,%zmm4,%zmm5
  1a7ee3:	62 b1 e5 48 6d de    	vpunpckhqdq %zmm22,%zmm3,%zmm3
  1a7ee9:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  1a7eef:	62 b1 7e 08 16 e5    	vmovshdup %xmm21,%xmm4
  1a7ef5:	62 f2 7d 48 18 e4    	vbroadcastss %xmm4,%zmm4
  1a7efb:	62 f1 7c 48 59 e4    	vmulps %zmm4,%zmm0,%zmm4
  1a7f01:	62 62 65 48 b8 fc    	vfmadd231ps %zmm4,%zmm3,%zmm31
  1a7f07:	62 61 7c 48 11 bc 24 	vmovups %zmm31,0x690(%rsp)
  1a7f0e:	90 06 00 00 
  1a7f12:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xe90(%rsp),%zmm1
  1a7f19:	90 0e 00 00 
  1a7f1d:	62 f1 75 48 fe a4 24 	vpaddd 0x1050(%rsp),%zmm1,%zmm4
  1a7f24:	50 10 00 00 
  1a7f28:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xe10(%rsp),%zmm1
  1a7f2f:	10 0e 00 00 
  1a7f33:	62 e1 75 48 fe b4 24 	vpaddd 0x1010(%rsp),%zmm1,%zmm22
  1a7f3a:	10 10 00 00 
  1a7f3e:	62 b1 dd 48 6c de    	vpunpcklqdq %zmm22,%zmm4,%zmm3
  1a7f44:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  1a7f4a:	62 31 d5 00 c6 fd 01 	vshufpd $0x1,%xmm21,%xmm21,%xmm15
  1a7f51:	62 52 7d 48 18 ff    	vbroadcastss %xmm15,%zmm15
  1a7f57:	62 51 7c 48 59 ff    	vmulps %zmm15,%zmm0,%zmm15
  1a7f5d:	62 42 65 48 b8 f7    	vfmadd231ps %zmm15,%zmm3,%zmm30
  1a7f63:	62 61 7c 48 11 b4 24 	vmovups %zmm30,0x6d0(%rsp)
  1a7f6a:	d0 06 00 00 
  1a7f6e:	62 b1 7d 28 70 d9 f5 	vpshufd $0xf5,%ymm17,%ymm3
  1a7f75:	62 f3 e5 48 43 db 55 	vshufi64x2 $0x55,%zmm3,%zmm3,%zmm3
  1a7f7c:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a7f82:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a7f88:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x550(%rsp),%zmm6
  1a7f8f:	50 05 00 00 
  1a7f93:	62 71 fe 48 6f 8c 24 	vmovdqu64 0x510(%rsp),%zmm9
  1a7f9a:	10 05 00 00 
  1a7f9e:	62 f2 35 4a 66 fe    	vpblendmb %zmm6,%zmm9,%zmm7{%k2}
  1a7fa4:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x390(%rsp),%zmm1
  1a7fab:	90 03 00 00 
  1a7faf:	62 72 75 4a 66 bc 24 	vpblendmb 0x1d0(%rsp),%zmm1,%zmm15{%k2}
  1a7fb6:	d0 01 00 00 
  1a7fba:	62 72 65 48 50 c7    	vpdpbusd %zmm7,%zmm3,%zmm8
  1a7fc0:	62 d2 65 48 50 ef    	vpdpbusd %zmm15,%zmm3,%zmm5
  1a7fc6:	62 61 fe 28 6f bc 18 	vmovdqu64 0x68(%rax,%rbx,1),%ymm31
  1a7fcd:	68 00 00 00 
  1a7fd1:	62 91 7d 08 70 df a0 	vpshufd $0xa0,%xmm31,%xmm3
  1a7fd8:	62 f3 e5 48 43 db 00 	vshufi64x2 $0x0,%zmm3,%zmm3,%zmm3
  1a7fdf:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a7fe5:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x950(%rsp),%zmm1
  1a7fec:	50 09 00 00 
  1a7ff0:	62 e2 75 4a 66 8c 24 	vpblendmb 0x990(%rsp),%zmm1,%zmm17{%k2}
  1a7ff7:	90 09 00 00 
  1a7ffb:	62 f2 7d 48 1c fb    	vpabsb %zmm3,%zmm7
  1a8001:	c4 41 01 ef ff       	vpxor  %xmm15,%xmm15,%xmm15
  1a8006:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x2d0(%rsp),%zmm1
  1a800d:	d0 02 00 00 
  1a8011:	62 f2 75 4a 66 9c 24 	vpblendmb 0x110(%rsp),%zmm1,%zmm3{%k2}
  1a8018:	10 01 00 00 
  1a801c:	62 32 45 48 50 f9    	vpdpbusd %zmm17,%zmm7,%zmm15
  1a8022:	62 a1 75 00 ef c9    	vpxord %xmm17,%xmm17,%xmm17
  1a8028:	62 e2 45 48 50 cb    	vpdpbusd %zmm3,%zmm7,%zmm17
  1a802e:	62 61 fe 28 6f ac 18 	vmovdqu64 0x48(%rax,%rbx,1),%ymm29
  1a8035:	48 00 00 00 
  1a8039:	62 91 7d 08 70 dd a0 	vpshufd $0xa0,%xmm29,%xmm3
  1a8040:	62 f3 e5 48 43 db 00 	vshufi64x2 $0x0,%zmm3,%zmm3,%zmm3
  1a8047:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a804d:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a8053:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x210(%rsp),%zmm1
  1a805a:	10 02 00 00 
  1a805e:	62 b2 75 4a 66 fc    	vpblendmb %zmm20,%zmm1,%zmm7{%k2}
  1a8064:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x450(%rsp),%zmm1
  1a806b:	50 04 00 00 
  1a806f:	62 62 75 4a 66 94 24 	vpblendmb 0x250(%rsp),%zmm1,%zmm26{%k2}
  1a8076:	50 02 00 00 
  1a807a:	62 72 65 48 50 ff    	vpdpbusd %zmm7,%zmm3,%zmm15
  1a8080:	62 82 65 48 50 ca    	vpdpbusd %zmm26,%zmm3,%zmm17
  1a8086:	62 b1 dd 48 6d de    	vpunpckhqdq %zmm22,%zmm4,%zmm3
  1a808c:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  1a8092:	62 b1 54 00 c6 e5 ff 	vshufps $0xff,%xmm21,%xmm21,%xmm4
  1a8099:	62 f2 7d 48 18 e4    	vbroadcastss %xmm4,%zmm4
  1a809f:	62 f1 7c 48 59 e4    	vmulps %zmm4,%zmm0,%zmm4
  1a80a5:	62 f2 65 48 b8 d4    	vfmadd231ps %zmm4,%zmm3,%zmm2
  1a80ab:	62 f1 7c 48 11 94 24 	vmovups %zmm2,0x710(%rsp)
  1a80b2:	10 07 00 00 
  1a80b6:	c5 fe 6f 7c 18 28    	vmovdqu 0x28(%rax,%rbx,1),%ymm7
  1a80bc:	c5 f9 70 df a0       	vpshufd $0xa0,%xmm7,%xmm3
  1a80c1:	62 f3 e5 48 43 db 00 	vshufi64x2 $0x0,%zmm3,%zmm3,%zmm3
  1a80c8:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a80ce:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a80d4:	62 61 fe 48 6f b4 24 	vmovdqu64 0xb90(%rsp),%zmm30
  1a80db:	90 0b 00 00 
  1a80df:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xa50(%rsp),%zmm1
  1a80e6:	50 0a 00 00 
  1a80ea:	62 92 75 4a 66 e6    	vpblendmb %zmm30,%zmm1,%zmm4{%k2}
  1a80f0:	62 72 65 48 50 fc    	vpdpbusd %zmm4,%zmm3,%zmm15
  1a80f6:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x150(%rsp),%zmm1
  1a80fd:	50 01 00 00 
  1a8101:	62 92 75 4a 66 e0    	vpblendmb %zmm24,%zmm1,%zmm4{%k2}
  1a8107:	62 e2 65 48 50 cc    	vpdpbusd %zmm4,%zmm3,%zmm17
  1a810d:	c5 fe 6f 64 18 08    	vmovdqu 0x8(%rax,%rbx,1),%ymm4
  1a8113:	c5 f9 70 dc a0       	vpshufd $0xa0,%xmm4,%xmm3
  1a8118:	62 f3 e5 48 43 db 00 	vshufi64x2 $0x0,%zmm3,%zmm3,%zmm3
  1a811f:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a8125:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a812b:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0x410(%rsp),%zmm1
  1a8132:	10 04 00 00 
  1a8136:	62 82 75 4a 66 e9    	vpblendmb %zmm25,%zmm1,%zmm21{%k2}
  1a813c:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xa10(%rsp),%zmm1
  1a8143:	10 0a 00 00 
  1a8147:	62 c2 75 4a 66 f4    	vpblendmb %zmm12,%zmm1,%zmm22{%k2}
  1a814d:	62 32 65 48 50 fd    	vpdpbusd %zmm21,%zmm3,%zmm15
  1a8153:	62 a2 65 48 50 ce    	vpdpbusd %zmm22,%zmm3,%zmm17
  1a8159:	62 91 7d 08 70 df f5 	vpshufd $0xf5,%xmm31,%xmm3
  1a8160:	62 f3 e5 48 43 db 00 	vshufi64x2 $0x0,%zmm3,%zmm3,%zmm3
  1a8167:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a816d:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a8173:	62 e2 15 4a 66 b4 24 	vpblendmb 0x9d0(%rsp),%zmm13,%zmm22{%k2}
  1a817a:	d0 09 00 00 
  1a817e:	62 a1 55 00 ef ed    	vpxord %xmm21,%xmm21,%xmm21
  1a8184:	62 a2 65 48 50 ee    	vpdpbusd %zmm22,%zmm3,%zmm21
  1a818a:	62 62 45 42 66 94 24 	vpblendmb 0x3d0(%rsp),%zmm23,%zmm26{%k2}
  1a8191:	d0 03 00 00 
  1a8195:	62 a1 4d 00 ef f6    	vpxord %xmm22,%xmm22,%xmm22
  1a819b:	62 82 65 48 50 f2    	vpdpbusd %zmm26,%zmm3,%zmm22
  1a81a1:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xdd0(%rsp),%zmm1
  1a81a8:	d0 0d 00 00 
  1a81ac:	62 f1 75 48 fe 9c 24 	vpaddd 0xfd0(%rsp),%zmm1,%zmm3
  1a81b3:	d0 0f 00 00 
  1a81b7:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xbd0(%rsp),%zmm1
  1a81be:	d0 0b 00 00 
  1a81c2:	62 e1 75 48 fe a4 24 	vpaddd 0xf90(%rsp),%zmm1,%zmm20
  1a81c9:	90 0f 00 00 
  1a81cd:	62 c1 7e 89 6f 3c 19 	vmovdqu32 (%r9,%rbx,1),%xmm23{%k1}{z}
  1a81d4:	62 a2 7d 08 13 ff    	vcvtph2ps %xmm23,%xmm23
  1a81da:	62 21 e5 48 6c c4    	vpunpcklqdq %zmm20,%zmm3,%zmm24
  1a81e0:	62 01 7c 48 5b c0    	vcvtdq2ps %zmm24,%zmm24
  1a81e6:	62 22 7d 48 18 cf    	vbroadcastss %xmm23,%zmm25
  1a81ec:	62 01 7c 48 59 c9    	vmulps %zmm25,%zmm0,%zmm25
  1a81f2:	62 f1 7c 48 10 8c 24 	vmovups 0x790(%rsp),%zmm1
  1a81f9:	90 07 00 00 
  1a81fd:	62 92 3d 40 b8 c9    	vfmadd231ps %zmm25,%zmm24,%zmm1
  1a8203:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x790(%rsp)
  1a820a:	90 07 00 00 
  1a820e:	62 01 7d 08 70 c5 f5 	vpshufd $0xf5,%xmm29,%xmm24
  1a8215:	62 03 bd 40 43 c0 00 	vshufi64x2 $0x0,%zmm24,%zmm24,%zmm24
  1a821c:	62 92 7e 48 29 d0    	vpmovb2m %zmm24,%k2
  1a8222:	62 02 7d 48 1c c0    	vpabsb %zmm24,%zmm24
  1a8228:	62 62 6d 42 66 8c 24 	vpblendmb 0x190(%rsp),%zmm18,%zmm25{%k2}
  1a822f:	90 01 00 00 
  1a8233:	62 42 2d 4a 66 d3    	vpblendmb %zmm11,%zmm10,%zmm26{%k2}
  1a8239:	62 82 3d 40 50 e9    	vpdpbusd %zmm25,%zmm24,%zmm21
  1a823f:	62 82 3d 40 50 f2    	vpdpbusd %zmm26,%zmm24,%zmm22
  1a8245:	62 b1 e5 48 6d dc    	vpunpckhqdq %zmm20,%zmm3,%zmm3
  1a824b:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  1a8251:	62 a1 7e 08 16 e7    	vmovshdup %xmm23,%xmm20
  1a8257:	62 a2 7d 48 18 e4    	vbroadcastss %xmm20,%zmm20
  1a825d:	62 a1 7c 48 59 e4    	vmulps %zmm20,%zmm0,%zmm20
  1a8263:	62 f1 7c 48 10 8c 24 	vmovups 0x7d0(%rsp),%zmm1
  1a826a:	d0 07 00 00 
  1a826e:	62 b2 65 48 b8 cc    	vfmadd231ps %zmm20,%zmm3,%zmm1
  1a8274:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x7d0(%rsp)
  1a827b:	d0 07 00 00 
  1a827f:	62 a1 5c 00 57 e4    	vxorps %xmm20,%xmm20,%xmm20
  1a8285:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xcd0(%rsp),%zmm1
  1a828c:	d0 0c 00 00 
  1a8290:	62 f1 75 48 fe 9c 24 	vpaddd 0xf50(%rsp),%zmm1,%zmm3
  1a8297:	50 0f 00 00 
  1a829b:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xc50(%rsp),%zmm1
  1a82a2:	50 0c 00 00 
  1a82a6:	62 71 75 48 fe 9c 24 	vpaddd 0xf10(%rsp),%zmm1,%zmm11
  1a82ad:	10 0f 00 00 
  1a82b1:	62 51 e5 48 6c eb    	vpunpcklqdq %zmm11,%zmm3,%zmm13
  1a82b7:	62 51 7c 48 5b ed    	vcvtdq2ps %zmm13,%zmm13
  1a82bd:	62 31 c5 00 c6 f7 01 	vshufpd $0x1,%xmm23,%xmm23,%xmm14
  1a82c4:	62 52 7d 48 18 f6    	vbroadcastss %xmm14,%zmm14
  1a82ca:	62 51 7c 48 59 f6    	vmulps %zmm14,%zmm0,%zmm14
  1a82d0:	62 f1 7c 48 10 8c 24 	vmovups 0x810(%rsp),%zmm1
  1a82d7:	10 08 00 00 
  1a82db:	62 d2 15 48 b8 ce    	vfmadd231ps %zmm14,%zmm13,%zmm1
  1a82e1:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x810(%rsp)
  1a82e8:	10 08 00 00 
  1a82ec:	c5 79 70 ef f5       	vpshufd $0xf5,%xmm7,%xmm13
  1a82f1:	62 53 95 48 43 ed 00 	vshufi64x2 $0x0,%zmm13,%zmm13,%zmm13
  1a82f8:	62 d2 7e 48 29 d5    	vpmovb2m %zmm13,%k2
  1a82fe:	62 52 7d 48 1c ed    	vpabsb %zmm13,%zmm13
  1a8304:	62 61 fe 48 6f 9c 24 	vmovdqu64 0x8d0(%rsp),%zmm27
  1a830b:	d0 08 00 00 
  1a830f:	62 32 25 42 66 f3    	vpblendmb %zmm19,%zmm27,%zmm14{%k2}
  1a8315:	62 e1 fe 48 6f 9c 24 	vmovdqu64 0x910(%rsp),%zmm19
  1a831c:	10 09 00 00 
  1a8320:	62 a2 1d 42 66 d3    	vpblendmb %zmm19,%zmm28,%zmm18{%k2}
  1a8326:	62 c2 15 48 50 ee    	vpdpbusd %zmm14,%zmm13,%zmm21
  1a832c:	62 a2 15 48 50 f2    	vpdpbusd %zmm18,%zmm13,%zmm22
  1a8332:	62 61 7c 48 10 84 24 	vmovups 0x1110(%rsp),%zmm24
  1a8339:	10 11 00 00 
  1a833d:	62 d1 e5 48 6d db    	vpunpckhqdq %zmm11,%zmm3,%zmm3
  1a8343:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  1a8349:	62 31 44 00 c6 df ff 	vshufps $0xff,%xmm23,%xmm23,%xmm11
  1a8350:	62 52 7d 48 18 db    	vbroadcastss %xmm11,%zmm11
  1a8356:	62 51 7c 48 59 db    	vmulps %zmm11,%zmm0,%zmm11
  1a835c:	62 f1 7c 48 10 8c 24 	vmovups 0x850(%rsp),%zmm1
  1a8363:	50 08 00 00 
  1a8367:	62 d2 65 48 b8 cb    	vfmadd231ps %zmm11,%zmm3,%zmm1
  1a836d:	62 f1 7c 48 11 8c 24 	vmovups %zmm1,0x850(%rsp)
  1a8374:	50 08 00 00 
  1a8378:	c5 f9 70 dc f5       	vpshufd $0xf5,%xmm4,%xmm3
  1a837d:	62 f3 e5 48 43 db 00 	vshufi64x2 $0x0,%zmm3,%zmm3,%zmm3
  1a8384:	62 f2 7e 48 29 d3    	vpmovb2m %zmm3,%k2
  1a838a:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a8390:	62 72 35 4a 66 de    	vpblendmb %zmm6,%zmm9,%zmm11{%k2}
  1a8396:	62 e1 fe 48 6f bc 24 	vmovdqu64 0x1d0(%rsp),%zmm23
  1a839d:	d0 01 00 00 
  1a83a1:	62 61 fe 48 6f 8c 24 	vmovdqu64 0x390(%rsp),%zmm25
  1a83a8:	90 03 00 00 
  1a83ac:	62 32 35 42 66 ef    	vpblendmb %zmm23,%zmm25,%zmm13{%k2}
  1a83b2:	62 c2 65 48 50 eb    	vpdpbusd %zmm11,%zmm3,%zmm21
  1a83b8:	62 c2 65 48 50 f5    	vpdpbusd %zmm13,%zmm3,%zmm22
  1a83be:	62 f1 fe 48 6f 8c 24 	vmovdqu64 0xc10(%rsp),%zmm1
  1a83c5:	10 0c 00 00 
  1a83c9:	62 f1 75 48 fe 94 24 	vpaddd 0xd10(%rsp),%zmm1,%zmm2
  1a83d0:	10 0d 00 00 
  1a83d4:	62 f1 7d 40 fe 9c 24 	vpaddd 0xc90(%rsp),%zmm16,%zmm3
  1a83db:	90 0c 00 00 
  1a83df:	62 d1 7e 89 6f 0c 18 	vmovdqu32 (%r8,%rbx,1),%xmm1{%k1}{z}
  1a83e6:	c4 e2 79 13 c9       	vcvtph2ps %xmm1,%xmm1
  1a83eb:	62 71 ed 48 6c cb    	vpunpcklqdq %zmm3,%zmm2,%zmm9
  1a83f1:	62 51 7c 48 5b c9    	vcvtdq2ps %zmm9,%zmm9
  1a83f7:	62 72 7d 48 18 d1    	vbroadcastss %xmm1,%zmm10
  1a83fd:	62 51 7c 48 59 d2    	vmulps %zmm10,%zmm0,%zmm10
  1a8403:	62 42 35 48 b8 c2    	vfmadd231ps %zmm10,%zmm9,%zmm24
  1a8409:	62 11 7d 28 70 cf a0 	vpshufd $0xa0,%ymm31,%ymm9
  1a8410:	62 53 b5 48 43 c9 55 	vshufi64x2 $0x55,%zmm9,%zmm9,%zmm9
  1a8417:	62 d2 7e 48 29 d1    	vpmovb2m %zmm9,%k2
  1a841d:	62 71 fe 48 6f a4 24 	vmovdqu64 0x950(%rsp),%zmm12
  1a8424:	50 09 00 00 
  1a8428:	62 71 7f 4a 6f a4 24 	vmovdqu8 0x990(%rsp),%zmm12{%k2}
  1a842f:	90 09 00 00 
  1a8433:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x2d0(%rsp),%zmm6
  1a843a:	d0 02 00 00 
  1a843e:	62 f1 7f 4a 6f b4 24 	vmovdqu8 0x110(%rsp),%zmm6{%k2}
  1a8445:	10 01 00 00 
  1a8449:	62 52 7d 48 1c d1    	vpabsb %zmm9,%zmm10
  1a844f:	c4 41 21 ef db       	vpxor  %xmm11,%xmm11,%xmm11
  1a8454:	62 52 2d 48 50 dc    	vpdpbusd %zmm12,%zmm10,%zmm11
  1a845a:	c4 41 31 ef c9       	vpxor  %xmm9,%xmm9,%xmm9
  1a845f:	62 72 2d 48 50 ce    	vpdpbusd %zmm6,%zmm10,%zmm9
  1a8465:	62 11 7d 28 70 d5 a0 	vpshufd $0xa0,%ymm29,%ymm10
  1a846c:	62 53 ad 48 43 d2 55 	vshufi64x2 $0x55,%zmm10,%zmm10,%zmm10
  1a8473:	62 d2 7e 48 29 d2    	vpmovb2m %zmm10,%k2
  1a8479:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x210(%rsp),%zmm6
  1a8480:	10 02 00 00 
  1a8484:	62 f1 7f 4a 6f b4 24 	vmovdqu8 0xb50(%rsp),%zmm6{%k2}
  1a848b:	50 0b 00 00 
  1a848f:	62 52 7d 48 1c d2    	vpabsb %zmm10,%zmm10
  1a8495:	62 72 2d 48 50 de    	vpdpbusd %zmm6,%zmm10,%zmm11
  1a849b:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x450(%rsp),%zmm6
  1a84a2:	50 04 00 00 
  1a84a6:	62 f1 7f 4a 6f b4 24 	vmovdqu8 0x250(%rsp),%zmm6{%k2}
  1a84ad:	50 02 00 00 
  1a84b1:	62 72 2d 48 50 ce    	vpdpbusd %zmm6,%zmm10,%zmm9
  1a84b7:	c5 7d 70 d7 a0       	vpshufd $0xa0,%ymm7,%ymm10
  1a84bc:	62 53 ad 48 43 d2 55 	vshufi64x2 $0x55,%zmm10,%zmm10,%zmm10
  1a84c3:	62 d2 7e 48 29 d2    	vpmovb2m %zmm10,%k2
  1a84c9:	62 f1 fe 48 6f b4 24 	vmovdqu64 0xa50(%rsp),%zmm6
  1a84d0:	50 0a 00 00 
  1a84d4:	62 91 7f 4a 6f f6    	vmovdqu8 %zmm30,%zmm6{%k2}
  1a84da:	62 71 fe 48 6f a4 24 	vmovdqu64 0x150(%rsp),%zmm12
  1a84e1:	50 01 00 00 
  1a84e5:	62 71 7f 4a 6f a4 24 	vmovdqu8 0x590(%rsp),%zmm12{%k2}
  1a84ec:	90 05 00 00 
  1a84f0:	c5 7d 70 ec a0       	vpshufd $0xa0,%ymm4,%ymm13
  1a84f5:	62 53 95 48 43 ed 55 	vshufi64x2 $0x55,%zmm13,%zmm13,%zmm13
  1a84fc:	62 d2 7e 48 29 d5    	vpmovb2m %zmm13,%k2
  1a8502:	62 61 fe 48 6f 94 24 	vmovdqu64 0x410(%rsp),%zmm26
  1a8509:	10 04 00 00 
  1a850d:	62 61 7f 4a 6f 94 24 	vmovdqu8 0x310(%rsp),%zmm26{%k2}
  1a8514:	10 03 00 00 
  1a8518:	62 61 fe 48 6f b4 24 	vmovdqu64 0xa10(%rsp),%zmm30
  1a851f:	10 0a 00 00 
  1a8523:	62 61 7f 4a 6f b4 24 	vmovdqu8 0x890(%rsp),%zmm30{%k2}
  1a852a:	90 08 00 00 
  1a852e:	62 11 7d 28 70 f7 f5 	vpshufd $0xf5,%ymm31,%ymm14
  1a8535:	62 53 8d 48 43 f6 55 	vshufi64x2 $0x55,%zmm14,%zmm14,%zmm14
  1a853c:	62 d2 7e 48 29 d6    	vpmovb2m %zmm14,%k2
  1a8542:	62 61 fe 48 6f a4 24 	vmovdqu64 0x490(%rsp),%zmm28
  1a8549:	90 04 00 00 
  1a854d:	62 61 7f 4a 6f a4 24 	vmovdqu8 0x9d0(%rsp),%zmm28{%k2}
  1a8554:	d0 09 00 00 
  1a8558:	62 61 fe 48 6f bc 24 	vmovdqu64 0x4d0(%rsp),%zmm31
  1a855f:	d0 04 00 00 
  1a8563:	62 61 7f 4a 6f bc 24 	vmovdqu8 0x3d0(%rsp),%zmm31{%k2}
  1a856a:	d0 03 00 00 
  1a856e:	c5 fd 70 ff f5       	vpshufd $0xf5,%ymm7,%ymm7
  1a8573:	62 f3 c5 48 43 ff 55 	vshufi64x2 $0x55,%zmm7,%zmm7,%zmm7
  1a857a:	62 f2 7e 48 29 d7    	vpmovb2m %zmm7,%k2
  1a8580:	62 61 7f 4a 6f 9c 24 	vmovdqu8 0x350(%rsp),%zmm27{%k2}
  1a8587:	50 03 00 00 
  1a858b:	62 e1 fe 48 6f 94 24 	vmovdqu64 0x290(%rsp),%zmm18
  1a8592:	90 02 00 00 
  1a8596:	62 a1 7f 4a 6f d3    	vmovdqu8 %zmm19,%zmm18{%k2}
  1a859c:	c5 fd 70 e4 f5       	vpshufd $0xf5,%ymm4,%ymm4
  1a85a1:	62 f3 dd 48 43 e4 55 	vshufi64x2 $0x55,%zmm4,%zmm4,%zmm4
  1a85a8:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  1a85ae:	62 e1 fe 48 6f 9c 24 	vmovdqu64 0x510(%rsp),%zmm19
  1a85b5:	10 05 00 00 
  1a85b9:	62 e1 7f 4a 6f 9c 24 	vmovdqu8 0x550(%rsp),%zmm19{%k2}
  1a85c0:	50 05 00 00 
  1a85c4:	62 21 7f 4a 6f cf    	vmovdqu8 %zmm23,%zmm25{%k2}
  1a85ca:	62 81 7d 28 70 c5 f5 	vpshufd $0xf5,%ymm29,%ymm16
  1a85d1:	62 a3 fd 40 43 c0 55 	vshufi64x2 $0x55,%zmm16,%zmm16,%zmm16
  1a85d8:	62 b2 7e 48 29 d0    	vpmovb2m %zmm16,%k2
  1a85de:	62 61 fe 48 6f ac 24 	vmovdqu64 0x650(%rsp),%zmm29
  1a85e5:	50 06 00 00 
  1a85e9:	62 61 7f 4a 6f ac 24 	vmovdqu8 0x190(%rsp),%zmm29{%k2}
  1a85f0:	90 01 00 00 
  1a85f4:	62 e1 fe 48 6f bc 24 	vmovdqu64 0x610(%rsp),%zmm23
  1a85fb:	10 06 00 00 
  1a85ff:	62 e1 7f 4a 6f bc 24 	vmovdqu8 0x5d0(%rsp),%zmm23{%k2}
  1a8606:	d0 05 00 00 
  1a860a:	62 52 7d 48 1c d2    	vpabsb %zmm10,%zmm10
  1a8610:	62 72 2d 48 50 de    	vpdpbusd %zmm6,%zmm10,%zmm11
  1a8616:	62 52 2d 48 50 cc    	vpdpbusd %zmm12,%zmm10,%zmm9
  1a861c:	62 52 7d 48 1c d5    	vpabsb %zmm13,%zmm10
  1a8622:	62 71 7c 48 10 ac 24 	vmovups 0x1150(%rsp),%zmm13
  1a8629:	50 11 00 00 
  1a862d:	62 12 2d 48 50 da    	vpdpbusd %zmm26,%zmm10,%zmm11
  1a8633:	62 12 2d 48 50 ce    	vpdpbusd %zmm30,%zmm10,%zmm9
  1a8639:	62 f1 ed 48 6d d3    	vpunpckhqdq %zmm3,%zmm2,%zmm2
  1a863f:	62 f1 7c 48 5b d2    	vcvtdq2ps %zmm2,%zmm2
  1a8645:	c5 fa 16 d9          	vmovshdup %xmm1,%xmm3
  1a8649:	62 f2 7d 48 18 db    	vbroadcastss %xmm3,%zmm3
  1a864f:	62 f1 7c 48 59 db    	vmulps %zmm3,%zmm0,%zmm3
  1a8655:	62 72 6d 48 b8 eb    	vfmadd231ps %zmm3,%zmm2,%zmm13
  1a865b:	62 d2 7d 48 1c d6    	vpabsb %zmm14,%zmm2
  1a8661:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  1a8665:	62 92 6d 48 50 dc    	vpdpbusd %zmm28,%zmm2,%zmm3
  1a866b:	c4 41 29 ef d2       	vpxor  %xmm10,%xmm10,%xmm10
  1a8670:	62 12 6d 48 50 d7    	vpdpbusd %zmm31,%zmm2,%zmm10
  1a8676:	62 f1 3d 48 fe 94 24 	vpaddd 0xd90(%rsp),%zmm8,%zmm2
  1a867d:	90 0d 00 00 
  1a8681:	62 71 7c 48 10 a4 24 	vmovups 0x1190(%rsp),%zmm12
  1a8688:	90 11 00 00 
  1a868c:	62 f1 55 48 fe ac 24 	vpaddd 0xd50(%rsp),%zmm5,%zmm5
  1a8693:	50 0d 00 00 
  1a8697:	62 b2 7d 48 1c f0    	vpabsb %zmm16,%zmm6
  1a869d:	62 92 4d 48 50 dd    	vpdpbusd %zmm29,%zmm6,%zmm3
  1a86a3:	62 32 4d 48 50 d7    	vpdpbusd %zmm23,%zmm6,%zmm10
  1a86a9:	62 f1 ed 48 6c f5    	vpunpcklqdq %zmm5,%zmm2,%zmm6
  1a86af:	62 f1 7c 48 5b f6    	vcvtdq2ps %zmm6,%zmm6
  1a86b5:	c5 71 c6 c1 01       	vshufpd $0x1,%xmm1,%xmm1,%xmm8
  1a86ba:	62 52 7d 48 18 c0    	vbroadcastss %xmm8,%zmm8
  1a86c0:	62 51 7c 48 59 c0    	vmulps %zmm8,%zmm0,%zmm8
  1a86c6:	62 52 4d 48 b8 e0    	vfmadd231ps %zmm8,%zmm6,%zmm12
  1a86cc:	62 71 fd 48 6f 05 6a 	vmovdqa64 -0x16f796(%rip),%zmm8        # 38f40 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x398>
  1a86d3:	08 e9 ff 
  1a86d6:	62 f2 7d 48 1c f7    	vpabsb %zmm7,%zmm6
  1a86dc:	62 92 4d 48 50 db    	vpdpbusd %zmm27,%zmm6,%zmm3
  1a86e2:	62 32 4d 48 50 d2    	vpdpbusd %zmm18,%zmm6,%zmm10
  1a86e8:	62 f2 7d 48 1c e4    	vpabsb %zmm4,%zmm4
  1a86ee:	62 b2 5d 48 50 db    	vpdpbusd %zmm19,%zmm4,%zmm3
  1a86f4:	62 12 5d 48 50 d1    	vpdpbusd %zmm25,%zmm4,%zmm10
  1a86fa:	62 d1 55 40 fe e7    	vpaddd %zmm15,%zmm21,%zmm4
  1a8700:	62 f1 ed 48 6d d5    	vpunpckhqdq %zmm5,%zmm2,%zmm2
  1a8706:	62 b1 4d 40 fe e9    	vpaddd %zmm17,%zmm22,%zmm5
  1a870c:	62 e1 fd 48 6f 0d 6a 	vmovdqa64 -0x16f796(%rip),%zmm17        # 38f80 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x3d8>
  1a8713:	08 e9 ff 
  1a8716:	62 d1 65 48 fe db    	vpaddd %zmm11,%zmm3,%zmm3
  1a871c:	62 71 7c 48 10 9c 24 	vmovups 0x11d0(%rsp),%zmm11
  1a8723:	d0 11 00 00 
  1a8727:	c5 f0 c6 c9 ff       	vshufps $0xff,%xmm1,%xmm1,%xmm1
  1a872c:	62 f1 7c 48 5b d2    	vcvtdq2ps %zmm2,%zmm2
  1a8732:	62 f2 7d 48 18 c9    	vbroadcastss %xmm1,%zmm1
  1a8738:	62 f1 7c 48 59 c9    	vmulps %zmm1,%zmm0,%zmm1
  1a873e:	62 d1 2d 48 fe f1    	vpaddd %zmm9,%zmm10,%zmm6
  1a8744:	62 71 7c 48 10 94 24 	vmovups 0x1210(%rsp),%zmm10
  1a874b:	10 12 00 00 
  1a874f:	62 72 6d 48 b8 d9    	vfmadd231ps %zmm1,%zmm2,%zmm11
  1a8755:	62 f1 7e 89 6f 0c 18 	vmovdqu32 (%rax,%rbx,1),%xmm1{%k1}{z}
  1a875c:	c4 e2 79 13 c9       	vcvtph2ps %xmm1,%xmm1
  1a8761:	62 f1 dd 48 6c d5    	vpunpcklqdq %zmm5,%zmm4,%zmm2
  1a8767:	62 f1 7c 48 5b d2    	vcvtdq2ps %zmm2,%zmm2
  1a876d:	62 f2 7d 48 18 f9    	vbroadcastss %xmm1,%zmm7
  1a8773:	62 f1 7c 48 59 ff    	vmulps %zmm7,%zmm0,%zmm7
  1a8779:	62 72 6d 48 b8 d7    	vfmadd231ps %zmm7,%zmm2,%zmm10
  1a877f:	62 f1 e5 48 6c d6    	vpunpcklqdq %zmm6,%zmm3,%zmm2
  1a8785:	c5 f1 c6 f9 01       	vshufpd $0x1,%xmm1,%xmm1,%xmm7
  1a878a:	62 f2 7d 48 18 ff    	vbroadcastss %xmm7,%zmm7
  1a8790:	62 f1 7c 48 5b d2    	vcvtdq2ps %zmm2,%zmm2
  1a8796:	62 f1 7c 48 59 ff    	vmulps %zmm7,%zmm0,%zmm7
  1a879c:	62 71 7c 48 10 8c 24 	vmovups 0xad0(%rsp),%zmm9
  1a87a3:	d0 0a 00 00 
  1a87a7:	62 72 6d 48 b8 cf    	vfmadd231ps %zmm7,%zmm2,%zmm9
  1a87ad:	62 71 7c 48 11 8c 24 	vmovups %zmm9,0xad0(%rsp)
  1a87b4:	d0 0a 00 00 
  1a87b8:	62 f1 fe 48 6f bc 24 	vmovdqu64 0xad0(%rsp),%zmm7
  1a87bf:	d0 0a 00 00 
  1a87c3:	62 f1 dd 48 6d d5    	vpunpckhqdq %zmm5,%zmm4,%zmm2
  1a87c9:	62 f1 7c 48 5b d2    	vcvtdq2ps %zmm2,%zmm2
  1a87cf:	c5 fa 16 e1          	vmovshdup %xmm1,%xmm4
  1a87d3:	62 f2 7d 48 18 e4    	vbroadcastss %xmm4,%zmm4
  1a87d9:	62 f1 7c 48 59 e4    	vmulps %zmm4,%zmm0,%zmm4
  1a87df:	62 f1 7c 48 10 ac 24 	vmovups 0xa90(%rsp),%zmm5
  1a87e6:	90 0a 00 00 
  1a87ea:	62 f2 6d 48 b8 ec    	vfmadd231ps %zmm4,%zmm2,%zmm5
  1a87f0:	62 f1 7c 48 11 ac 24 	vmovups %zmm5,0xa90(%rsp)
  1a87f7:	90 0a 00 00 
  1a87fb:	62 f1 7c 48 10 a4 24 	vmovups 0xa90(%rsp),%zmm4
  1a8802:	90 0a 00 00 
  1a8806:	c5 f0 c6 c9 ff       	vshufps $0xff,%xmm1,%xmm1,%xmm1
  1a880b:	62 f2 7d 48 18 c9    	vbroadcastss %xmm1,%zmm1
  1a8811:	62 f1 7c 48 59 c1    	vmulps %zmm1,%zmm0,%zmm0
  1a8817:	62 f1 e5 48 6d ce    	vpunpckhqdq %zmm6,%zmm3,%zmm1
  1a881d:	62 f1 7c 48 5b c9    	vcvtdq2ps %zmm1,%zmm1
  1a8823:	62 f1 7c 48 10 94 24 	vmovups 0xb10(%rsp),%zmm2
  1a882a:	10 0b 00 00 
  1a882e:	62 f2 75 48 b8 d0    	vfmadd231ps %zmm0,%zmm1,%zmm2
  1a8834:	62 f1 7c 48 11 94 24 	vmovups %zmm2,0xb10(%rsp)
  1a883b:	10 0b 00 00 
  1a883f:	62 f1 7c 48 10 84 24 	vmovups 0xb10(%rsp),%zmm0
  1a8846:	10 0b 00 00 
  1a884a:	48 81 c3 88 00 00 00 	add    $0x88,%rbx
  1a8851:	49 ff cb             	dec    %r11
  1a8854:	0f 85 56 e3 ff ff    	jne    1a6bb0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x320>
  1a885a:	48 89 ca             	mov    %rcx,%rdx
  1a885d:	48 c1 e2 05          	shl    $0x5,%rdx
  1a8861:	48 03 14 24          	add    (%rsp),%rdx
  1a8865:	48 8b b4 24 00 01 00 	mov    0x100(%rsp),%rsi
  1a886c:	00 
  1a886d:	62 f1 7c 48 10 8c 24 	vmovups 0x750(%rsp),%zmm1
  1a8874:	50 07 00 00 
  1a8878:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a887f:	48 8b b4 24 f8 00 00 	mov    0xf8(%rsp),%rsi
  1a8886:	00 
  1a8887:	62 f1 7c 48 10 8c 24 	vmovups 0x690(%rsp),%zmm1
  1a888e:	90 06 00 00 
  1a8892:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a8899:	48 8b b4 24 f0 00 00 	mov    0xf0(%rsp),%rsi
  1a88a0:	00 
  1a88a1:	62 f1 7c 48 10 8c 24 	vmovups 0x6d0(%rsp),%zmm1
  1a88a8:	d0 06 00 00 
  1a88ac:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a88b3:	48 8b b4 24 e8 00 00 	mov    0xe8(%rsp),%rsi
  1a88ba:	00 
  1a88bb:	62 f1 7c 48 10 8c 24 	vmovups 0x710(%rsp),%zmm1
  1a88c2:	10 07 00 00 
  1a88c6:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a88cd:	48 8b b4 24 e0 00 00 	mov    0xe0(%rsp),%rsi
  1a88d4:	00 
  1a88d5:	62 f1 7c 48 10 8c 24 	vmovups 0x790(%rsp),%zmm1
  1a88dc:	90 07 00 00 
  1a88e0:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a88e7:	48 8b b4 24 d8 00 00 	mov    0xd8(%rsp),%rsi
  1a88ee:	00 
  1a88ef:	62 f1 7c 48 10 8c 24 	vmovups 0x7d0(%rsp),%zmm1
  1a88f6:	d0 07 00 00 
  1a88fa:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a8901:	48 8b b4 24 d0 00 00 	mov    0xd0(%rsp),%rsi
  1a8908:	00 
  1a8909:	62 f1 7c 48 10 8c 24 	vmovups 0x810(%rsp),%zmm1
  1a8910:	10 08 00 00 
  1a8914:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a891b:	48 8b b4 24 c8 00 00 	mov    0xc8(%rsp),%rsi
  1a8922:	00 
  1a8923:	62 f1 7c 48 10 8c 24 	vmovups 0x850(%rsp),%zmm1
  1a892a:	50 08 00 00 
  1a892e:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  1a8935:	48 8b b4 24 c0 00 00 	mov    0xc0(%rsp),%rsi
  1a893c:	00 
  1a893d:	62 61 7c 48 11 04 b2 	vmovups %zmm24,(%rdx,%rsi,4)
  1a8944:	48 8b b4 24 b8 00 00 	mov    0xb8(%rsp),%rsi
  1a894b:	00 
  1a894c:	62 71 7c 48 11 2c b2 	vmovups %zmm13,(%rdx,%rsi,4)
  1a8953:	48 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%rsi
  1a895a:	00 
  1a895b:	62 71 7c 48 11 24 b2 	vmovups %zmm12,(%rdx,%rsi,4)
  1a8962:	48 8b b4 24 a8 00 00 	mov    0xa8(%rsp),%rsi
  1a8969:	00 
  1a896a:	62 71 7c 48 11 1c b2 	vmovups %zmm11,(%rdx,%rsi,4)
  1a8971:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
  1a8978:	00 
  1a8979:	62 71 7c 48 11 14 b2 	vmovups %zmm10,(%rdx,%rsi,4)
  1a8980:	48 8b b4 24 98 00 00 	mov    0x98(%rsp),%rsi
  1a8987:	00 
  1a8988:	62 f1 7c 48 11 24 b2 	vmovups %zmm4,(%rdx,%rsi,4)
  1a898f:	48 8b b4 24 90 00 00 	mov    0x90(%rsp),%rsi
  1a8996:	00 
  1a8997:	62 f1 fe 48 7f 3c b2 	vmovdqu64 %zmm7,(%rdx,%rsi,4)
  1a899e:	48 8b b4 24 88 00 00 	mov    0x88(%rsp),%rsi
  1a89a5:	00 
  1a89a6:	62 f1 7c 48 11 04 b2 	vmovups %zmm0,(%rdx,%rsi,4)
  1a89ad:	48 83 c1 02          	add    $0x2,%rcx
  1a89b1:	48 8b 94 24 80 00 00 	mov    0x80(%rsp),%rdx
  1a89b8:	00 
  1a89b9:	49 01 d6             	add    %rdx,%r14
  1a89bc:	49 01 d5             	add    %rdx,%r13
  1a89bf:	48 8b 94 24 08 01 00 	mov    0x108(%rsp),%rdx
  1a89c6:	00 
  1a89c7:	48 ff ca             	dec    %rdx
  1a89ca:	4c 8b 5c 24 28       	mov    0x28(%rsp),%r11
  1a89cf:	0f 85 4b e1 ff ff    	jne    1a6b20 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x290>
  1a89d5:	48 8b 6c 24 78       	mov    0x78(%rsp),%rbp
  1a89da:	48 83 c5 04          	add    $0x4,%rbp
  1a89de:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
  1a89e3:	48 01 c8             	add    %rcx,%rax
  1a89e6:	49 01 c8             	add    %rcx,%r8
  1a89e9:	49 01 c9             	add    %rcx,%r9
  1a89ec:	49 01 ca             	add    %rcx,%r10
  1a89ef:	48 3b 6c 24 18       	cmp    0x18(%rsp),%rbp
  1a89f4:	4c 8b 6c 24 48       	mov    0x48(%rsp),%r13
  1a89f9:	0f 82 c1 df ff ff    	jb     1a69c0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x130>
  1a89ff:	e9 f3 00 00 00       	jmp    1a8af7 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2267>
  1a8a04:	4c 89 e8             	mov    %r13,%rax
  1a8a07:	48 c1 e0 06          	shl    $0x6,%rax
  1a8a0b:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
  1a8a12:	00 
  1a8a13:	31 ed                	xor    %ebp,%ebp
  1a8a15:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a8a19:	48 8d 34 09          	lea    (%rcx,%rcx,1),%rsi
  1a8a1d:	48 8b 3c 24          	mov    (%rsp),%rdi
  1a8a21:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1a8a28:	0f 1f 84 00 00 00 00 
  1a8a2f:	00 
  1a8a30:	49 89 f8             	mov    %rdi,%r8
  1a8a33:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
  1a8a38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  1a8a3f:	00 
  1a8a40:	62 d1 7c 48 11 00    	vmovups %zmm0,(%r8)
  1a8a46:	49 8d 14 08          	lea    (%r8,%rcx,1),%rdx
  1a8a4a:	62 91 7c 48 11 04 a8 	vmovups %zmm0,(%r8,%r13,4)
  1a8a51:	48 01 ca             	add    %rcx,%rdx
  1a8a54:	62 91 7c 48 11 04 e8 	vmovups %zmm0,(%r8,%r13,8)
  1a8a5b:	62 b1 7c 48 11 04 aa 	vmovups %zmm0,(%rdx,%r13,4)
  1a8a62:	62 b1 7c 48 11 04 ea 	vmovups %zmm0,(%rdx,%r13,8)
  1a8a69:	48 01 f2             	add    %rsi,%rdx
  1a8a6c:	4e 8d 14 aa          	lea    (%rdx,%r13,4),%r10
  1a8a70:	62 b1 7c 48 11 04 aa 	vmovups %zmm0,(%rdx,%r13,4)
  1a8a77:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8a7e:	49 01 ca             	add    %rcx,%r10
  1a8a81:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8a88:	49 01 ca             	add    %rcx,%r10
  1a8a8b:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8a92:	49 01 ca             	add    %rcx,%r10
  1a8a95:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8a9c:	49 01 ca             	add    %rcx,%r10
  1a8a9f:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8aa6:	49 01 ca             	add    %rcx,%r10
  1a8aa9:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8ab0:	49 01 ca             	add    %rcx,%r10
  1a8ab3:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8aba:	49 01 ca             	add    %rcx,%r10
  1a8abd:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8ac4:	49 01 ca             	add    %rcx,%r10
  1a8ac7:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8ace:	49 01 ca             	add    %rcx,%r10
  1a8ad1:	62 b1 7c 48 11 04 11 	vmovups %zmm0,(%rcx,%r10,1)
  1a8ad8:	49 83 c0 40          	add    $0x40,%r8
  1a8adc:	49 ff c9             	dec    %r9
  1a8adf:	0f 85 5b ff ff ff    	jne    1a8a40 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x21b0>
  1a8ae5:	48 83 c5 04          	add    $0x4,%rbp
  1a8ae9:	48 01 c7             	add    %rax,%rdi
  1a8aec:	48 3b 6c 24 18       	cmp    0x18(%rsp),%rbp
  1a8af1:	0f 82 39 ff ff ff    	jb     1a8a30 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x21a0>
  1a8af7:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  1a8afc:	48 c1 e8 02          	shr    $0x2,%rax
  1a8b00:	48 89 84 24 90 02 00 	mov    %rax,0x290(%rsp)
  1a8b07:	00 
  1a8b08:	48 39 c5             	cmp    %rax,%rbp
  1a8b0b:	0f 83 31 09 00 00    	jae    1a9442 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bb2>
  1a8b11:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  1a8b16:	48 89 e9             	mov    %rbp,%rcx
  1a8b19:	48 85 c0             	test   %rax,%rax
  1a8b1c:	0f 84 78 09 00 00    	je     1a949a <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2c0a>
  1a8b22:	48 0f af c8          	imul   %rax,%rcx
  1a8b26:	48 89 ca             	mov    %rcx,%rdx
  1a8b29:	48 c1 e2 07          	shl    $0x7,%rdx
  1a8b2d:	48 8d 0c ca          	lea    (%rdx,%rcx,8),%rcx
  1a8b31:	48 03 4c 24 10       	add    0x10(%rsp),%rcx
  1a8b36:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  1a8b3b:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  1a8b40:	48 01 f2             	add    %rsi,%rdx
  1a8b43:	48 89 94 24 10 03 00 	mov    %rdx,0x310(%rsp)
  1a8b4a:	00 
  1a8b4b:	48 69 f8 10 01 00 00 	imul   $0x110,%rax,%rdi
  1a8b52:	62 61 fd 48 6f 35 e4 	vmovdqa64 -0x16fc1c(%rip),%zmm30        # 38f40 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x398>
  1a8b59:	03 e9 ff 
  1a8b5c:	62 f1 fd 48 6f 2d 1a 	vmovdqa64 -0x16fbe6(%rip),%zmm5        # 38f80 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x3d8>
  1a8b63:	04 e9 ff 
  1a8b66:	4c 8d 05 8f 32 ea ff 	lea    -0x15cd71(%rip),%r8        # 4bdfc <_RNvNtCs96HZWesffxA_4ggml8quants_k9IQ3S_GRID+0x1cac>
  1a8b6d:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  1a8b71:	b2 03                	mov    $0x3,%dl
  1a8b73:	c5 fb 92 ca          	kmovd  %edx,%k1
  1a8b77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  1a8b7e:	00 00 
  1a8b80:	4c 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%r9
  1a8b87:	00 
  1a8b88:	4d 0f af cd          	imul   %r13,%r9
  1a8b8c:	4c 8d 14 ad 01 00 00 	lea    0x1(,%rbp,4),%r10
  1a8b93:	00 
  1a8b94:	4d 0f af d5          	imul   %r13,%r10
  1a8b98:	4c 8d 1c ad 02 00 00 	lea    0x2(,%rbp,4),%r11
  1a8b9f:	00 
  1a8ba0:	4d 0f af dd          	imul   %r13,%r11
  1a8ba4:	48 89 e8             	mov    %rbp,%rax
  1a8ba7:	48 8d 1c ad 03 00 00 	lea    0x3(,%rbp,4),%rbx
  1a8bae:	00 
  1a8baf:	49 0f af dd          	imul   %r13,%rbx
  1a8bb3:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
  1a8bb8:	4c 8b bc 24 10 03 00 	mov    0x310(%rsp),%r15
  1a8bbf:	00 
  1a8bc0:	45 31 e4             	xor    %r12d,%r12d
  1a8bc3:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
  1a8bc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  1a8bcf:	00 
  1a8bd0:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a8bd4:	48 8b 6c 24 28       	mov    0x28(%rsp),%rbp
  1a8bd9:	31 f6                	xor    %esi,%esi
  1a8bdb:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  1a8bdf:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
  1a8be3:	c4 41 30 57 c9       	vxorps %xmm9,%xmm9,%xmm9
  1a8be8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  1a8bef:	00 
  1a8bf0:	62 71 7c 48 11 8c 24 	vmovups %zmm9,0x2d0(%rsp)
  1a8bf7:	d0 02 00 00 
  1a8bfb:	62 f1 7c 48 11 a4 24 	vmovups %zmm4,0x110(%rsp)
  1a8c02:	10 01 00 00 
  1a8c06:	62 f1 fe 48 7f b4 24 	vmovdqu64 %zmm6,0x150(%rsp)
  1a8c0d:	50 01 00 00 
  1a8c11:	62 f1 7c 48 11 84 24 	vmovups %zmm0,0x190(%rsp)
  1a8c18:	90 01 00 00 
  1a8c1c:	62 d1 fe 48 6f 84 37 	vmovdqu64 0x8(%r15,%rsi,1),%zmm0
  1a8c23:	08 00 00 00 
  1a8c27:	62 d1 fe 48 6f 8c 37 	vmovdqu64 0x48(%r15,%rsi,1),%zmm1
  1a8c2e:	48 00 00 00 
  1a8c32:	62 d1 fe 48 6f 94 36 	vmovdqu64 0x8(%r14,%rsi,1),%zmm2
  1a8c39:	08 00 00 00 
  1a8c3d:	62 d1 fe 48 6f 9c 36 	vmovdqu64 0x48(%r14,%rsi,1),%zmm3
  1a8c44:	48 00 00 00 
  1a8c48:	62 73 ed 48 43 c0 88 	vshufi64x2 $0x88,%zmm0,%zmm2,%zmm8
  1a8c4f:	62 71 fe 48 7f 84 24 	vmovdqu64 %zmm8,0x250(%rsp)
  1a8c56:	50 02 00 00 
  1a8c5a:	62 f3 e5 48 43 e1 88 	vshufi64x2 $0x88,%zmm1,%zmm3,%zmm4
  1a8c61:	62 f3 ed 48 43 f0 dd 	vshufi64x2 $0xdd,%zmm0,%zmm2,%zmm6
  1a8c68:	62 f1 fe 48 7f b4 24 	vmovdqu64 %zmm6,0x450(%rsp)
  1a8c6f:	50 04 00 00 
  1a8c73:	62 f3 e5 48 43 c1 dd 	vshufi64x2 $0xdd,%zmm1,%zmm3,%zmm0
  1a8c7a:	62 91 dd 48 db ce    	vpandq %zmm30,%zmm4,%zmm1
  1a8c80:	62 d1 6d 48 71 d0 04 	vpsrlw $0x4,%zmm8,%zmm2
  1a8c87:	62 e2 55 48 00 d9    	vpshufb %zmm1,%zmm5,%zmm19
  1a8c8d:	62 91 ed 48 db de    	vpandq %zmm30,%zmm2,%zmm3
  1a8c93:	62 f1 75 48 71 d4 04 	vpsrlw $0x4,%zmm4,%zmm1
  1a8c9a:	62 91 fd 48 db e6    	vpandq %zmm30,%zmm0,%zmm4
  1a8ca0:	62 91 f5 48 db d6    	vpandq %zmm30,%zmm1,%zmm2
  1a8ca6:	62 e2 55 48 00 f4    	vpshufb %zmm4,%zmm5,%zmm22
  1a8cac:	62 f1 75 48 71 d6 04 	vpsrlw $0x4,%zmm6,%zmm1
  1a8cb3:	62 91 f5 48 db ce    	vpandq %zmm30,%zmm1,%zmm1
  1a8cb9:	62 e2 55 48 00 cb    	vpshufb %zmm3,%zmm5,%zmm17
  1a8cbf:	62 f1 7d 48 71 d0 04 	vpsrlw $0x4,%zmm0,%zmm0
  1a8cc6:	62 91 fd 48 db c6    	vpandq %zmm30,%zmm0,%zmm0
  1a8ccc:	c5 7e 6f 44 31 28    	vmovdqu 0x28(%rcx,%rsi,1),%ymm8
  1a8cd2:	62 f2 55 48 00 e2    	vpshufb %zmm2,%zmm5,%zmm4
  1a8cd8:	c5 7e 6f 6c 31 48    	vmovdqu 0x48(%rcx,%rsi,1),%ymm13
  1a8cde:	62 e1 fe 28 6f 84 31 	vmovdqu64 0x68(%rcx,%rsi,1),%ymm16
  1a8ce5:	68 00 00 00 
  1a8ce9:	62 b1 7d 08 70 d0 a0 	vpshufd $0xa0,%xmm16,%xmm2
  1a8cf0:	62 f2 55 48 00 f1    	vpshufb %zmm1,%zmm5,%zmm6
  1a8cf6:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a8cfd:	62 f2 7d 48 1c ca    	vpabsb %zmm2,%zmm1
  1a8d03:	62 f2 7e 48 29 da    	vpmovb2m %zmm2,%k3
  1a8d09:	62 f2 55 48 00 c0    	vpshufb %zmm0,%zmm5,%zmm0
  1a8d0f:	c4 41 18 57 e4       	vxorps %xmm12,%xmm12,%xmm12
  1a8d14:	c4 c1 79 70 d5 a0    	vpshufd $0xa0,%xmm13,%xmm2
  1a8d1a:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a8d21:	62 e1 7d 48 70 ec 88 	vpshufd $0x88,%zmm4,%zmm21
  1a8d28:	62 72 7d 48 1c ca    	vpabsb %zmm2,%zmm9
  1a8d2e:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a8d34:	62 b1 7d 48 70 e9 88 	vpshufd $0x88,%zmm17,%zmm5
  1a8d3b:	62 b1 45 48 f8 dd    	vpsubb %zmm21,%zmm7,%zmm3
  1a8d41:	c4 c1 79 70 d0 a0    	vpshufd $0xa0,%xmm8,%xmm2
  1a8d47:	62 c1 fd 28 6f f8    	vmovdqa64 %ymm8,%ymm23
  1a8d4d:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a8d54:	62 f2 7e 48 29 e2    	vpmovb2m %zmm2,%k4
  1a8d5a:	62 61 45 48 f8 cd    	vpsubb %zmm5,%zmm7,%zmm25
  1a8d60:	62 31 7d 48 70 db 88 	vpshufd $0x88,%zmm19,%zmm11
  1a8d67:	62 51 45 48 f8 d3    	vpsubb %zmm11,%zmm7,%zmm10
  1a8d6d:	62 71 fe 48 7f 94 24 	vmovdqu64 %zmm10,0x210(%rsp)
  1a8d74:	10 02 00 00 
  1a8d78:	62 42 25 4c 66 da    	vpblendmb %zmm10,%zmm11,%zmm27{%k4}
  1a8d7e:	62 62 7d 48 1c e2    	vpabsb %zmm2,%zmm28
  1a8d84:	62 b1 7d 08 70 d0 f5 	vpshufd $0xf5,%xmm16,%xmm2
  1a8d8b:	62 72 55 43 66 f3    	vpblendmb %zmm3,%zmm21,%zmm14{%k3}
  1a8d91:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a8d98:	62 62 7d 48 1c ea    	vpabsb %zmm2,%zmm29
  1a8d9e:	62 02 55 4a 66 d1    	vpblendmb %zmm25,%zmm5,%zmm26{%k2}
  1a8da4:	62 71 fd 48 6f d5    	vmovdqa64 %zmm5,%zmm10
  1a8daa:	62 71 7d 48 70 c0 88 	vpshufd $0x88,%zmm0,%zmm8
  1a8db1:	62 d1 45 48 f8 e8    	vpsubb %zmm8,%zmm7,%zmm5
  1a8db7:	62 e2 3d 4b 66 e5    	vpblendmb %zmm5,%zmm8,%zmm20{%k3}
  1a8dbd:	62 f2 7e 48 29 da    	vpmovb2m %zmm2,%k3
  1a8dc3:	62 e1 7d 48 70 d4 dd 	vpshufd $0xdd,%zmm4,%zmm18
  1a8dca:	62 b1 45 48 f8 d2    	vpsubb %zmm18,%zmm7,%zmm2
  1a8dd0:	62 f1 fe 48 7f 94 24 	vmovdqu64 %zmm2,0x1d0(%rsp)
  1a8dd7:	d0 01 00 00 
  1a8ddb:	62 52 75 48 50 e6    	vpdpbusd %zmm14,%zmm1,%zmm12
  1a8de1:	62 f2 6d 43 66 d2    	vpblendmb %zmm2,%zmm18,%zmm2{%k3}
  1a8de7:	c4 41 09 ef f6       	vpxor  %xmm14,%xmm14,%xmm14
  1a8dec:	62 72 15 40 50 f2    	vpdpbusd %zmm2,%zmm29,%zmm14
  1a8df2:	c4 c1 79 70 d5 f5    	vpshufd $0xf5,%xmm13,%xmm2
  1a8df8:	c5 7e 7f ac 24 50 03 	vmovdqu %ymm13,0x350(%rsp)
  1a8dff:	00 00 
  1a8e01:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a8e08:	c4 41 01 ef ff       	vpxor  %xmm15,%xmm15,%xmm15
  1a8e0d:	62 32 75 48 50 fc    	vpdpbusd %zmm20,%zmm1,%zmm15
  1a8e13:	62 61 7d 48 70 c6 88 	vpshufd $0x88,%zmm6,%zmm24
  1a8e1a:	62 91 45 48 f8 e0    	vpsubb %zmm24,%zmm7,%zmm4
  1a8e20:	62 f2 3d 42 66 cc    	vpblendmb %zmm4,%zmm24,%zmm1{%k2}
  1a8e26:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a8e2c:	62 f2 7d 48 1c d2    	vpabsb %zmm2,%zmm2
  1a8e32:	62 a1 7d 48 70 e1 dd 	vpshufd $0xdd,%zmm17,%zmm20
  1a8e39:	62 12 35 48 50 e2    	vpdpbusd %zmm26,%zmm9,%zmm12
  1a8e3f:	62 72 35 48 50 f9    	vpdpbusd %zmm1,%zmm9,%zmm15
  1a8e45:	62 21 7d 48 70 d6 88 	vpshufd $0x88,%zmm22,%zmm26
  1a8e4c:	62 91 45 48 f8 ca    	vpsubb %zmm26,%zmm7,%zmm1
  1a8e52:	62 12 1d 40 50 e3    	vpdpbusd %zmm27,%zmm28,%zmm12
  1a8e58:	62 72 2d 44 66 c9    	vpblendmb %zmm1,%zmm26,%zmm9{%k4}
  1a8e5e:	62 61 7d 48 70 d8 dd 	vpshufd $0xdd,%zmm0,%zmm27
  1a8e65:	62 52 1d 40 50 f9    	vpdpbusd %zmm9,%zmm28,%zmm15
  1a8e6b:	62 91 45 48 f8 c3    	vpsubb %zmm27,%zmm7,%zmm0
  1a8e71:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x390(%rsp)
  1a8e78:	90 03 00 00 
  1a8e7c:	62 a1 75 00 ef c9    	vpxord %xmm17,%xmm17,%xmm17
  1a8e82:	62 72 25 43 66 c8    	vpblendmb %zmm0,%zmm27,%zmm9{%k3}
  1a8e88:	62 c2 15 40 50 c9    	vpdpbusd %zmm9,%zmm29,%zmm17
  1a8e8e:	62 b1 45 48 f8 c4    	vpsubb %zmm20,%zmm7,%zmm0
  1a8e94:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x410(%rsp)
  1a8e9b:	10 04 00 00 
  1a8e9f:	62 72 5d 42 66 c8    	vpblendmb %zmm0,%zmm20,%zmm9{%k2}
  1a8ea5:	62 61 7d 48 70 e6 dd 	vpshufd $0xdd,%zmm6,%zmm28
  1a8eac:	62 91 45 48 f8 c4    	vpsubb %zmm28,%zmm7,%zmm0
  1a8eb2:	62 f1 fe 48 7f 84 24 	vmovdqu64 %zmm0,0x3d0(%rsp)
  1a8eb9:	d0 03 00 00 
  1a8ebd:	62 f2 1d 42 66 f0    	vpblendmb %zmm0,%zmm28,%zmm6{%k2}
  1a8ec3:	62 52 6d 48 50 f1    	vpdpbusd %zmm9,%zmm2,%zmm14
  1a8ec9:	62 e2 6d 48 50 ce    	vpdpbusd %zmm6,%zmm2,%zmm17
  1a8ecf:	62 b1 fd 28 6f f7    	vmovdqa64 %ymm23,%ymm6
  1a8ed5:	c5 f9 70 d6 f5       	vpshufd $0xf5,%xmm6,%xmm2
  1a8eda:	62 f3 ed 48 43 d2 00 	vshufi64x2 $0x0,%zmm2,%zmm2,%zmm2
  1a8ee1:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a8ee7:	62 f2 7d 48 1c d2    	vpabsb %zmm2,%zmm2
  1a8eed:	62 a1 7d 48 70 db dd 	vpshufd $0xdd,%zmm19,%zmm19
  1a8ef4:	62 21 45 48 f8 eb    	vpsubb %zmm19,%zmm7,%zmm29
  1a8efa:	62 12 65 42 66 cd    	vpblendmb %zmm29,%zmm19,%zmm9{%k2}
  1a8f00:	62 52 6d 48 50 f1    	vpdpbusd %zmm9,%zmm2,%zmm14
  1a8f06:	62 a1 7d 48 70 f6 dd 	vpshufd $0xdd,%zmm22,%zmm22
  1a8f0d:	62 21 45 48 f8 fe    	vpsubb %zmm22,%zmm7,%zmm31
  1a8f13:	62 82 4d 42 66 ff    	vpblendmb %zmm31,%zmm22,%zmm23{%k2}
  1a8f19:	62 a2 6d 48 50 cf    	vpdpbusd %zmm23,%zmm2,%zmm17
  1a8f1f:	62 b1 7d 28 70 d0 a0 	vpshufd $0xa0,%ymm16,%ymm2
  1a8f26:	62 f3 ed 48 43 d2 55 	vshufi64x2 $0x55,%zmm2,%zmm2,%zmm2
  1a8f2d:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a8f33:	62 e1 7f 4a 6f eb    	vmovdqu8 %zmm3,%zmm21{%k2}
  1a8f39:	62 71 7f 4a 6f c5    	vmovdqu8 %zmm5,%zmm8{%k2}
  1a8f3f:	62 f2 7d 48 1c d2    	vpabsb %zmm2,%zmm2
  1a8f45:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
  1a8f49:	62 b2 6d 48 50 ed    	vpdpbusd %zmm21,%zmm2,%zmm5
  1a8f4f:	62 a1 55 00 ef ed    	vpxord %xmm21,%xmm21,%xmm21
  1a8f55:	62 c2 6d 48 50 e8    	vpdpbusd %zmm8,%zmm2,%zmm21
  1a8f5b:	c4 c1 7d 70 d5 a0    	vpshufd $0xa0,%ymm13,%ymm2
  1a8f61:	c4 41 31 ef c9       	vpxor  %xmm9,%xmm9,%xmm9
  1a8f66:	62 f3 ed 48 43 fa 55 	vshufi64x2 $0x55,%zmm2,%zmm2,%zmm7
  1a8f6d:	62 f2 7e 48 29 d7    	vpmovb2m %zmm7,%k2
  1a8f73:	62 11 7f 4a 6f d1    	vmovdqu8 %zmm25,%zmm10{%k2}
  1a8f79:	c5 fd 70 d6 a0       	vpshufd $0xa0,%ymm6,%ymm2
  1a8f7e:	62 73 ed 48 43 c2 55 	vshufi64x2 $0x55,%zmm2,%zmm2,%zmm8
  1a8f85:	62 d2 7e 48 29 e0    	vpmovb2m %zmm8,%k4
  1a8f8b:	62 71 7f 4c 6f 9c 24 	vmovdqu8 0x210(%rsp),%zmm11{%k4}
  1a8f92:	10 02 00 00 
  1a8f96:	62 71 fe 48 7f 9c 24 	vmovdqu64 %zmm11,0x210(%rsp)
  1a8f9d:	10 02 00 00 
  1a8fa1:	c5 fe 6f 5c 31 08    	vmovdqu 0x8(%rcx,%rsi,1),%ymm3
  1a8fa7:	62 f1 8d 40 db 94 24 	vpandq 0x250(%rsp),%zmm30,%zmm2
  1a8fae:	50 02 00 00 
  1a8fb2:	62 61 7f 4a 6f c4    	vmovdqu8 %zmm4,%zmm24{%k2}
  1a8fb8:	c5 f9 70 e3 a0       	vpshufd $0xa0,%xmm3,%xmm4
  1a8fbd:	62 73 dd 48 43 dc 00 	vshufi64x2 $0x0,%zmm4,%zmm4,%zmm11
  1a8fc4:	62 f1 fd 48 6f 05 b2 	vmovdqa64 -0x17004e(%rip),%zmm0        # 38f80 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x3d8>
  1a8fcb:	ff e8 ff 
  1a8fce:	62 f2 7d 48 00 e2    	vpshufb %zmm2,%zmm0,%zmm4
  1a8fd4:	62 d2 7e 48 29 db    	vpmovb2m %zmm11,%k3
  1a8fda:	c5 fd 70 d3 a0       	vpshufd $0xa0,%ymm3,%ymm2
  1a8fdf:	62 f3 ed 48 43 d2 55 	vshufi64x2 $0x55,%zmm2,%zmm2,%zmm2
  1a8fe6:	62 f2 7e 48 29 d2    	vpmovb2m %zmm2,%k2
  1a8fec:	62 61 7d 48 70 cc 88 	vpshufd $0x88,%zmm4,%zmm25
  1a8ff3:	62 91 35 48 f8 c1    	vpsubb %zmm25,%zmm9,%zmm0
  1a8ff9:	62 72 35 43 66 e8    	vpblendmb %zmm0,%zmm25,%zmm13{%k3}
  1a8fff:	62 71 fe 48 7f ac 24 	vmovdqu64 %zmm13,0x250(%rsp)
  1a9006:	50 02 00 00 
  1a900a:	62 61 7f 4a 6f c8    	vmovdqu8 %zmm0,%zmm25{%k2}
  1a9010:	62 61 7f 4c 6f d1    	vmovdqu8 %zmm1,%zmm26{%k4}
  1a9016:	62 b1 7d 28 70 c0 f5 	vpshufd $0xf5,%ymm16,%ymm0
  1a901d:	62 73 fd 48 43 e8 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm13
  1a9024:	62 d2 7e 48 29 e5    	vpmovb2m %zmm13,%k4
  1a902a:	62 e1 7f 4c 6f 94 24 	vmovdqu8 0x1d0(%rsp),%zmm18{%k4}
  1a9031:	d0 01 00 00 
  1a9035:	62 f1 8d 40 db 84 24 	vpandq 0x450(%rsp),%zmm30,%zmm0
  1a903c:	50 04 00 00 
  1a9040:	62 f1 fd 48 6f 0d 36 	vmovdqa64 -0x1700ca(%rip),%zmm1        # 38f80 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x3d8>
  1a9047:	ff e8 ff 
  1a904a:	62 e2 75 48 00 c0    	vpshufb %zmm0,%zmm1,%zmm16
  1a9050:	62 b1 7d 48 70 c8 88 	vpshufd $0x88,%zmm16,%zmm1
  1a9057:	62 f1 35 48 f8 c1    	vpsubb %zmm1,%zmm9,%zmm0
  1a905d:	62 81 fd 48 6f fe    	vmovdqa64 %zmm30,%zmm23
  1a9063:	62 62 75 4b 66 f0    	vpblendmb %zmm0,%zmm1,%zmm30{%k3}
  1a9069:	62 f1 7f 4a 6f c8    	vmovdqu8 %zmm0,%zmm1{%k2}
  1a906f:	c5 fd 70 c6 f5       	vpshufd $0xf5,%ymm6,%ymm0
  1a9074:	62 f3 fd 48 43 c0 55 	vshufi64x2 $0x55,%zmm0,%zmm0,%zmm0
  1a907b:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  1a9081:	62 81 7f 4a 6f dd    	vmovdqu8 %zmm29,%zmm19{%k2}
  1a9087:	62 61 7f 4c 6f 9c 24 	vmovdqu8 0x390(%rsp),%zmm27{%k4}
  1a908e:	90 03 00 00 
  1a9092:	62 81 7f 4a 6f f7    	vmovdqu8 %zmm31,%zmm22{%k2}
  1a9098:	c5 f9 70 f3 f5       	vpshufd $0xf5,%xmm3,%xmm6
  1a909d:	62 63 cd 48 43 ee 00 	vshufi64x2 $0x0,%zmm6,%zmm6,%zmm29
  1a90a4:	62 92 7e 48 29 d5    	vpmovb2m %zmm29,%k2
  1a90aa:	c5 fd 70 db f5       	vpshufd $0xf5,%ymm3,%ymm3
  1a90af:	62 f1 7d 48 70 e4 dd 	vpshufd $0xdd,%zmm4,%zmm4
  1a90b6:	62 f3 e5 48 43 db 55 	vshufi64x2 $0x55,%zmm3,%zmm3,%zmm3
  1a90bd:	62 f1 35 48 f8 f4    	vpsubb %zmm4,%zmm9,%zmm6
  1a90c3:	62 62 5d 4a 66 fe    	vpblendmb %zmm6,%zmm4,%zmm31{%k2}
  1a90c9:	62 f2 7e 48 29 db    	vpmovb2m %zmm3,%k3
  1a90cf:	62 f1 7f 4b 6f e6    	vmovdqu8 %zmm6,%zmm4{%k3}
  1a90d5:	62 b1 7d 48 70 f0 dd 	vpshufd $0xdd,%zmm16,%zmm6
  1a90dc:	62 e1 35 48 f8 c6    	vpsubb %zmm6,%zmm9,%zmm16
  1a90e2:	62 32 4d 4a 66 c8    	vpblendmb %zmm16,%zmm6,%zmm9{%k2}
  1a90e8:	62 b1 7f 4b 6f f0    	vmovdqu8 %zmm16,%zmm6{%k3}
  1a90ee:	62 e1 7d 28 70 84 24 	vpshufd $0xf5,0x350(%rsp),%ymm16
  1a90f5:	50 03 00 00 f5 
  1a90fa:	62 a3 fd 40 43 c0 55 	vshufi64x2 $0x55,%zmm16,%zmm16,%zmm16
  1a9101:	62 b2 7e 48 29 d0    	vpmovb2m %zmm16,%k2
  1a9107:	62 e1 7f 4a 6f a4 24 	vmovdqu8 0x410(%rsp),%zmm20{%k2}
  1a910e:	10 04 00 00 
  1a9112:	62 61 7f 4a 6f a4 24 	vmovdqu8 0x3d0(%rsp),%zmm28{%k2}
  1a9119:	d0 03 00 00 
  1a911d:	62 f2 7d 48 1c ff    	vpabsb %zmm7,%zmm7
  1a9123:	62 d2 45 48 50 ea    	vpdpbusd %zmm10,%zmm7,%zmm5
  1a9129:	62 82 45 48 50 e8    	vpdpbusd %zmm24,%zmm7,%zmm21
  1a912f:	62 d2 7d 48 1c f8    	vpabsb %zmm8,%zmm7
  1a9135:	62 f2 45 48 50 ac 24 	vpdpbusd 0x210(%rsp),%zmm7,%zmm5
  1a913c:	10 02 00 00 
  1a9140:	62 82 45 48 50 ea    	vpdpbusd %zmm26,%zmm7,%zmm21
  1a9146:	62 f2 7d 48 1c d2    	vpabsb %zmm2,%zmm2
  1a914c:	62 92 6d 48 50 e9    	vpdpbusd %zmm25,%zmm2,%zmm5
  1a9152:	62 e2 6d 48 50 e9    	vpdpbusd %zmm1,%zmm2,%zmm21
  1a9158:	62 d2 7d 48 1c cd    	vpabsb %zmm13,%zmm1
  1a915e:	c5 e9 ef d2          	vpxor  %xmm2,%xmm2,%xmm2
  1a9162:	62 b2 75 48 50 d2    	vpdpbusd %zmm18,%zmm1,%zmm2
  1a9168:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  1a916c:	62 92 75 48 50 fb    	vpdpbusd %zmm27,%zmm1,%zmm7
  1a9172:	62 b2 7d 48 1c c8    	vpabsb %zmm16,%zmm1
  1a9178:	62 b2 75 48 50 d4    	vpdpbusd %zmm20,%zmm1,%zmm2
  1a917e:	62 92 75 48 50 fc    	vpdpbusd %zmm28,%zmm1,%zmm7
  1a9184:	62 d2 7d 48 1c cb    	vpabsb %zmm11,%zmm1
  1a918a:	62 72 75 48 50 a4 24 	vpdpbusd 0x250(%rsp),%zmm1,%zmm12
  1a9191:	50 02 00 00 
  1a9195:	62 12 75 48 50 fe    	vpdpbusd %zmm30,%zmm1,%zmm15
  1a919b:	62 21 fd 48 6f f7    	vmovdqa64 %zmm23,%zmm30
  1a91a1:	62 f2 7d 48 1c c0    	vpabsb %zmm0,%zmm0
  1a91a7:	62 b2 7d 48 50 d3    	vpdpbusd %zmm19,%zmm0,%zmm2
  1a91ad:	62 b2 7d 48 50 fe    	vpdpbusd %zmm22,%zmm0,%zmm7
  1a91b3:	41 0f b6 54 37 04    	movzbl 0x4(%r15,%rsi,1),%edx
  1a91b9:	c4 c1 7a 10 04 90    	vmovss (%r8,%rdx,4),%xmm0
  1a91bf:	41 0f b6 54 37 05    	movzbl 0x5(%r15,%rsi,1),%edx
  1a91c5:	c4 c3 79 21 04 90 10 	vinsertps $0x10,(%r8,%rdx,4),%xmm0,%xmm0
  1a91cc:	41 0f b6 54 37 06    	movzbl 0x6(%r15,%rsi,1),%edx
  1a91d2:	c4 c3 79 21 04 90 20 	vinsertps $0x20,(%r8,%rdx,4),%xmm0,%xmm0
  1a91d9:	41 0f b6 14 37       	movzbl (%r15,%rsi,1),%edx
  1a91de:	62 92 7d 48 1c cd    	vpabsb %zmm29,%zmm1
  1a91e4:	62 12 75 48 50 f7    	vpdpbusd %zmm31,%zmm1,%zmm14
  1a91ea:	c4 41 7a 10 04 90    	vmovss (%r8,%rdx,4),%xmm8
  1a91f0:	41 0f b6 54 37 01    	movzbl 0x1(%r15,%rsi,1),%edx
  1a91f6:	c4 43 39 21 04 90 10 	vinsertps $0x10,(%r8,%rdx,4),%xmm8,%xmm8
  1a91fd:	41 0f b6 54 36 04    	movzbl 0x4(%r14,%rsi,1),%edx
  1a9203:	62 c2 75 48 50 c9    	vpdpbusd %zmm9,%zmm1,%zmm17
  1a9209:	c4 c1 7a 10 0c 90    	vmovss (%r8,%rdx,4),%xmm1
  1a920f:	41 0f b6 54 37 02    	movzbl 0x2(%r15,%rsi,1),%edx
  1a9215:	c4 43 39 21 04 90 20 	vinsertps $0x20,(%r8,%rdx,4),%xmm8,%xmm8
  1a921c:	41 0f b6 54 36 05    	movzbl 0x5(%r14,%rsi,1),%edx
  1a9222:	c4 c3 71 21 0c 90 10 	vinsertps $0x10,(%r8,%rdx,4),%xmm1,%xmm1
  1a9229:	41 0f b6 14 36       	movzbl (%r14,%rsi,1),%edx
  1a922e:	c4 41 7a 10 0c 90    	vmovss (%r8,%rdx,4),%xmm9
  1a9234:	41 0f b6 54 36 06    	movzbl 0x6(%r14,%rsi,1),%edx
  1a923a:	c4 c3 71 21 0c 90 20 	vinsertps $0x20,(%r8,%rdx,4),%xmm1,%xmm1
  1a9241:	41 0f b6 54 36 01    	movzbl 0x1(%r14,%rsi,1),%edx
  1a9247:	c4 43 31 21 0c 90 10 	vinsertps $0x10,(%r8,%rdx,4),%xmm9,%xmm9
  1a924e:	62 f2 7d 48 1c db    	vpabsb %zmm3,%zmm3
  1a9254:	62 f2 65 48 50 d4    	vpdpbusd %zmm4,%zmm3,%zmm2
  1a925a:	41 0f b6 54 37 07    	movzbl 0x7(%r15,%rsi,1),%edx
  1a9260:	c4 c3 79 21 04 90 30 	vinsertps $0x30,(%r8,%rdx,4),%xmm0,%xmm0
  1a9267:	62 f2 65 48 50 fe    	vpdpbusd %zmm6,%zmm3,%zmm7
  1a926d:	41 0f b6 54 37 03    	movzbl 0x3(%r15,%rsi,1),%edx
  1a9273:	c4 c3 39 21 1c 90 30 	vinsertps $0x30,(%r8,%rdx,4),%xmm8,%xmm3
  1a927a:	c4 e3 65 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm3,%ymm0
  1a9280:	41 0f b6 54 36 07    	movzbl 0x7(%r14,%rsi,1),%edx
  1a9286:	c4 c3 71 21 0c 90 30 	vinsertps $0x30,(%r8,%rdx,4),%xmm1,%xmm1
  1a928d:	41 0f b6 54 36 02    	movzbl 0x2(%r14,%rsi,1),%edx
  1a9293:	c4 c3 31 21 1c 90 20 	vinsertps $0x20,(%r8,%rdx,4),%xmm9,%xmm3
  1a929a:	62 71 7c 48 10 8c 24 	vmovups 0x2d0(%rsp),%zmm9
  1a92a1:	d0 02 00 00 
  1a92a5:	41 0f b6 54 36 03    	movzbl 0x3(%r14,%rsi,1),%edx
  1a92ab:	c4 c3 61 21 1c 90 30 	vinsertps $0x30,(%r8,%rdx,4),%xmm3,%xmm3
  1a92b2:	c4 e3 65 18 c9 01    	vinsertf128 $0x1,%xmm1,%ymm3,%ymm1
  1a92b8:	62 f3 f5 48 1a c0 01 	vinsertf64x4 $0x1,%ymm0,%zmm1,%zmm0
  1a92bf:	62 d1 0d 48 fe cc    	vpaddd %zmm12,%zmm14,%zmm1
  1a92c5:	62 f1 6d 48 fe d5    	vpaddd %zmm5,%zmm2,%zmm2
  1a92cb:	62 b1 45 48 fe dd    	vpaddd %zmm21,%zmm7,%zmm3
  1a92d1:	62 d1 75 40 fe e7    	vpaddd %zmm15,%zmm17,%zmm4
  1a92d7:	62 f1 7e 89 6f 2c 31 	vmovdqu32 (%rcx,%rsi,1),%xmm5{%k1}{z}
  1a92de:	c4 e2 79 13 ed       	vcvtph2ps %xmm5,%xmm5
  1a92e3:	62 f1 f5 48 6c f4    	vpunpcklqdq %zmm4,%zmm1,%zmm6
  1a92e9:	62 f1 7c 48 5b f6    	vcvtdq2ps %zmm6,%zmm6
  1a92ef:	62 f2 7d 48 18 fd    	vbroadcastss %xmm5,%zmm7
  1a92f5:	62 f1 7c 48 59 ff    	vmulps %zmm7,%zmm0,%zmm7
  1a92fb:	62 72 4d 48 b8 cf    	vfmadd231ps %zmm7,%zmm6,%zmm9
  1a9301:	62 f1 ed 48 6c f3    	vpunpcklqdq %zmm3,%zmm2,%zmm6
  1a9307:	c5 d1 c6 fd 01       	vshufpd $0x1,%xmm5,%xmm5,%xmm7
  1a930c:	62 f2 7d 48 18 ff    	vbroadcastss %xmm7,%zmm7
  1a9312:	62 f1 7c 48 5b f6    	vcvtdq2ps %zmm6,%zmm6
  1a9318:	62 f1 7c 48 59 ff    	vmulps %zmm7,%zmm0,%zmm7
  1a931e:	62 71 7c 48 10 84 24 	vmovups 0x150(%rsp),%zmm8
  1a9325:	50 01 00 00 
  1a9329:	62 72 4d 48 b8 c7    	vfmadd231ps %zmm7,%zmm6,%zmm8
  1a932f:	62 71 7c 48 11 84 24 	vmovups %zmm8,0x150(%rsp)
  1a9336:	50 01 00 00 
  1a933a:	62 f1 fe 48 6f b4 24 	vmovdqu64 0x150(%rsp),%zmm6
  1a9341:	50 01 00 00 
  1a9345:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
  1a9349:	62 f1 f5 48 6d cc    	vpunpckhqdq %zmm4,%zmm1,%zmm1
  1a934f:	62 f1 7c 48 5b c9    	vcvtdq2ps %zmm1,%zmm1
  1a9355:	c5 fa 16 e5          	vmovshdup %xmm5,%xmm4
  1a9359:	62 f2 7d 48 18 e4    	vbroadcastss %xmm4,%zmm4
  1a935f:	62 f1 7c 48 59 e4    	vmulps %zmm4,%zmm0,%zmm4
  1a9365:	62 71 7c 48 10 84 24 	vmovups 0x110(%rsp),%zmm8
  1a936c:	10 01 00 00 
  1a9370:	62 72 75 48 b8 c4    	vfmadd231ps %zmm4,%zmm1,%zmm8
  1a9376:	62 71 7c 48 11 84 24 	vmovups %zmm8,0x110(%rsp)
  1a937d:	10 01 00 00 
  1a9381:	62 f1 7c 48 10 a4 24 	vmovups 0x110(%rsp),%zmm4
  1a9388:	10 01 00 00 
  1a938c:	c5 d0 c6 cd ff       	vshufps $0xff,%xmm5,%xmm5,%xmm1
  1a9391:	62 f1 fd 48 6f 2d e5 	vmovdqa64 -0x17041b(%rip),%zmm5        # 38f80 <_RNvNvNtCskQw6xMrwh4w_15crossbeam_epoch5guard11unprotected11UNPROTECTED+0x3d8>
  1a9398:	fb e8 ff 
  1a939b:	62 f2 7d 48 18 c9    	vbroadcastss %xmm1,%zmm1
  1a93a1:	62 f1 7c 48 59 c1    	vmulps %zmm1,%zmm0,%zmm0
  1a93a7:	62 f1 ed 48 6d cb    	vpunpckhqdq %zmm3,%zmm2,%zmm1
  1a93ad:	62 f1 7c 48 5b c9    	vcvtdq2ps %zmm1,%zmm1
  1a93b3:	62 f1 7c 48 10 94 24 	vmovups 0x190(%rsp),%zmm2
  1a93ba:	90 01 00 00 
  1a93be:	62 f2 75 48 b8 d0    	vfmadd231ps %zmm0,%zmm1,%zmm2
  1a93c4:	62 f1 7c 48 11 94 24 	vmovups %zmm2,0x190(%rsp)
  1a93cb:	90 01 00 00 
  1a93cf:	62 f1 7c 48 10 84 24 	vmovups 0x190(%rsp),%zmm0
  1a93d6:	90 01 00 00 
  1a93da:	48 81 c6 88 00 00 00 	add    $0x88,%rsi
  1a93e1:	48 ff cd             	dec    %rbp
  1a93e4:	0f 85 06 f8 ff ff    	jne    1a8bf0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2360>
  1a93ea:	4c 89 e2             	mov    %r12,%rdx
  1a93ed:	48 c1 e2 05          	shl    $0x5,%rdx
  1a93f1:	48 03 14 24          	add    (%rsp),%rdx
  1a93f5:	62 31 7c 48 11 0c 8a 	vmovups %zmm9,(%rdx,%r9,4)
  1a93fc:	62 b1 7c 48 11 24 92 	vmovups %zmm4,(%rdx,%r10,4)
  1a9403:	62 b1 fe 48 7f 34 9a 	vmovdqu64 %zmm6,(%rdx,%r11,4)
  1a940a:	62 f1 7c 48 11 04 9a 	vmovups %zmm0,(%rdx,%rbx,4)
  1a9411:	49 83 c4 02          	add    $0x2,%r12
  1a9415:	49 01 ff             	add    %rdi,%r15
  1a9418:	49 01 fe             	add    %rdi,%r14
  1a941b:	49 ff cd             	dec    %r13
  1a941e:	0f 85 ac f7 ff ff    	jne    1a8bd0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2340>
  1a9424:	48 89 c5             	mov    %rax,%rbp
  1a9427:	48 ff c5             	inc    %rbp
  1a942a:	48 03 4c 24 40       	add    0x40(%rsp),%rcx
  1a942f:	48 3b ac 24 90 02 00 	cmp    0x290(%rsp),%rbp
  1a9436:	00 
  1a9437:	4c 8b 6c 24 48       	mov    0x48(%rsp),%r13
  1a943c:	0f 85 3e f7 ff ff    	jne    1a8b80 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x22f0>
  1a9442:	48 8b 84 24 90 12 00 	mov    0x1290(%rsp),%rax
  1a9449:	00 
  1a944a:	48 39 44 24 58       	cmp    %rax,0x58(%rsp)
  1a944f:	4c 8b 4c 24 38       	mov    0x38(%rsp),%r9
  1a9454:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
  1a9459:	4c 8b 54 24 60       	mov    0x60(%rsp),%r10
  1a945e:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  1a9463:	75 09                	jne    1a946e <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bde>
  1a9465:	48 81 c4 58 12 00 00 	add    $0x1258,%rsp
  1a946c:	eb 1e                	jmp    1a948c <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bfc>
  1a946e:	48 8b 34 24          	mov    (%rsp),%rsi
  1a9472:	4c 89 ea             	mov    %r13,%rdx
  1a9475:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  1a947a:	41 52                	push   %r10
  1a947c:	50                   	push   %rax
  1a947d:	c5 f8 77             	vzeroupper
  1a9480:	e8 4b 01 00 00       	call   1a95d0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8614gemm_evex_vnni>
  1a9485:	48 81 c4 68 12 00 00 	add    $0x1268,%rsp
  1a948c:	5b                   	pop    %rbx
  1a948d:	41 5c                	pop    %r12
  1a948f:	41 5d                	pop    %r13
  1a9491:	41 5e                	pop    %r14
  1a9493:	41 5f                	pop    %r15
  1a9495:	5d                   	pop    %rbp
  1a9496:	c5 f8 77             	vzeroupper
  1a9499:	c3                   	ret
  1a949a:	49 0f af cd          	imul   %r13,%rcx
  1a949e:	48 c1 e1 04          	shl    $0x4,%rcx
  1a94a2:	4c 89 ee             	mov    %r13,%rsi
  1a94a5:	48 c1 e6 04          	shl    $0x4,%rsi
  1a94a9:	48 8d 3c ad 01 00 00 	lea    0x1(,%rbp,4),%rdi
  1a94b0:	00 
  1a94b1:	49 0f af fd          	imul   %r13,%rdi
  1a94b5:	4c 8d 04 ad 02 00 00 	lea    0x2(,%rbp,4),%r8
  1a94bc:	00 
  1a94bd:	4d 0f af c5          	imul   %r13,%r8
  1a94c1:	4c 8d 0c ad 03 00 00 	lea    0x3(,%rbp,4),%r9
  1a94c8:	00 
  1a94c9:	4d 0f af cd          	imul   %r13,%r9
  1a94cd:	4c 8b 94 24 90 12 00 	mov    0x1290(%rsp),%r10
  1a94d4:	00 
  1a94d5:	49 c1 ea 03          	shr    $0x3,%r10
  1a94d9:	49 83 e2 fc          	and    $0xfffffffffffffffc,%r10
  1a94dd:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  1a94e1:	4c 8b 1c 24          	mov    (%rsp),%r11
  1a94e5:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
  1a94ea:	eb 6e                	jmp    1a955a <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2cca>
  1a94ec:	0f 1f 40 00          	nopl   0x0(%rax)
  1a94f0:	31 db                	xor    %ebx,%ebx
  1a94f2:	48 8d 14 ad 00 00 00 	lea    0x0(,%rbp,4),%rdx
  1a94f9:	00 
  1a94fa:	49 0f af d5          	imul   %r13,%rdx
  1a94fe:	4c 8d 34 ad 01 00 00 	lea    0x1(,%rbp,4),%r14
  1a9505:	00 
  1a9506:	4d 0f af f5          	imul   %r13,%r14
  1a950a:	4c 8d 3c ad 02 00 00 	lea    0x2(,%rbp,4),%r15
  1a9511:	00 
  1a9512:	4d 0f af fd          	imul   %r13,%r15
  1a9516:	4c 8d 24 ad 03 00 00 	lea    0x3(,%rbp,4),%r12
  1a951d:	00 
  1a951e:	4d 0f af e5          	imul   %r13,%r12
  1a9522:	48 c1 e3 05          	shl    $0x5,%rbx
  1a9526:	48 03 1c 24          	add    (%rsp),%rbx
  1a952a:	62 f1 7c 48 11 04 93 	vmovups %zmm0,(%rbx,%rdx,4)
  1a9531:	62 b1 7c 48 11 04 b3 	vmovups %zmm0,(%rbx,%r14,4)
  1a9538:	62 b1 7c 48 11 04 bb 	vmovups %zmm0,(%rbx,%r15,4)
  1a953f:	62 b1 7c 48 11 04 a3 	vmovups %zmm0,(%rbx,%r12,4)
  1a9546:	48 ff c5             	inc    %rbp
  1a9549:	49 01 f3             	add    %rsi,%r11
  1a954c:	48 3b ac 24 90 02 00 	cmp    0x290(%rsp),%rbp
  1a9553:	00 
  1a9554:	0f 84 e8 fe ff ff    	je     1a9442 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2bb2>
  1a955a:	48 83 f8 01          	cmp    $0x1,%rax
  1a955e:	74 90                	je     1a94f0 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2c60>
  1a9560:	4d 89 de             	mov    %r11,%r14
  1a9563:	31 db                	xor    %ebx,%ebx
  1a9565:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  1a956c:	00 00 00 00 
  1a9570:	62 d1 7c 48 11 04 0e 	vmovups %zmm0,(%r14,%rcx,1)
  1a9577:	62 d1 7c 48 11 04 be 	vmovups %zmm0,(%r14,%rdi,4)
  1a957e:	62 91 7c 48 11 04 86 	vmovups %zmm0,(%r14,%r8,4)
  1a9585:	62 91 7c 48 11 04 8e 	vmovups %zmm0,(%r14,%r9,4)
  1a958c:	62 d1 7c 48 11 44 0e 	vmovups %zmm0,0x40(%r14,%rcx,1)
  1a9593:	01 
  1a9594:	62 d1 7c 48 11 44 be 	vmovups %zmm0,0x40(%r14,%rdi,4)
  1a959b:	01 
  1a959c:	62 91 7c 48 11 44 86 	vmovups %zmm0,0x40(%r14,%r8,4)
  1a95a3:	01 
  1a95a4:	62 91 7c 48 11 44 8e 	vmovups %zmm0,0x40(%r14,%r9,4)
  1a95ab:	01 
  1a95ac:	48 83 c3 04          	add    $0x4,%rbx
  1a95b0:	49 83 ee 80          	sub    $0xffffffffffffff80,%r14
  1a95b4:	49 39 da             	cmp    %rbx,%r10
  1a95b7:	75 b7                	jne    1a9570 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2ce0>
  1a95b9:	a8 01                	test   $0x1,%al
  1a95bb:	0f 85 31 ff ff ff    	jne    1a94f2 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2c62>
  1a95c1:	eb 83                	jmp    1a9546 <_RNvNtNtCs96HZWesffxA_4ggml6repack8simd_x8611gemm_avx512+0x2cb6>
  1a95c3:	cc                   	int3
  1a95c4:	cc                   	int3
  1a95c5:	cc                   	int3
  1a95c6:	cc                   	int3
  1a95c7:	cc                   	int3
  1a95c8:	cc                   	int3
  1a95c9:	cc                   	int3
  1a95ca:	cc                   	int3
  1a95cb:	cc                   	int3
  1a95cc:	cc                   	int3
  1a95cd:	cc                   	int3
  1a95ce:	cc                   	int3
  1a95cf:	cc                   	int3
