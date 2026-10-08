  10ad50:	55                   	push   %rbp
  10ad51:	49 89 d3             	mov    %rdx,%r11
  10ad54:	62 e1 fd 28 6f c0    	vmovdqa64 %ymm0,%ymm16
  10ad5a:	c5 f1 76 c9          	vpcmpeqd %xmm1,%xmm1,%xmm1
  10ad5e:	48 89 e5             	mov    %rsp,%rbp
  10ad61:	41 57                	push   %r15
  10ad63:	41 56                	push   %r14
  10ad65:	41 55                	push   %r13
  10ad67:	41 54                	push   %r12
  10ad69:	53                   	push   %rbx
  10ad6a:	4d 89 c6             	mov    %r8,%r14
  10ad6d:	45 89 c8             	mov    %r9d,%r8d
  10ad70:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
  10ad74:	45 89 c2             	mov    %r8d,%r10d
  10ad77:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  10ad7b:	c4 e3 79 02 e9 03    	vpblendd $0x3,%xmm1,%xmm0,%xmm5
  10ad81:	48 81 ec 00 0a 00 00 	sub    $0xa00,%rsp
  10ad88:	44 8b 4d 10          	mov    0x10(%rbp),%r9d
  10ad8c:	62 b3 7d 40 3a f0 01 	vinserti32x8 $0x1,%ymm16,%zmm16,%zmm6
  10ad93:	89 bc 24 8c 05 00 00 	mov    %edi,0x58c(%rsp)
  10ad9a:	48 89 94 24 78 01 00 	mov    %rdx,0x178(%rsp)
  10ada1:	00 
  10ada2:	48 89 b4 24 b8 00 00 	mov    %rsi,0xb8(%rsp)
  10ada9:	00 
  10adaa:	44 89 c2             	mov    %r8d,%edx
  10adad:	48 89 8c 24 80 05 00 	mov    %rcx,0x580(%rsp)
  10adb4:	00 
  10adb5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  10adbc:	00 00 
  10adbe:	48 89 84 24 f8 09 00 	mov    %rax,0x9f8(%rsp)
  10adc5:	00 
  10adc6:	31 c0                	xor    %eax,%eax
  10adc8:	c5 f9 7f ac 24 90 05 	vmovdqa %xmm5,0x590(%rsp)
  10adcf:	00 00 
  10add1:	85 ff                	test   %edi,%edi
  10add3:	8d 47 1f             	lea    0x1f(%rdi),%eax
  10add6:	0f 49 c7             	cmovns %edi,%eax
  10add9:	44 89 cf             	mov    %r9d,%edi
  10addc:	c1 fa 1f             	sar    $0x1f,%edx
  10addf:	c1 f8 05             	sar    $0x5,%eax
  10ade2:	c1 ea 1c             	shr    $0x1c,%edx
  10ade5:	4c 63 f8             	movslq %eax,%r15
  10ade8:	41 8d 04 10          	lea    (%r8,%rdx,1),%eax
  10adec:	83 e0 0f             	and    $0xf,%eax
  10adef:	29 d0                	sub    %edx,%eax
  10adf1:	44 89 ca             	mov    %r9d,%edx
  10adf4:	c1 fa 1f             	sar    $0x1f,%edx
  10adf7:	c1 ea 1c             	shr    $0x1c,%edx
  10adfa:	41 29 c2             	sub    %eax,%r10d
  10adfd:	41 8d 04 11          	lea    (%r9,%rdx,1),%eax
  10ae01:	83 e0 0f             	and    $0xf,%eax
  10ae04:	29 d0                	sub    %edx,%eax
  10ae06:	29 c7                	sub    %eax,%edi
  10ae08:	45 85 d2             	test   %r10d,%r10d
  10ae0b:	41 8d 42 03          	lea    0x3(%r10),%eax
  10ae0f:	41 0f 49 c2          	cmovns %r10d,%eax
  10ae13:	c1 f8 02             	sar    $0x2,%eax
  10ae16:	48 98                	cltq
  10ae18:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  10ae1f:	00 
  10ae20:	41 83 fa 03          	cmp    $0x3,%r10d
  10ae24:	0f 8e c9 31 00 00    	jle    10dff3 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x32a3>
  10ae2a:	49 69 c7 88 00 00 00 	imul   $0x88,%r15,%rax
  10ae31:	8d 5f 07             	lea    0x7(%rdi),%ebx
  10ae34:	85 ff                	test   %edi,%edi
  10ae36:	4c 89 d9             	mov    %r11,%rcx
  10ae39:	62 f1 7d 48 7f 74 24 	vmovdqa32 %zmm6,0x40(%rsp)
  10ae40:	01 
  10ae41:	62 e1 fd 48 6f fe    	vmovdqa64 %zmm6,%zmm23
  10ae47:	48 89 b4 24 10 01 00 	mov    %rsi,0x110(%rsp)
  10ae4e:	00 
  10ae4f:	44 89 84 24 90 00 00 	mov    %r8d,0x90(%rsp)
  10ae56:	00 
  10ae57:	4c 89 f2             	mov    %r14,%rdx
  10ae5a:	44 89 94 24 a0 00 00 	mov    %r10d,0xa0(%rsp)
  10ae61:	00 
  10ae62:	89 bc 24 e0 00 00 00 	mov    %edi,0xe0(%rsp)
  10ae69:	4c 89 b4 24 98 00 00 	mov    %r14,0x98(%rsp)
  10ae70:	00 
  10ae71:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
  10ae78:	00 
  10ae79:	89 f8                	mov    %edi,%eax
  10ae7b:	62 e1 fd 28 7f 44 24 	vmovdqa64 %ymm16,0x20(%rsp)
  10ae82:	01 
  10ae83:	4c 89 bc 24 00 03 00 	mov    %r15,0x300(%rsp)
  10ae8a:	00 
  10ae8b:	0f 48 c3             	cmovs  %ebx,%eax
  10ae8e:	45 31 e4             	xor    %r12d,%r12d
  10ae91:	89 9c 24 88 00 00 00 	mov    %ebx,0x88(%rsp)
  10ae98:	49 c1 e3 04          	shl    $0x4,%r11
  10ae9c:	4c 89 9c 24 00 01 00 	mov    %r11,0x100(%rsp)
  10aea3:	00 
  10aea4:	49 89 cb             	mov    %rcx,%r11
  10aea7:	48 8d 0c cd 00 00 00 	lea    0x0(,%rcx,8),%rcx
  10aeae:	00 
  10aeaf:	4d 89 e0             	mov    %r12,%r8
  10aeb2:	48 89 8c 24 90 01 00 	mov    %rcx,0x190(%rsp)
  10aeb9:	00 
  10aeba:	48 8d 8c 24 a0 05 00 	lea    0x5a0(%rsp),%rcx
  10aec1:	00 
  10aec2:	48 89 8c 24 c0 02 00 	mov    %rcx,0x2c0(%rsp)
  10aec9:	00 
  10aeca:	b9 0f 0f 0f 0f       	mov    $0xf0f0f0f,%ecx
  10aecf:	62 f2 7d 48 7c f9    	vpbroadcastd %ecx,%zmm7
  10aed5:	49 69 cf 98 01 00 00 	imul   $0x198,%r15,%rcx
  10aedc:	c1 f8 03             	sar    $0x3,%eax
  10aedf:	48 98                	cltq
  10aee1:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  10aee8:	00 
  10aee9:	62 f1 7d 48 7f 7c 24 	vmovdqa32 %zmm7,0x280(%rsp)
  10aef0:	0a 
  10aef1:	49 69 c7 10 01 00 00 	imul   $0x110,%r15,%rax
  10aef8:	49 c1 e3 06          	shl    $0x6,%r11
  10aefc:	c4 61 f9 6e f0       	vmovq  %rax,%xmm14
  10af01:	4c 01 f0             	add    %r14,%rax
  10af04:	4c 89 9c 24 d0 00 00 	mov    %r11,0xd0(%rsp)
  10af0b:	00 
  10af0c:	45 31 db             	xor    %r11d,%r11d
  10af0f:	48 89 c6             	mov    %rax,%rsi
  10af12:	c5 79 d6 b4 24 18 01 	vmovq  %xmm14,0x118(%rsp)
  10af19:	00 00 
  10af1b:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  10af22:	00 
  10af23:	48 89 b4 24 b0 05 00 	mov    %rsi,0x5b0(%rsp)
  10af2a:	00 
  10af2b:	48 89 94 24 a0 05 00 	mov    %rdx,0x5a0(%rsp)
  10af32:	00 
  10af33:	48 01 c6             	add    %rax,%rsi
  10af36:	83 bc 24 e0 00 00 00 	cmpl   $0x7,0xe0(%rsp)
  10af3d:	07 
  10af3e:	48 8d 3c 02          	lea    (%rdx,%rax,1),%rdi
  10af42:	48 89 bc 24 a8 05 00 	mov    %rdi,0x5a8(%rsp)
  10af49:	00 
  10af4a:	48 89 b4 24 b8 05 00 	mov    %rsi,0x5b8(%rsp)
  10af51:	00 
  10af52:	0f 8e c6 0c 00 00    	jle    10bc1e <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0xece>
  10af58:	49 89 c1             	mov    %rax,%r9
  10af5b:	48 8b 84 24 78 01 00 	mov    0x178(%rsp),%rax
  10af62:	00 
  10af63:	48 8b 94 24 10 01 00 	mov    0x110(%rsp),%rdx
  10af6a:	00 
  10af6b:	45 31 f6             	xor    %r14d,%r14d
  10af6e:	4c 89 9c 24 c8 00 00 	mov    %r11,0xc8(%rsp)
  10af75:	00 
  10af76:	4c 8d 94 24 c0 09 00 	lea    0x9c0(%rsp),%r10
  10af7d:	00 
  10af7e:	62 a1 5d 00 ef e4    	vpxord %xmm20,%xmm20,%xmm20
  10af84:	4d 89 cb             	mov    %r9,%r11
  10af87:	62 a1 65 00 ef db    	vpxord %xmm19,%xmm19,%xmm19
  10af8d:	48 89 bc 24 c0 00 00 	mov    %rdi,0xc0(%rsp)
  10af94:	00 
  10af95:	48 89 b4 24 b0 00 00 	mov    %rsi,0xb0(%rsp)
  10af9c:	00 
  10af9d:	48 89 8c 24 a8 00 00 	mov    %rcx,0xa8(%rsp)
  10afa4:	00 
  10afa5:	49 8d 1c 40          	lea    (%r8,%rax,2),%rbx
  10afa9:	48 89 9c 24 88 01 00 	mov    %rbx,0x188(%rsp)
  10afb0:	00 
  10afb1:	48 01 c3             	add    %rax,%rbx
  10afb4:	48 89 9c 24 80 01 00 	mov    %rbx,0x180(%rsp)
  10afbb:	00 
  10afbc:	48 01 c3             	add    %rax,%rbx
  10afbf:	48 89 9c 24 70 01 00 	mov    %rbx,0x170(%rsp)
  10afc6:	00 
  10afc7:	48 01 c3             	add    %rax,%rbx
  10afca:	48 89 9c 24 68 01 00 	mov    %rbx,0x168(%rsp)
  10afd1:	00 
  10afd2:	48 01 c3             	add    %rax,%rbx
  10afd5:	48 89 9c 24 60 01 00 	mov    %rbx,0x160(%rsp)
  10afdc:	00 
  10afdd:	48 01 c3             	add    %rax,%rbx
  10afe0:	48 89 9c 24 58 01 00 	mov    %rbx,0x158(%rsp)
  10afe7:	00 
  10afe8:	48 8b 9c 24 90 01 00 	mov    0x190(%rsp),%rbx
  10afef:	00 
  10aff0:	48 01 c3             	add    %rax,%rbx
  10aff3:	48 89 9c 24 50 01 00 	mov    %rbx,0x150(%rsp)
  10affa:	00 
  10affb:	48 01 c3             	add    %rax,%rbx
  10affe:	48 89 9c 24 48 01 00 	mov    %rbx,0x148(%rsp)
  10b005:	00 
  10b006:	48 01 c3             	add    %rax,%rbx
  10b009:	48 89 9c 24 40 01 00 	mov    %rbx,0x140(%rsp)
  10b010:	00 
  10b011:	48 01 c3             	add    %rax,%rbx
  10b014:	48 89 9c 24 38 01 00 	mov    %rbx,0x138(%rsp)
  10b01b:	00 
  10b01c:	48 01 c3             	add    %rax,%rbx
  10b01f:	48 89 9c 24 30 01 00 	mov    %rbx,0x130(%rsp)
  10b026:	00 
  10b027:	48 01 c3             	add    %rax,%rbx
  10b02a:	48 01 d8             	add    %rbx,%rax
  10b02d:	48 89 9c 24 28 01 00 	mov    %rbx,0x128(%rsp)
  10b034:	00 
  10b035:	48 8d 9c 24 c0 05 00 	lea    0x5c0(%rsp),%rbx
  10b03c:	00 
  10b03d:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
  10b044:	00 
  10b045:	48 8b 84 24 80 05 00 	mov    0x580(%rsp),%rax
  10b04c:	00 
  10b04d:	48 89 9c 24 40 03 00 	mov    %rbx,0x340(%rsp)
  10b054:	00 
  10b055:	4c 89 f3             	mov    %r14,%rbx
  10b058:	48 f7 d8             	neg    %rax
  10b05b:	48 89 84 24 80 03 00 	mov    %rax,0x380(%rsp)
  10b062:	00 
  10b063:	31 c0                	xor    %eax,%eax
  10b065:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  10b06c:	00 00 00 00 
  10b070:	83 bc 24 8c 05 00 00 	cmpl   $0x1f,0x58c(%rsp)
  10b077:	1f 
  10b078:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  10b07c:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x5c0(%rsp)
  10b083:	17 
  10b084:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x600(%rsp)
  10b08b:	18 
  10b08c:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x640(%rsp)
  10b093:	19 
  10b094:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x680(%rsp)
  10b09b:	1a 
  10b09c:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x6c0(%rsp)
  10b0a3:	1b 
  10b0a4:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x700(%rsp)
  10b0ab:	1c 
  10b0ac:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x740(%rsp)
  10b0b3:	1d 
  10b0b4:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x780(%rsp)
  10b0bb:	1e 
  10b0bc:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x7c0(%rsp)
  10b0c3:	1f 
  10b0c4:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x800(%rsp)
  10b0cb:	20 
  10b0cc:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x840(%rsp)
  10b0d3:	21 
  10b0d4:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x880(%rsp)
  10b0db:	22 
  10b0dc:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x8c0(%rsp)
  10b0e3:	23 
  10b0e4:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x900(%rsp)
  10b0eb:	24 
  10b0ec:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x940(%rsp)
  10b0f3:	25 
  10b0f4:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x980(%rsp)
  10b0fb:	26 
  10b0fc:	0f 8e 4b 2e 00 00    	jle    10df4d <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x31fd>
  10b102:	48 8b 8c 24 80 05 00 	mov    0x580(%rsp),%rcx
  10b109:	00 
  10b10a:	48 8b 35 0f 0e 04 00 	mov    0x40e0f(%rip),%rsi        # 14bf20 <ggml_table_f32_e8m0_half@@Base-0x1420>
  10b111:	62 31 fd 48 6f c7    	vmovdqa64 %zmm23,%zmm8
  10b117:	45 31 e4             	xor    %r12d,%r12d
  10b11a:	48 89 9c 24 40 02 00 	mov    %rbx,0x240(%rsp)
  10b121:	00 
  10b122:	48 89 94 24 00 02 00 	mov    %rdx,0x200(%rsp)
  10b129:	00 
  10b12a:	48 89 84 24 e0 01 00 	mov    %rax,0x1e0(%rsp)
  10b131:	00 
  10b132:	4c 89 84 24 c0 01 00 	mov    %r8,0x1c0(%rsp)
  10b139:	00 
  10b13a:	4c 89 9c 24 a0 01 00 	mov    %r11,0x1a0(%rsp)
  10b141:	00 
  10b142:	48 8d 3c 19          	lea    (%rcx,%rbx,1),%rdi
  10b146:	4e 8d 0c 19          	lea    (%rcx,%r11,1),%r9
  10b14a:	b9 cc cc ff ff       	mov    $0xffffcccc,%ecx
  10b14f:	c5 f8 92 c9          	kmovw  %ecx,%k1
  10b153:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  10b15a:	00 00 00 00 
  10b15e:	66 90                	xchg   %ax,%ax
  10b160:	c5 fd 6f 15 f8 3e 02 	vmovdqa 0x23ef8(%rip),%ymm2        # 12f060 <_ZL11iq2xxs_grid+0xb20>
  10b167:	00 
  10b168:	c5 fe 6f 77 08       	vmovdqu 0x8(%rdi),%ymm6
  10b16d:	c4 62 6d 36 57 28    	vpermd 0x28(%rdi),%ymm2,%ymm10
  10b173:	c4 62 6d 36 5f 68    	vpermd 0x68(%rdi),%ymm2,%ymm11
  10b179:	c4 c2 6d 36 69 28    	vpermd 0x28(%r9),%ymm2,%ymm5
  10b17f:	48 8b 84 24 80 03 00 	mov    0x380(%rsp),%rax
  10b186:	00 
  10b187:	62 f1 7d 48 6f 7c 24 	vmovdqa32 0x280(%rsp),%zmm7
  10b18e:	0a 
  10b18f:	c4 c2 6d 36 61 68    	vpermd 0x68(%r9),%ymm2,%ymm4
  10b195:	0f b6 57 01          	movzbl 0x1(%rdi),%edx
  10b199:	44 0f b6 6f 03       	movzbl 0x3(%rdi),%r13d
  10b19e:	0f b6 1f             	movzbl (%rdi),%ebx
  10b1a1:	44 0f b6 47 02       	movzbl 0x2(%rdi),%r8d
  10b1a6:	44 0f b6 7f 04       	movzbl 0x4(%rdi),%r15d
  10b1ab:	44 0f b6 5f 05       	movzbl 0x5(%rdi),%r11d
  10b1b0:	0f b6 4f 06          	movzbl 0x6(%rdi),%ecx
  10b1b4:	c4 e2 6d 36 c6       	vpermd %ymm6,%ymm2,%ymm0
  10b1b9:	c4 43 4d 02 d2 f0    	vpblendd $0xf0,%ymm10,%ymm6,%ymm10
  10b1bf:	c5 fe 6f 77 48       	vmovdqu 0x48(%rdi),%ymm6
  10b1c4:	4c 8d 34 38          	lea    (%rax,%rdi,1),%r14
  10b1c8:	0f b6 47 07          	movzbl 0x7(%rdi),%eax
  10b1cc:	c4 e3 7d 02 47 28 f0 	vpblendd $0xf0,0x28(%rdi),%ymm0,%ymm0
  10b1d3:	c4 e2 6d 36 ce       	vpermd %ymm6,%ymm2,%ymm1
  10b1d8:	c4 43 4d 02 db f0    	vpblendd $0xf0,%ymm11,%ymm6,%ymm11
  10b1de:	c4 c1 7e 6f 71 08    	vmovdqu 0x8(%r9),%ymm6
  10b1e4:	c4 61 f9 6e e0       	vmovq  %rax,%xmm12
  10b1e9:	41 0f b6 01          	movzbl (%r9),%eax
  10b1ed:	c4 e3 75 02 4f 68 f0 	vpblendd $0xf0,0x68(%rdi),%ymm1,%ymm1
  10b1f4:	c4 e3 4d 02 ed f0    	vpblendd $0xf0,%ymm5,%ymm6,%ymm5
  10b1fa:	c4 e2 6d 36 de       	vpermd %ymm6,%ymm2,%ymm3
  10b1ff:	c4 c1 7e 6f 71 48    	vmovdqu 0x48(%r9),%ymm6
  10b205:	c4 c3 65 02 59 28 f0 	vpblendd $0xf0,0x28(%r9),%ymm3,%ymm3
  10b20c:	62 73 ad 48 3a d5 01 	vinserti64x4 $0x1,%ymm5,%zmm10,%zmm10
  10b213:	62 f3 fd 48 3a c3 01 	vinserti64x4 $0x1,%ymm3,%zmm0,%zmm0
  10b21a:	c4 e2 6d 36 d6       	vpermd %ymm6,%ymm2,%ymm2
  10b21f:	c4 e3 4d 02 e4 f0    	vpblendd $0xf0,%ymm4,%ymm6,%ymm4
  10b225:	c4 c3 6d 02 51 68 f0 	vpblendd $0xf0,0x68(%r9),%ymm2,%ymm2
  10b22c:	62 73 a5 48 3a dc 01 	vinserti64x4 $0x1,%ymm4,%zmm11,%zmm11
  10b233:	62 f1 2d 48 db e7    	vpandd %zmm7,%zmm10,%zmm4
  10b239:	62 d1 2d 48 71 d2 04 	vpsrlw $0x4,%zmm10,%zmm10
  10b240:	62 f3 f5 48 3a ca 01 	vinserti64x4 $0x1,%ymm2,%zmm1,%zmm1
  10b247:	62 71 25 48 db ef    	vpandd %zmm7,%zmm11,%zmm13
  10b24d:	62 d1 25 48 71 d3 04 	vpsrlw $0x4,%zmm11,%zmm11
  10b254:	62 f1 7d 48 db d7    	vpandd %zmm7,%zmm0,%zmm2
  10b25a:	62 f1 75 48 db df    	vpandd %zmm7,%zmm1,%zmm3
  10b260:	62 f1 7d 48 71 d0 04 	vpsrlw $0x4,%zmm0,%zmm0
  10b267:	62 f1 75 48 71 d1 04 	vpsrlw $0x4,%zmm1,%zmm1
  10b26e:	62 71 2d 48 db d7    	vpandd %zmm7,%zmm10,%zmm10
  10b274:	62 f1 7d 48 db c7    	vpandd %zmm7,%zmm0,%zmm0
  10b27a:	62 71 25 48 db df    	vpandd %zmm7,%zmm11,%zmm11
  10b280:	62 f1 75 48 db cf    	vpandd %zmm7,%zmm1,%zmm1
  10b286:	c4 e1 f9 6e f8       	vmovq  %rax,%xmm7
  10b28b:	41 0f b6 41 01       	movzbl 0x1(%r9),%eax
  10b290:	62 f2 3d 48 00 c9    	vpshufb %zmm1,%zmm8,%zmm1
  10b296:	62 f2 3d 48 00 c0    	vpshufb %zmm0,%zmm8,%zmm0
  10b29c:	62 e1 7d 48 70 c9 88 	vpshufd $0x88,%zmm1,%zmm17
  10b2a3:	62 f1 7d 48 70 c9 dd 	vpshufd $0xdd,%zmm1,%zmm1
  10b2aa:	62 e1 fd 08 6e e8    	vmovq  %rax,%xmm21
  10b2b0:	41 0f b6 41 02       	movzbl 0x2(%r9),%eax
  10b2b5:	62 f2 3d 48 00 e4    	vpshufb %zmm4,%zmm8,%zmm4
  10b2bb:	62 f2 3d 48 00 db    	vpshufb %zmm3,%zmm8,%zmm3
  10b2c1:	62 f2 3d 48 00 d2    	vpshufb %zmm2,%zmm8,%zmm2
  10b2c7:	62 52 3d 48 00 ed    	vpshufb %zmm13,%zmm8,%zmm13
  10b2cd:	62 e1 fd 08 6e f8    	vmovq  %rax,%xmm23
  10b2d3:	41 0f b6 41 03       	movzbl 0x3(%r9),%eax
  10b2d8:	62 52 3d 48 00 d2    	vpshufb %zmm10,%zmm8,%zmm10
  10b2de:	62 52 3d 48 00 db    	vpshufb %zmm11,%zmm8,%zmm11
  10b2e4:	62 e1 7d 48 70 d4 88 	vpshufd $0x88,%zmm4,%zmm18
  10b2eb:	62 e1 7d 48 70 c0 88 	vpshufd $0x88,%zmm0,%zmm16
  10b2f2:	62 e1 fd 08 6e f0    	vmovq  %rax,%xmm22
  10b2f8:	41 0f b6 41 04       	movzbl 0x4(%r9),%eax
  10b2fd:	62 71 7d 48 70 fb 88 	vpshufd $0x88,%zmm3,%zmm15
  10b304:	c4 e1 f9 6e e8       	vmovq  %rax,%xmm5
  10b309:	41 0f b6 41 05       	movzbl 0x5(%r9),%eax
  10b30e:	62 61 fd 08 6e c8    	vmovq  %rax,%xmm25
  10b314:	41 0f b6 41 06       	movzbl 0x6(%r9),%eax
  10b319:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  10b31e:	41 0f b6 41 07       	movzbl 0x7(%r9),%eax
  10b323:	62 61 fd 08 6e d0    	vmovq  %rax,%xmm26
  10b329:	c4 e1 f9 7e f0       	vmovq  %xmm6,%rax
  10b32e:	c5 fa 10 34 86       	vmovss (%rsi,%rax,4),%xmm6
  10b333:	62 61 fd 08 7e d0    	vmovq  %xmm26,%rax
  10b339:	c4 e3 49 21 34 86 10 	vinsertps $0x10,(%rsi,%rax,4),%xmm6,%xmm6
  10b340:	c4 e1 f9 7e e8       	vmovq  %xmm5,%rax
  10b345:	62 f1 7d 48 7f 4c 24 	vmovdqa32 %zmm1,0x480(%rsp)
  10b34c:	12 
  10b34d:	c5 fa 10 2c 86       	vmovss (%rsi,%rax,4),%xmm5
  10b352:	62 61 fd 08 7e c8    	vmovq  %xmm25,%rax
  10b358:	c4 e3 51 21 2c 86 10 	vinsertps $0x10,(%rsi,%rax,4),%xmm5,%xmm5
  10b35f:	62 e1 fd 08 7e f8    	vmovq  %xmm23,%rax
  10b365:	62 f1 7d 48 70 c8 dd 	vpshufd $0xdd,%zmm0,%zmm1
  10b36c:	62 f1 7d 48 7f 4c 24 	vmovdqa32 %zmm1,0x440(%rsp)
  10b373:	11 
  10b374:	62 f1 7d 48 70 cb dd 	vpshufd $0xdd,%zmm3,%zmm1
  10b37b:	62 f1 7d 48 7f 4c 24 	vmovdqa32 %zmm1,0x400(%rsp)
  10b382:	10 
  10b383:	62 71 fd 48 7f 44 24 	vmovdqa64 %zmm8,0x3c0(%rsp)
  10b38a:	0f 
  10b38b:	62 c1 7d 48 70 fb 88 	vpshufd $0x88,%zmm11,%zmm23
  10b392:	c5 d0 16 ee          	vmovlhps %xmm6,%xmm5,%xmm5
  10b396:	c5 fa 10 34 86       	vmovss (%rsi,%rax,4),%xmm6
  10b39b:	62 e1 fd 08 7e f0    	vmovq  %xmm22,%rax
  10b3a1:	c4 e3 49 21 34 86 10 	vinsertps $0x10,(%rsi,%rax,4),%xmm6,%xmm6
  10b3a8:	c4 e1 f9 7e f8       	vmovq  %xmm7,%rax
  10b3ad:	62 c1 7d 48 70 f2 88 	vpshufd $0x88,%zmm10,%zmm22
  10b3b4:	c5 fa 10 3c 86       	vmovss (%rsi,%rax,4),%xmm7
  10b3b9:	62 e1 fd 08 7e e8    	vmovq  %xmm21,%rax
  10b3bf:	c4 e3 41 21 3c 86 10 	vinsertps $0x10,(%rsi,%rax,4),%xmm7,%xmm7
  10b3c6:	c4 61 f9 7e e0       	vmovq  %xmm12,%rax
  10b3cb:	62 c1 7d 48 70 ed 88 	vpshufd $0x88,%zmm13,%zmm21
  10b3d2:	62 51 7d 48 70 db dd 	vpshufd $0xdd,%zmm11,%zmm11
  10b3d9:	62 51 7d 48 70 d2 dd 	vpshufd $0xdd,%zmm10,%zmm10
  10b3e0:	62 51 7d 48 70 ed dd 	vpshufd $0xdd,%zmm13,%zmm13
  10b3e7:	62 71 7d 48 70 e2 dd 	vpshufd $0xdd,%zmm2,%zmm12
  10b3ee:	c5 c0 16 fe          	vmovlhps %xmm6,%xmm7,%xmm7
  10b3f2:	c5 fa 10 34 8e       	vmovss (%rsi,%rcx,4),%xmm6
  10b3f7:	c4 e3 49 21 34 86 10 	vinsertps $0x10,(%rsi,%rax,4),%xmm6,%xmm6
  10b3fe:	c4 e3 45 18 fd 01    	vinsertf128 $0x1,%xmm5,%ymm7,%ymm7
  10b404:	c4 a1 7a 10 2c be    	vmovss (%rsi,%r15,4),%xmm5
  10b40a:	c4 a3 51 21 2c 9e 10 	vinsertps $0x10,(%rsi,%r11,4),%xmm5,%xmm5
  10b411:	c5 50 16 f6          	vmovlhps %xmm6,%xmm5,%xmm14
  10b415:	c4 a1 7a 10 34 86    	vmovss (%rsi,%r8,4),%xmm6
  10b41b:	c5 fa 10 2c 9e       	vmovss (%rsi,%rbx,4),%xmm5
  10b420:	c4 a3 49 21 34 ae 10 	vinsertps $0x10,(%rsi,%r13,4),%xmm6,%xmm6
  10b427:	c4 e3 51 21 2c 96 10 	vinsertps $0x10,(%rsi,%rdx,4),%xmm5,%xmm5
  10b42e:	4c 8b ac 24 c0 02 00 	mov    0x2c0(%rsp),%r13
  10b435:	00 
  10b436:	48 8b 94 24 40 03 00 	mov    0x340(%rsp),%rdx
  10b43d:	00 
  10b43e:	c5 d0 16 ee          	vmovlhps %xmm6,%xmm5,%xmm5
  10b442:	c4 c3 55 18 ee 01    	vinsertf128 $0x1,%xmm14,%ymm5,%ymm5
  10b448:	62 71 7d 48 70 f2 88 	vpshufd $0x88,%zmm2,%zmm14
  10b44f:	62 f3 d5 48 1a f7 01 	vinsertf64x4 $0x1,%ymm7,%zmm5,%zmm6
  10b456:	62 f1 7d 48 70 fc dd 	vpshufd $0xdd,%zmm4,%zmm7
  10b45d:	62 f1 7d 48 7f 7c 24 	vmovdqa32 %zmm7,0x4c0(%rsp)
  10b464:	13 
  10b465:	62 f1 7c 48 29 74 24 	vmovaps %zmm6,0x500(%rsp)
  10b46c:	14 
  10b46d:	49 8b 45 00          	mov    0x0(%r13),%rax
  10b471:	62 21 fd 48 6f ff    	vmovdqa64 %zmm23,%zmm31
  10b477:	48 81 c2 00 01 00 00 	add    $0x100,%rdx
  10b47e:	49 83 c5 08          	add    $0x8,%r13
  10b482:	4c 01 f0             	add    %r14,%rax
  10b485:	c5 fe 6f 58 68       	vmovdqu 0x68(%rax),%ymm3
  10b48a:	c5 fe 6f 50 48       	vmovdqu 0x48(%rax),%ymm2
  10b48f:	c5 fe 6f 48 28       	vmovdqu 0x28(%rax),%ymm1
  10b494:	c5 fe 6f 40 08       	vmovdqu 0x8(%rax),%ymm0
  10b499:	c4 e3 65 46 fb 00    	vperm2i128 $0x0,%ymm3,%ymm3,%ymm7
  10b49f:	c4 e3 6d 46 f2 00    	vperm2i128 $0x0,%ymm2,%ymm2,%ymm6
  10b4a5:	c4 e3 65 46 db 11    	vperm2i128 $0x11,%ymm3,%ymm3,%ymm3
  10b4ab:	c4 e3 6d 46 d2 11    	vperm2i128 $0x11,%ymm2,%ymm2,%ymm2
  10b4b1:	62 f3 c5 48 3a ff 01 	vinserti64x4 $0x1,%ymm7,%zmm7,%zmm7
  10b4b8:	62 f3 cd 48 3a f6 01 	vinserti64x4 $0x1,%ymm6,%zmm6,%zmm6
  10b4bf:	c4 e3 75 46 e9 00    	vperm2i128 $0x0,%ymm1,%ymm1,%ymm5
  10b4c5:	c4 e3 7d 46 e0 00    	vperm2i128 $0x0,%ymm0,%ymm0,%ymm4
  10b4cb:	62 71 7d 48 70 c7 a0 	vpshufd $0xa0,%zmm7,%zmm8
  10b4d2:	62 f3 d5 48 3a ed 01 	vinserti64x4 $0x1,%ymm5,%zmm5,%zmm5
  10b4d9:	62 d2 7e 48 29 e8    	vpmovb2m %zmm8,%k5
  10b4df:	62 42 7d 48 1c d8    	vpabsb %zmm8,%zmm27
  10b4e5:	62 31 7d 48 6f c4    	vmovdqa32 %zmm20,%zmm8
  10b4eb:	62 f3 dd 48 3a e4 01 	vinserti64x4 $0x1,%ymm4,%zmm4,%zmm4
  10b4f2:	62 f3 e5 48 3a db 01 	vinserti64x4 $0x1,%ymm3,%zmm3,%zmm3
  10b4f9:	62 61 7d 48 70 f6 a0 	vpshufd $0xa0,%zmm6,%zmm30
  10b500:	62 f3 ed 48 3a d2 01 	vinserti64x4 $0x1,%ymm2,%zmm2,%zmm2
  10b507:	c4 e3 75 46 c9 11    	vperm2i128 $0x11,%ymm1,%ymm1,%ymm1
  10b50d:	c4 e3 7d 46 c0 11    	vperm2i128 $0x11,%ymm0,%ymm0,%ymm0
  10b513:	62 61 7d 48 70 ed a0 	vpshufd $0xa0,%zmm5,%zmm29
  10b51a:	62 f3 f5 48 3a c9 01 	vinserti64x4 $0x1,%ymm1,%zmm1,%zmm1
  10b521:	62 92 7e 48 29 e6    	vpmovb2m %zmm30,%k4
  10b527:	62 21 65 45 f8 ff    	vpsubb %zmm23,%zmm19,%zmm31{%k5}
  10b52d:	62 61 7d 48 70 e4 a0 	vpshufd $0xa0,%zmm4,%zmm28
  10b534:	62 f3 fd 48 3a c0 01 	vinserti64x4 $0x1,%ymm0,%zmm0,%zmm0
  10b53b:	62 61 7d 48 70 d3 a0 	vpshufd $0xa0,%zmm3,%zmm26
  10b542:	62 61 7d 48 70 ca a0 	vpshufd $0xa0,%zmm2,%zmm25
  10b549:	62 12 25 40 50 c7    	vpdpbusd %zmm31,%zmm27,%zmm8
  10b54f:	62 02 7d 48 1c fe    	vpabsb %zmm30,%zmm31
  10b555:	62 21 fd 48 6f f6    	vmovdqa64 %zmm22,%zmm30
  10b55b:	62 61 7d 48 70 c1 a0 	vpshufd $0xa0,%zmm1,%zmm24
  10b562:	62 71 7d 48 70 c8 a0 	vpshufd $0xa0,%zmm0,%zmm9
  10b569:	62 92 7e 48 29 dd    	vpmovb2m %zmm29,%k3
  10b56f:	62 21 65 44 f8 f6    	vpsubb %zmm22,%zmm19,%zmm30{%k4}
  10b575:	62 f1 7d 48 70 ff f5 	vpshufd $0xf5,%zmm7,%zmm7
  10b57c:	62 f1 7d 48 70 f6 f5 	vpshufd $0xf5,%zmm6,%zmm6
  10b583:	62 f1 7d 48 70 ed f5 	vpshufd $0xf5,%zmm5,%zmm5
  10b58a:	62 f1 7d 48 70 e4 f5 	vpshufd $0xf5,%zmm4,%zmm4
  10b591:	62 12 05 40 50 c6    	vpdpbusd %zmm30,%zmm31,%zmm8
  10b597:	62 02 7d 48 1c f5    	vpabsb %zmm29,%zmm30
  10b59d:	62 21 fd 48 6f ed    	vmovdqa64 %zmm21,%zmm29
  10b5a3:	62 f1 7d 48 70 db f5 	vpshufd $0xf5,%zmm3,%zmm3
  10b5aa:	62 f1 7d 48 70 d2 f5 	vpshufd $0xf5,%zmm2,%zmm2
  10b5b1:	62 92 7e 48 29 d4    	vpmovb2m %zmm28,%k2
  10b5b7:	62 21 65 43 f8 ed    	vpsubb %zmm21,%zmm19,%zmm29{%k3}
  10b5bd:	62 f1 7d 48 70 c9 f5 	vpshufd $0xf5,%zmm1,%zmm1
  10b5c4:	62 f1 7d 48 70 c0 f5 	vpshufd $0xf5,%zmm0,%zmm0
  10b5cb:	62 12 0d 40 50 c5    	vpdpbusd %zmm29,%zmm30,%zmm8
  10b5d1:	62 02 7d 48 1c ec    	vpabsb %zmm28,%zmm29
  10b5d7:	62 21 fd 48 6f e2    	vmovdqa64 %zmm18,%zmm28
  10b5dd:	62 21 65 42 f8 e2    	vpsubb %zmm18,%zmm19,%zmm28{%k2}
  10b5e3:	62 12 15 40 50 c4    	vpdpbusd %zmm28,%zmm29,%zmm8
  10b5e9:	62 21 fd 48 6f e1    	vmovdqa64 %zmm17,%zmm28
  10b5ef:	62 71 7d 48 7f 44 24 	vmovdqa32 %zmm8,0x540(%rsp)
  10b5f6:	15 
  10b5f7:	62 31 7d 48 6f c4    	vmovdqa32 %zmm20,%zmm8
  10b5fd:	62 21 65 45 f8 e1    	vpsubb %zmm17,%zmm19,%zmm28{%k5}
  10b603:	62 92 7e 48 29 ea    	vpmovb2m %zmm26,%k5
  10b609:	62 12 25 40 50 c4    	vpdpbusd %zmm28,%zmm27,%zmm8
  10b60f:	62 21 fd 48 6f d8    	vmovdqa64 %zmm16,%zmm27
  10b615:	62 21 65 44 f8 d8    	vpsubb %zmm16,%zmm19,%zmm27{%k4}
  10b61b:	62 61 fd 48 6f 64 24 	vmovdqa64 0x480(%rsp),%zmm28
  10b622:	12 
  10b623:	62 12 05 40 50 c3    	vpdpbusd %zmm27,%zmm31,%zmm8
  10b629:	62 41 fd 48 6f ff    	vmovdqa64 %zmm15,%zmm31
  10b62f:	62 41 65 43 f8 ff    	vpsubb %zmm15,%zmm19,%zmm31{%k3}
  10b635:	62 61 fd 48 6f 5c 24 	vmovdqa64 0x4c0(%rsp),%zmm27
  10b63c:	13 
  10b63d:	62 12 0d 40 50 c7    	vpdpbusd %zmm31,%zmm30,%zmm8
  10b643:	62 41 fd 48 6f f6    	vmovdqa64 %zmm14,%zmm30
  10b649:	62 41 7d 48 6f f8    	vmovdqa32 %zmm8,%zmm31
  10b64f:	62 41 65 42 f8 f6    	vpsubb %zmm14,%zmm19,%zmm30{%k2}
  10b655:	62 31 7d 48 6f c4    	vmovdqa32 %zmm20,%zmm8
  10b65b:	62 92 7e 48 29 e1    	vpmovb2m %zmm25,%k4
  10b661:	62 02 15 40 50 fe    	vpdpbusd %zmm30,%zmm29,%zmm31
  10b667:	62 02 7d 48 1c ea    	vpabsb %zmm26,%zmm29
  10b66d:	62 21 fd 48 6f d7    	vmovdqa64 %zmm23,%zmm26
  10b673:	62 21 65 45 f8 d7    	vpsubb %zmm23,%zmm19,%zmm26{%k5}
  10b679:	62 12 15 40 50 c2    	vpdpbusd %zmm26,%zmm29,%zmm8
  10b67f:	62 02 7d 48 1c d1    	vpabsb %zmm25,%zmm26
  10b685:	62 21 fd 48 6f ce    	vmovdqa64 %zmm22,%zmm25
  10b68b:	62 92 7e 48 29 d8    	vpmovb2m %zmm24,%k3
  10b691:	62 21 65 44 f8 ce    	vpsubb %zmm22,%zmm19,%zmm25{%k4}
  10b697:	62 12 2d 40 50 c1    	vpdpbusd %zmm25,%zmm26,%zmm8
  10b69d:	62 02 7d 48 1c c8    	vpabsb %zmm24,%zmm25
  10b6a3:	62 21 fd 48 6f c5    	vmovdqa64 %zmm21,%zmm24
  10b6a9:	62 21 65 43 f8 c5    	vpsubb %zmm21,%zmm19,%zmm24{%k3}
  10b6af:	62 12 35 40 50 c0    	vpdpbusd %zmm24,%zmm25,%zmm8
  10b6b5:	62 42 7d 48 1c c1    	vpabsb %zmm9,%zmm24
  10b6bb:	62 d2 7e 48 29 d1    	vpmovb2m %zmm9,%k2
  10b6c1:	62 31 fd 48 6f ca    	vmovdqa64 %zmm18,%zmm9
  10b6c7:	62 41 7d 48 6f f0    	vmovdqa32 %zmm8,%zmm30
  10b6cd:	62 31 7d 48 6f c4    	vmovdqa32 %zmm20,%zmm8
  10b6d3:	62 31 65 42 f8 ca    	vpsubb %zmm18,%zmm19,%zmm9{%k2}
  10b6d9:	62 42 3d 40 50 f1    	vpdpbusd %zmm9,%zmm24,%zmm30
  10b6df:	62 31 fd 48 6f c9    	vmovdqa64 %zmm17,%zmm9
  10b6e5:	62 31 65 45 f8 c9    	vpsubb %zmm17,%zmm19,%zmm9{%k5}
  10b6eb:	62 52 15 40 50 c1    	vpdpbusd %zmm9,%zmm29,%zmm8
  10b6f1:	62 31 fd 48 6f c8    	vmovdqa64 %zmm16,%zmm9
  10b6f7:	62 31 65 44 f8 c8    	vpsubb %zmm16,%zmm19,%zmm9{%k4}
  10b6fd:	62 61 fd 48 6f 6c 24 	vmovdqa64 0x400(%rsp),%zmm29
  10b704:	10 
  10b705:	62 52 2d 40 50 c1    	vpdpbusd %zmm9,%zmm26,%zmm8
  10b70b:	62 51 fd 48 6f cf    	vmovdqa64 %zmm15,%zmm9
  10b711:	62 51 65 43 f8 cf    	vpsubb %zmm15,%zmm19,%zmm9{%k3}
  10b717:	62 61 fd 48 6f 54 24 	vmovdqa64 0x440(%rsp),%zmm26
  10b71e:	11 
  10b71f:	62 52 35 40 50 c1    	vpdpbusd %zmm9,%zmm25,%zmm8
  10b725:	62 51 fd 48 6f ce    	vmovdqa64 %zmm14,%zmm9
  10b72b:	62 41 fd 48 6f cd    	vmovdqa64 %zmm13,%zmm25
  10b731:	62 f2 7e 48 29 ef    	vpmovb2m %zmm7,%k5
  10b737:	62 51 65 42 f8 ce    	vpsubb %zmm14,%zmm19,%zmm9{%k2}
  10b73d:	62 52 3d 40 50 c1    	vpdpbusd %zmm9,%zmm24,%zmm8
  10b743:	62 72 7d 48 1c cf    	vpabsb %zmm7,%zmm9
  10b749:	62 d1 fd 48 6f fb    	vmovdqa64 %zmm11,%zmm7
  10b74f:	62 21 7d 48 6f c4    	vmovdqa32 %zmm20,%zmm24
  10b755:	62 f2 7e 48 29 e6    	vpmovb2m %zmm6,%k4
  10b75b:	62 d1 65 45 f8 fb    	vpsubb %zmm11,%zmm19,%zmm7{%k5}
  10b761:	62 62 35 48 50 c7    	vpdpbusd %zmm7,%zmm9,%zmm24
  10b767:	62 f2 7d 48 1c fe    	vpabsb %zmm6,%zmm7
  10b76d:	62 d1 fd 48 6f f2    	vmovdqa64 %zmm10,%zmm6
  10b773:	62 d1 65 44 f8 f2    	vpsubb %zmm10,%zmm19,%zmm6{%k4}
  10b779:	62 62 45 48 50 c6    	vpdpbusd %zmm6,%zmm7,%zmm24
  10b77f:	62 f2 7d 48 1c f5    	vpabsb %zmm5,%zmm6
  10b785:	62 f2 7e 48 29 dd    	vpmovb2m %zmm5,%k3
  10b78b:	62 91 7d 48 6f e8    	vmovdqa32 %zmm24,%zmm5
  10b791:	62 62 7d 48 1c c4    	vpabsb %zmm4,%zmm24
  10b797:	62 f2 7e 48 29 d4    	vpmovb2m %zmm4,%k2
  10b79d:	62 41 65 43 f8 cd    	vpsubb %zmm13,%zmm19,%zmm25{%k3}
  10b7a3:	62 92 4d 48 50 e9    	vpdpbusd %zmm25,%zmm6,%zmm5
  10b7a9:	62 01 fd 48 6f cb    	vmovdqa64 %zmm27,%zmm25
  10b7af:	62 01 65 42 f8 cb    	vpsubb %zmm27,%zmm19,%zmm25{%k2}
  10b7b5:	62 92 3d 40 50 e9    	vpdpbusd %zmm25,%zmm24,%zmm5
  10b7bb:	62 01 fd 48 6f cc    	vmovdqa64 %zmm28,%zmm25
  10b7c1:	62 f1 7d 48 6f e5    	vmovdqa32 %zmm5,%zmm4
  10b7c7:	62 b1 7d 48 6f ec    	vmovdqa32 %zmm20,%zmm5
  10b7cd:	62 01 65 45 f8 cc    	vpsubb %zmm28,%zmm19,%zmm25{%k5}
  10b7d3:	62 f1 5d 48 fe 64 24 	vpaddd 0x540(%rsp),%zmm4,%zmm4
  10b7da:	15 
  10b7db:	62 f2 7e 48 29 eb    	vpmovb2m %zmm3,%k5
  10b7e1:	62 92 35 48 50 e9    	vpdpbusd %zmm25,%zmm9,%zmm5
  10b7e7:	62 11 fd 48 6f ca    	vmovdqa64 %zmm26,%zmm9
  10b7ed:	62 11 65 44 f8 ca    	vpsubb %zmm26,%zmm19,%zmm9{%k4}
  10b7f3:	62 d2 45 48 50 e9    	vpdpbusd %zmm9,%zmm7,%zmm5
  10b7f9:	62 91 fd 48 6f fd    	vmovdqa64 %zmm29,%zmm7
  10b7ff:	62 51 fd 48 6f cd    	vmovdqa64 %zmm13,%zmm9
  10b805:	62 91 65 43 f8 fd    	vpsubb %zmm29,%zmm19,%zmm7{%k3}
  10b80b:	62 f2 4d 48 50 ef    	vpdpbusd %zmm7,%zmm6,%zmm5
  10b811:	62 d1 fd 48 6f f4    	vmovdqa64 %zmm12,%zmm6
  10b817:	62 b1 7d 48 6f fc    	vmovdqa32 %zmm20,%zmm7
  10b81d:	62 d1 65 42 f8 f4    	vpsubb %zmm12,%zmm19,%zmm6{%k2}
  10b823:	62 f2 3d 40 50 ee    	vpdpbusd %zmm6,%zmm24,%zmm5
  10b829:	62 f2 7d 48 1c f3    	vpabsb %zmm3,%zmm6
  10b82f:	62 d1 fd 48 6f db    	vmovdqa64 %zmm11,%zmm3
  10b835:	62 f2 7e 48 29 e2    	vpmovb2m %zmm2,%k4
  10b83b:	62 d1 65 45 f8 db    	vpsubb %zmm11,%zmm19,%zmm3{%k5}
  10b841:	62 91 55 48 fe ef    	vpaddd %zmm31,%zmm5,%zmm5
  10b847:	62 f2 4d 48 50 fb    	vpdpbusd %zmm3,%zmm6,%zmm7
  10b84d:	62 f2 7d 48 1c da    	vpabsb %zmm2,%zmm3
  10b853:	62 d1 fd 48 6f d2    	vmovdqa64 %zmm10,%zmm2
  10b859:	62 d1 65 44 f8 d2    	vpsubb %zmm10,%zmm19,%zmm2{%k4}
  10b85f:	62 f2 65 48 50 fa    	vpdpbusd %zmm2,%zmm3,%zmm7
  10b865:	62 f2 7d 48 1c d1    	vpabsb %zmm1,%zmm2
  10b86b:	62 f2 7e 48 29 d9    	vpmovb2m %zmm1,%k3
  10b871:	62 f1 7d 48 6f cf    	vmovdqa32 %zmm7,%zmm1
  10b877:	62 f2 7d 48 1c f8    	vpabsb %zmm0,%zmm7
  10b87d:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  10b883:	62 51 65 43 f8 cd    	vpsubb %zmm13,%zmm19,%zmm9{%k3}
  10b889:	62 d2 6d 48 50 c9    	vpdpbusd %zmm9,%zmm2,%zmm1
  10b88f:	62 11 fd 48 6f cb    	vmovdqa64 %zmm27,%zmm9
  10b895:	62 11 65 42 f8 cb    	vpsubb %zmm27,%zmm19,%zmm9{%k2}
  10b89b:	62 d2 45 48 50 c9    	vpdpbusd %zmm9,%zmm7,%zmm1
  10b8a1:	62 11 fd 48 6f cc    	vmovdqa64 %zmm28,%zmm9
  10b8a7:	62 f1 7d 48 6f c1    	vmovdqa32 %zmm1,%zmm0
  10b8ad:	62 b1 7d 48 6f cc    	vmovdqa32 %zmm20,%zmm1
  10b8b3:	62 11 65 45 f8 cc    	vpsubb %zmm28,%zmm19,%zmm9{%k5}
  10b8b9:	62 d2 4d 48 50 c9    	vpdpbusd %zmm9,%zmm6,%zmm1
  10b8bf:	62 91 fd 48 6f f2    	vmovdqa64 %zmm26,%zmm6
  10b8c5:	62 91 7d 48 fe c6    	vpaddd %zmm30,%zmm0,%zmm0
  10b8cb:	62 91 65 44 f8 f2    	vpsubb %zmm26,%zmm19,%zmm6{%k4}
  10b8d1:	62 f2 65 48 50 ce    	vpdpbusd %zmm6,%zmm3,%zmm1
  10b8d7:	62 91 fd 48 6f dd    	vmovdqa64 %zmm29,%zmm3
  10b8dd:	62 91 65 43 f8 dd    	vpsubb %zmm29,%zmm19,%zmm3{%k3}
  10b8e3:	c5 f9 6f b4 24 90 05 	vmovdqa 0x590(%rsp),%xmm6
  10b8ea:	00 00 
  10b8ec:	62 f2 6d 48 50 cb    	vpdpbusd %zmm3,%zmm2,%zmm1
  10b8f2:	62 d1 fd 48 6f d4    	vmovdqa64 %zmm12,%zmm2
  10b8f8:	62 f1 7d 48 6f dc    	vmovdqa32 %zmm4,%zmm3
  10b8fe:	62 f1 7d 48 70 e4 4e 	vpshufd $0x4e,%zmm4,%zmm4
  10b905:	62 f1 7d 49 6f e5    	vmovdqa32 %zmm5,%zmm4{%k1}
  10b90b:	62 d1 65 42 f8 d4    	vpsubb %zmm12,%zmm19,%zmm2{%k2}
  10b911:	62 f1 7d 49 70 dd 4e 	vpshufd $0x4e,%zmm5,%zmm3{%k1}
  10b918:	62 f2 45 48 50 ca    	vpdpbusd %zmm2,%zmm7,%zmm1
  10b91e:	62 f1 7d 48 6f e8    	vmovdqa32 %zmm0,%zmm5
  10b924:	62 f1 7d 48 70 c0 4e 	vpshufd $0x4e,%zmm0,%zmm0
  10b92b:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  10b931:	62 f1 7c 48 5b e4    	vcvtdq2ps %zmm4,%zmm4
  10b937:	c4 e2 49 8c 10       	vpmaskmovd (%rax),%xmm6,%xmm2
  10b93c:	62 f1 7c 48 28 74 24 	vmovaps 0x500(%rsp),%zmm6
  10b943:	14 
  10b944:	62 d1 75 48 fe c8    	vpaddd %zmm8,%zmm1,%zmm1
  10b94a:	62 f1 7d 49 6f c1    	vmovdqa32 %zmm1,%zmm0{%k1}
  10b950:	62 f1 7d 49 70 e9 4e 	vpshufd $0x4e,%zmm1,%zmm5{%k1}
  10b957:	c5 f9 70 d2 44       	vpshufd $0x44,%xmm2,%xmm2
  10b95c:	c4 e3 6d 38 d2 01    	vinserti128 $0x1,%xmm2,%ymm2,%ymm2
  10b962:	62 f2 7d 48 13 d2    	vcvtph2ps %ymm2,%zmm2
  10b968:	62 f1 7c 48 5b c0    	vcvtdq2ps %zmm0,%zmm0
  10b96e:	62 f3 7d 48 04 ca 00 	vpermilps $0x0,%zmm2,%zmm1
  10b975:	62 f1 74 48 59 ce    	vmulps %zmm6,%zmm1,%zmm1
  10b97b:	62 f2 75 48 a8 5a fc 	vfmadd213ps -0x100(%rdx),%zmm1,%zmm3
  10b982:	62 f3 7d 48 04 ca 55 	vpermilps $0x55,%zmm2,%zmm1
  10b989:	62 f1 7c 48 29 5a fc 	vmovaps %zmm3,-0x100(%rdx)
  10b990:	62 f1 74 48 59 ce    	vmulps %zmm6,%zmm1,%zmm1
  10b996:	62 f3 7d 48 04 da aa 	vpermilps $0xaa,%zmm2,%zmm3
  10b99d:	62 f2 75 48 a8 62 fd 	vfmadd213ps -0xc0(%rdx),%zmm1,%zmm4
  10b9a4:	62 f1 7c 48 29 62 fd 	vmovaps %zmm4,-0xc0(%rdx)
  10b9ab:	62 f3 7d 48 04 d2 ff 	vpermilps $0xff,%zmm2,%zmm2
  10b9b2:	62 f1 7c 48 5b cd    	vcvtdq2ps %zmm5,%zmm1
  10b9b8:	62 f1 64 48 59 de    	vmulps %zmm6,%zmm3,%zmm3
  10b9be:	62 f2 65 48 a8 4a fe 	vfmadd213ps -0x80(%rdx),%zmm3,%zmm1
  10b9c5:	62 f1 7c 48 29 4a fe 	vmovaps %zmm1,-0x80(%rdx)
  10b9cc:	62 f1 6c 48 59 d6    	vmulps %zmm6,%zmm2,%zmm2
  10b9d2:	62 f2 6d 48 a8 42 ff 	vfmadd213ps -0x40(%rdx),%zmm2,%zmm0
  10b9d9:	62 f1 7c 48 29 42 ff 	vmovaps %zmm0,-0x40(%rdx)
  10b9e0:	49 39 d2             	cmp    %rdx,%r10
  10b9e3:	0f 85 84 fa ff ff    	jne    10b46d <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x71d>
  10b9e9:	62 71 fd 48 6f 44 24 	vmovdqa64 0x3c0(%rsp),%zmm8
  10b9f0:	0f 
  10b9f1:	49 ff c4             	inc    %r12
  10b9f4:	48 81 c7 88 00 00 00 	add    $0x88,%rdi
  10b9fb:	49 81 c1 88 00 00 00 	add    $0x88,%r9
  10ba02:	4c 39 a4 24 00 03 00 	cmp    %r12,0x300(%rsp)
  10ba09:	00 
  10ba0a:	0f 85 50 f7 ff ff    	jne    10b160 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x410>
  10ba10:	62 c1 fd 48 6f f8    	vmovdqa64 %zmm8,%zmm23
  10ba16:	62 e1 7c 48 28 54 24 	vmovaps 0x5c0(%rsp),%zmm18
  10ba1d:	17 
  10ba1e:	62 e1 7c 48 28 4c 24 	vmovaps 0x600(%rsp),%zmm17
  10ba25:	18 
  10ba26:	62 71 7c 48 28 7c 24 	vmovaps 0x640(%rsp),%zmm15
  10ba2d:	19 
  10ba2e:	62 71 7c 48 28 6c 24 	vmovaps 0x680(%rsp),%zmm13
  10ba35:	1a 
  10ba36:	62 71 7c 48 28 5c 24 	vmovaps 0x6c0(%rsp),%zmm11
  10ba3d:	1b 
  10ba3e:	62 71 7c 48 28 54 24 	vmovaps 0x700(%rsp),%zmm10
  10ba45:	1c 
  10ba46:	62 71 7c 48 28 4c 24 	vmovaps 0x740(%rsp),%zmm9
  10ba4d:	1d 
  10ba4e:	62 71 7c 48 28 44 24 	vmovaps 0x780(%rsp),%zmm8
  10ba55:	1e 
  10ba56:	62 f1 7c 48 28 7c 24 	vmovaps 0x7c0(%rsp),%zmm7
  10ba5d:	1f 
  10ba5e:	62 f1 7c 48 28 74 24 	vmovaps 0x800(%rsp),%zmm6
  10ba65:	20 
  10ba66:	62 f1 7c 48 28 6c 24 	vmovaps 0x840(%rsp),%zmm5
  10ba6d:	21 
  10ba6e:	62 f1 7c 48 28 64 24 	vmovaps 0x880(%rsp),%zmm4
  10ba75:	22 
  10ba76:	62 f1 7c 48 28 5c 24 	vmovaps 0x8c0(%rsp),%zmm3
  10ba7d:	23 
  10ba7e:	62 f1 7c 48 28 54 24 	vmovaps 0x900(%rsp),%zmm2
  10ba85:	24 
  10ba86:	62 f1 7c 48 28 4c 24 	vmovaps 0x940(%rsp),%zmm1
  10ba8d:	25 
  10ba8e:	62 f1 7c 48 28 44 24 	vmovaps 0x980(%rsp),%zmm0
  10ba95:	26 
  10ba96:	48 8b 9c 24 40 02 00 	mov    0x240(%rsp),%rbx
  10ba9d:	00 
  10ba9e:	48 8b 94 24 00 02 00 	mov    0x200(%rsp),%rdx
  10baa5:	00 
  10baa6:	48 8b 84 24 e0 01 00 	mov    0x1e0(%rsp),%rax
  10baad:	00 
  10baae:	4c 8b 84 24 c0 01 00 	mov    0x1c0(%rsp),%r8
  10bab5:	00 
  10bab6:	4c 8b 9c 24 a0 01 00 	mov    0x1a0(%rsp),%r11
  10babd:	00 
  10babe:	48 8b b4 24 88 01 00 	mov    0x188(%rsp),%rsi
  10bac5:	00 
  10bac6:	48 8b 8c 24 78 01 00 	mov    0x178(%rsp),%rcx
  10bacd:	00 
  10bace:	62 e1 7c 48 11 12    	vmovups %zmm18,(%rdx)
  10bad4:	48 83 c0 02          	add    $0x2,%rax
  10bad8:	4c 29 c6             	sub    %r8,%rsi
  10badb:	62 e1 7c 48 11 0c 8a 	vmovups %zmm17,(%rdx,%rcx,4)
  10bae2:	48 8b 8c 24 18 01 00 	mov    0x118(%rsp),%rcx
  10bae9:	00 
  10baea:	62 71 7c 48 11 3c b2 	vmovups %zmm15,(%rdx,%rsi,4)
  10baf1:	48 8b b4 24 80 01 00 	mov    0x180(%rsp),%rsi
  10baf8:	00 
  10baf9:	48 01 cb             	add    %rcx,%rbx
  10bafc:	49 01 cb             	add    %rcx,%r11
  10baff:	4c 29 c6             	sub    %r8,%rsi
  10bb02:	62 71 7c 48 11 2c b2 	vmovups %zmm13,(%rdx,%rsi,4)
  10bb09:	48 8b b4 24 70 01 00 	mov    0x170(%rsp),%rsi
  10bb10:	00 
  10bb11:	4c 29 c6             	sub    %r8,%rsi
  10bb14:	62 71 7c 48 11 1c b2 	vmovups %zmm11,(%rdx,%rsi,4)
  10bb1b:	48 8b b4 24 68 01 00 	mov    0x168(%rsp),%rsi
  10bb22:	00 
  10bb23:	4c 29 c6             	sub    %r8,%rsi
  10bb26:	62 71 7c 48 11 14 b2 	vmovups %zmm10,(%rdx,%rsi,4)
  10bb2d:	48 8b b4 24 60 01 00 	mov    0x160(%rsp),%rsi
  10bb34:	00 
  10bb35:	4c 29 c6             	sub    %r8,%rsi
  10bb38:	62 71 7c 48 11 0c b2 	vmovups %zmm9,(%rdx,%rsi,4)
  10bb3f:	48 8b b4 24 58 01 00 	mov    0x158(%rsp),%rsi
  10bb46:	00 
  10bb47:	4c 29 c6             	sub    %r8,%rsi
  10bb4a:	62 71 7c 48 11 04 b2 	vmovups %zmm8,(%rdx,%rsi,4)
  10bb51:	48 8b b4 24 90 01 00 	mov    0x190(%rsp),%rsi
  10bb58:	00 
  10bb59:	4c 29 c6             	sub    %r8,%rsi
  10bb5c:	62 f1 7c 48 11 3c b2 	vmovups %zmm7,(%rdx,%rsi,4)
  10bb63:	48 8b b4 24 50 01 00 	mov    0x150(%rsp),%rsi
  10bb6a:	00 
  10bb6b:	4c 29 c6             	sub    %r8,%rsi
  10bb6e:	62 f1 7c 48 11 34 b2 	vmovups %zmm6,(%rdx,%rsi,4)
  10bb75:	48 8b b4 24 48 01 00 	mov    0x148(%rsp),%rsi
  10bb7c:	00 
  10bb7d:	4c 29 c6             	sub    %r8,%rsi
  10bb80:	62 f1 7c 48 11 2c b2 	vmovups %zmm5,(%rdx,%rsi,4)
  10bb87:	48 8b b4 24 40 01 00 	mov    0x140(%rsp),%rsi
  10bb8e:	00 
  10bb8f:	4c 29 c6             	sub    %r8,%rsi
  10bb92:	62 f1 7c 48 11 24 b2 	vmovups %zmm4,(%rdx,%rsi,4)
  10bb99:	48 8b b4 24 38 01 00 	mov    0x138(%rsp),%rsi
  10bba0:	00 
  10bba1:	4c 29 c6             	sub    %r8,%rsi
  10bba4:	62 f1 7c 48 11 1c b2 	vmovups %zmm3,(%rdx,%rsi,4)
  10bbab:	48 8b b4 24 30 01 00 	mov    0x130(%rsp),%rsi
  10bbb2:	00 
  10bbb3:	4c 29 c6             	sub    %r8,%rsi
  10bbb6:	62 f1 7c 48 11 14 b2 	vmovups %zmm2,(%rdx,%rsi,4)
  10bbbd:	48 8b b4 24 28 01 00 	mov    0x128(%rsp),%rsi
  10bbc4:	00 
  10bbc5:	4c 29 c6             	sub    %r8,%rsi
  10bbc8:	62 f1 7c 48 11 0c b2 	vmovups %zmm1,(%rdx,%rsi,4)
  10bbcf:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
  10bbd6:	00 
  10bbd7:	4c 29 c6             	sub    %r8,%rsi
  10bbda:	62 f1 7c 48 11 04 b2 	vmovups %zmm0,(%rdx,%rsi,4)
  10bbe1:	48 29 8c 24 80 03 00 	sub    %rcx,0x380(%rsp)
  10bbe8:	00 
  10bbe9:	48 8b bc 24 98 01 00 	mov    0x198(%rsp),%rdi
  10bbf0:	00 
  10bbf1:	48 83 c2 40          	add    $0x40,%rdx
  10bbf5:	48 39 f8             	cmp    %rdi,%rax
  10bbf8:	0f 8c 72 f4 ff ff    	jl     10b070 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x320>
  10bbfe:	4c 8b 9c 24 c8 00 00 	mov    0xc8(%rsp),%r11
  10bc05:	00 
  10bc06:	48 8b bc 24 c0 00 00 	mov    0xc0(%rsp),%rdi
  10bc0d:	00 
  10bc0e:	48 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%rsi
  10bc15:	00 
  10bc16:	48 8b 8c 24 a8 00 00 	mov    0xa8(%rsp),%rcx
  10bc1d:	00 
  10bc1e:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  10bc25:	00 
  10bc26:	48 8b 9c 24 d0 00 00 	mov    0xd0(%rsp),%rbx
  10bc2d:	00 
  10bc2e:	49 83 c3 04          	add    $0x4,%r11
  10bc32:	48 8d 14 0f          	lea    (%rdi,%rcx,1),%rdx
  10bc36:	48 01 84 24 90 01 00 	add    %rax,0x190(%rsp)
  10bc3d:	00 
  10bc3e:	48 01 ce             	add    %rcx,%rsi
  10bc41:	48 01 9c 24 10 01 00 	add    %rbx,0x110(%rsp)
  10bc48:	00 
  10bc49:	49 01 c0             	add    %rax,%r8
  10bc4c:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  10bc53:	00 
  10bc54:	49 39 c3             	cmp    %rax,%r11
  10bc57:	0f 8c be f2 ff ff    	jl     10af1b <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1cb>
  10bc5d:	44 8b 84 24 90 00 00 	mov    0x90(%rsp),%r8d
  10bc64:	00 
  10bc65:	48 ff c8             	dec    %rax
  10bc68:	62 f1 7d 48 6f 74 24 	vmovdqa32 0x40(%rsp),%zmm6
  10bc6f:	01 
  10bc70:	62 e1 fd 28 6f 44 24 	vmovdqa64 0x20(%rsp),%ymm16
  10bc77:	01 
  10bc78:	44 8b 94 24 a0 00 00 	mov    0xa0(%rsp),%r10d
  10bc7f:	00 
  10bc80:	8b bc 24 e0 00 00 00 	mov    0xe0(%rsp),%edi
  10bc87:	4c 8b b4 24 98 00 00 	mov    0x98(%rsp),%r14
  10bc8e:	00 
  10bc8f:	44 8b 4d 10          	mov    0x10(%rbp),%r9d
  10bc93:	4c 8b bc 24 00 03 00 	mov    0x300(%rsp),%r15
  10bc9a:	00 
  10bc9b:	8b 9c 24 88 00 00 00 	mov    0x88(%rsp),%ebx
  10bca2:	48 c1 e8 02          	shr    $0x2,%rax
  10bca6:	45 85 c0             	test   %r8d,%r8d
  10bca9:	48 8d 0c 85 04 00 00 	lea    0x4(,%rax,4),%rcx
  10bcb0:	00 
  10bcb1:	41 8d 40 03          	lea    0x3(%r8),%eax
  10bcb5:	41 0f 49 c0          	cmovns %r8d,%eax
  10bcb9:	c1 f8 02             	sar    $0x2,%eax
  10bcbc:	48 98                	cltq
  10bcbe:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  10bcc5:	00 
  10bcc6:	48 39 c8             	cmp    %rcx,%rax
  10bcc9:	0f 8e 03 23 00 00    	jle    10dfd2 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x3282>
  10bccf:	85 ff                	test   %edi,%edi
  10bcd1:	4c 8b 9c 24 78 01 00 	mov    0x178(%rsp),%r11
  10bcd8:	00 
  10bcd9:	89 d8                	mov    %ebx,%eax
  10bcdb:	48 8b 9c 24 b8 00 00 	mov    0xb8(%rsp),%rbx
  10bce2:	00 
  10bce3:	0f 49 c7             	cmovns %edi,%eax
  10bce6:	48 8d 34 8d 01 00 00 	lea    0x1(,%rcx,4),%rsi
  10bced:	00 
  10bcee:	62 61 fd 48 6f c6    	vmovdqa64 %zmm6,%zmm24
  10bcf4:	62 01 2d 00 ef d2    	vpxord %xmm26,%xmm26,%xmm26
  10bcfa:	c1 f8 03             	sar    $0x3,%eax
  10bcfd:	44 89 94 24 10 01 00 	mov    %r10d,0x110(%rsp)
  10bd04:	00 
  10bd05:	4c 89 b4 24 08 01 00 	mov    %r14,0x108(%rsp)
  10bd0c:	00 
  10bd0d:	44 89 4d 10          	mov    %r9d,0x10(%rbp)
  10bd11:	62 e1 fd 28 7f 44 24 	vmovdqa64 %ymm16,0xe0(%rsp)
  10bd18:	07 
  10bd19:	4c 89 bc 24 e0 01 00 	mov    %r15,0x1e0(%rsp)
  10bd20:	00 
  10bd21:	48 98                	cltq
  10bd23:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  10bd2a:	00 
  10bd2b:	4a 8d 04 9d 00 00 00 	lea    0x0(,%r11,4),%rax
  10bd32:	00 
  10bd33:	49 0f af f3          	imul   %r11,%rsi
  10bd37:	c4 61 f9 6e d0       	vmovq  %rax,%xmm10
  10bd3c:	4c 89 d8             	mov    %r11,%rax
  10bd3f:	48 c1 e0 04          	shl    $0x4,%rax
  10bd43:	c4 41 f9 7e d0       	vmovq  %xmm10,%r8
  10bd48:	c5 79 d6 94 24 48 01 	vmovq  %xmm10,0x148(%rsp)
  10bd4f:	00 00 
  10bd51:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
  10bd58:	00 
  10bd59:	48 0f af c1          	imul   %rcx,%rax
  10bd5d:	49 89 f5             	mov    %rsi,%r13
  10bd60:	4c 0f af c1          	imul   %rcx,%r8
  10bd64:	48 8d 14 03          	lea    (%rbx,%rax,1),%rdx
  10bd68:	49 69 c7 88 00 00 00 	imul   $0x88,%r15,%rax
  10bd6f:	49 69 df 10 01 00 00 	imul   $0x110,%r15,%rbx
  10bd76:	4d 29 c5             	sub    %r8,%r13
  10bd79:	4c 89 ac 24 60 01 00 	mov    %r13,0x160(%rsp)
  10bd80:	00 
  10bd81:	48 89 84 24 50 01 00 	mov    %rax,0x150(%rsp)
  10bd88:	00 
  10bd89:	48 89 c8             	mov    %rcx,%rax
  10bd8c:	49 0f af c7          	imul   %r15,%rax
  10bd90:	48 69 c0 88 00 00 00 	imul   $0x88,%rax,%rax
  10bd97:	4c 01 f0             	add    %r14,%rax
  10bd9a:	48 89 84 24 70 01 00 	mov    %rax,0x170(%rsp)
  10bda1:	00 
  10bda2:	4c 89 c0             	mov    %r8,%rax
  10bda5:	49 89 d8             	mov    %rbx,%r8
  10bda8:	48 29 f0             	sub    %rsi,%rax
  10bdab:	48 89 84 24 18 01 00 	mov    %rax,0x118(%rsp)
  10bdb2:	00 
  10bdb3:	b8 0f 0f 0f 0f       	mov    $0xf0f0f0f,%eax
  10bdb8:	62 62 7d 48 7c c8    	vpbroadcastd %eax,%zmm25
  10bdbe:	4b 8d 44 1d 00       	lea    0x0(%r13,%r11,1),%rax
  10bdc3:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
  10bdca:	00 
  10bdcb:	83 ff 07             	cmp    $0x7,%edi
  10bdce:	0f 8e 50 0a 00 00    	jle    10c824 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1ad4>
  10bdd4:	41 bf cc cc ff ff    	mov    $0xffffcccc,%r15d
  10bdda:	48 8b 84 24 78 01 00 	mov    0x178(%rsp),%rax
  10bde1:	00 
  10bde2:	4c 8b 94 24 50 01 00 	mov    0x150(%rsp),%r10
  10bde9:	00 
  10bdea:	62 81 fd 48 6f e8    	vmovdqa64 %zmm24,%zmm21
  10bdf0:	c4 c1 78 92 cf       	kmovw  %r15d,%k1
  10bdf5:	4c 8b bc 24 18 01 00 	mov    0x118(%rsp),%r15
  10bdfc:	00 
  10bdfd:	31 db                	xor    %ebx,%ebx
  10bdff:	49 89 d1             	mov    %rdx,%r9
  10be02:	45 31 db             	xor    %r11d,%r11d
  10be05:	62 a1 4d 00 ef f6    	vpxord %xmm22,%xmm22,%xmm22
  10be0b:	49 89 dd             	mov    %rbx,%r13
  10be0e:	48 89 94 24 38 01 00 	mov    %rdx,0x138(%rsp)
  10be15:	00 
  10be16:	89 bc 24 30 01 00 00 	mov    %edi,0x130(%rsp)
  10be1d:	48 89 8c 24 28 01 00 	mov    %rcx,0x128(%rsp)
  10be24:	00 
  10be25:	48 89 b4 24 20 01 00 	mov    %rsi,0x120(%rsp)
  10be2c:	00 
  10be2d:	48 01 c0             	add    %rax,%rax
  10be30:	4c 29 f8             	sub    %r15,%rax
  10be33:	48 89 84 24 68 01 00 	mov    %rax,0x168(%rsp)
  10be3a:	00 
  10be3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  10be40:	83 bc 24 8c 05 00 00 	cmpl   $0x1f,0x58c(%rsp)
  10be47:	1f 
  10be48:	0f 8e c0 20 00 00    	jle    10df0e <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x31be>
  10be4e:	48 8b 84 24 80 05 00 	mov    0x580(%rsp),%rax
  10be55:	00 
  10be56:	48 8b b4 24 70 01 00 	mov    0x170(%rsp),%rsi
  10be5d:	00 
  10be5e:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
  10be62:	31 ff                	xor    %edi,%edi
  10be64:	62 f1 7c 48 29 64 24 	vmovaps %zmm4,0x500(%rsp)
  10be6b:	14 
  10be6c:	4c 89 ac 24 c0 01 00 	mov    %r13,0x1c0(%rsp)
  10be73:	00 
  10be74:	4c 89 94 24 a0 01 00 	mov    %r10,0x1a0(%rsp)
  10be7b:	00 
  10be7c:	4c 89 84 24 90 01 00 	mov    %r8,0x190(%rsp)
  10be83:	00 
  10be84:	4c 89 9c 24 88 01 00 	mov    %r11,0x188(%rsp)
  10be8b:	00 
  10be8c:	4c 89 8c 24 80 01 00 	mov    %r9,0x180(%rsp)
  10be93:	00 
  10be94:	62 f1 7c 48 29 64 24 	vmovaps %zmm4,0x4c0(%rsp)
  10be9b:	13 
  10be9c:	62 f1 7c 48 29 64 24 	vmovaps %zmm4,0x540(%rsp)
  10bea3:	15 
  10bea4:	4a 8d 0c 18          	lea    (%rax,%r11,1),%rcx
  10bea8:	4a 8d 14 10          	lea    (%rax,%r10,1),%rdx
  10beac:	48 8b 05 6d 00 04 00 	mov    0x4006d(%rip),%rax        # 14bf20 <ggml_table_f32_e8m0_half@@Base-0x1420>
  10beb3:	62 f1 7c 48 29 64 24 	vmovaps %zmm4,0x480(%rsp)
  10beba:	12 
  10bebb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  10bec0:	c5 fd 6f 0d 98 31 02 	vmovdqa 0x23198(%rip),%ymm1        # 12f060 <_ZL11iq2xxs_grid+0xb20>
  10bec7:	00 
  10bec8:	c5 fe 6f 71 08       	vmovdqu 0x8(%rcx),%ymm6
  10becd:	c4 e2 75 36 41 28    	vpermd 0x28(%rcx),%ymm1,%ymm0
  10bed3:	48 ff c7             	inc    %rdi
  10bed6:	44 0f b6 42 02       	movzbl 0x2(%rdx),%r8d
  10bedb:	0f b6 59 01          	movzbl 0x1(%rcx),%ebx
  10bedf:	44 0f b6 3a          	movzbl (%rdx),%r15d
  10bee3:	48 81 c1 88 00 00 00 	add    $0x88,%rcx
  10beea:	44 0f b6 89 78 ff ff 	movzbl -0x88(%rcx),%r9d
  10bef1:	ff 
  10bef2:	44 0f b6 99 7a ff ff 	movzbl -0x86(%rcx),%r11d
  10bef9:	ff 
  10befa:	44 0f b6 91 7b ff ff 	movzbl -0x85(%rcx),%r10d
  10bf01:	ff 
  10bf02:	48 81 c2 88 00 00 00 	add    $0x88,%rdx
  10bf09:	44 0f b6 a1 7c ff ff 	movzbl -0x84(%rcx),%r12d
  10bf10:	ff 
  10bf11:	44 0f b6 b1 7e ff ff 	movzbl -0x82(%rcx),%r14d
  10bf18:	ff 
  10bf19:	44 0f b6 a9 7f ff ff 	movzbl -0x81(%rcx),%r13d
  10bf20:	ff 
  10bf21:	48 81 c6 88 00 00 00 	add    $0x88,%rsi
  10bf28:	c5 fd 6f d1          	vmovdqa %ymm1,%ymm2
  10bf2c:	c4 e2 75 36 ee       	vpermd %ymm6,%ymm1,%ymm5
  10bf31:	c4 e3 4d 02 c0 f0    	vpblendd $0xf0,%ymm0,%ymm6,%ymm0
  10bf37:	c4 e2 75 36 49 e0    	vpermd -0x20(%rcx),%ymm1,%ymm1
  10bf3d:	c5 fe 6f 71 c0       	vmovdqu -0x40(%rcx),%ymm6
  10bf42:	c4 e2 6d 36 7a a0    	vpermd -0x60(%rdx),%ymm2,%ymm7
  10bf48:	c4 41 f9 6e c8       	vmovq  %r8,%xmm9
  10bf4d:	44 0f b6 82 7b ff ff 	movzbl -0x85(%rdx),%r8d
  10bf54:	ff 
  10bf55:	c4 e3 55 02 69 a0 f0 	vpblendd $0xf0,-0x60(%rcx),%ymm5,%ymm5
  10bf5c:	c4 41 f9 6e c0       	vmovq  %r8,%xmm8
  10bf61:	44 0f b6 82 7c ff ff 	movzbl -0x84(%rdx),%r8d
  10bf68:	ff 
  10bf69:	c4 e2 6d 36 e6       	vpermd %ymm6,%ymm2,%ymm4
  10bf6e:	c4 e3 4d 02 c9 f0    	vpblendd $0xf0,%ymm1,%ymm6,%ymm1
  10bf74:	c5 fe 6f 72 80       	vmovdqu -0x80(%rdx),%ymm6
  10bf79:	c4 e3 5d 02 61 e0 f0 	vpblendd $0xf0,-0x20(%rcx),%ymm4,%ymm4
  10bf80:	c4 e3 4d 02 ff f0    	vpblendd $0xf0,%ymm7,%ymm6,%ymm7
  10bf86:	c4 e2 6d 36 de       	vpermd %ymm6,%ymm2,%ymm3
  10bf8b:	c4 e2 6d 36 72 e0    	vpermd -0x20(%rdx),%ymm2,%ymm6
  10bf91:	c5 fe 6f 52 c0       	vmovdqu -0x40(%rdx),%ymm2
  10bf96:	c4 e3 65 02 5a a0 f0 	vpblendd $0xf0,-0x60(%rdx),%ymm3,%ymm3
  10bf9d:	62 f3 fd 48 3a c7 01 	vinserti64x4 $0x1,%ymm7,%zmm0,%zmm0
  10bfa4:	62 f3 d5 48 3a eb 01 	vinserti64x4 $0x1,%ymm3,%zmm5,%zmm5
  10bfab:	62 91 55 48 db f9    	vpandd %zmm25,%zmm5,%zmm7
  10bfb1:	62 f1 55 48 71 d5 04 	vpsrlw $0x4,%zmm5,%zmm5
  10bfb8:	62 91 55 48 db e9    	vpandd %zmm25,%zmm5,%zmm5
  10bfbe:	62 f2 55 40 00 ed    	vpshufb %zmm5,%zmm21,%zmm5
  10bfc4:	c4 e3 6d 02 f6 f0    	vpblendd $0xf0,%ymm6,%ymm2,%ymm6
  10bfca:	c5 fd 6f 15 8e 30 02 	vmovdqa 0x2308e(%rip),%ymm2        # 12f060 <_ZL11iq2xxs_grid+0xb20>
  10bfd1:	00 
  10bfd2:	c4 e2 6d 36 52 c0    	vpermd -0x40(%rdx),%ymm2,%ymm2
  10bfd8:	62 61 7d 48 70 dd dd 	vpshufd $0xdd,%zmm5,%zmm27
  10bfdf:	62 f3 f5 48 3a ce 01 	vinserti64x4 $0x1,%ymm6,%zmm1,%zmm1
  10bfe6:	62 71 7d 48 70 e5 88 	vpshufd $0x88,%zmm5,%zmm12
  10bfed:	62 91 75 48 db d9    	vpandd %zmm25,%zmm1,%zmm3
  10bff3:	62 f1 75 48 71 d1 04 	vpsrlw $0x4,%zmm1,%zmm1
  10bffa:	c4 c1 f9 6e e8       	vmovq  %r8,%xmm5
  10bfff:	44 0f b6 82 7d ff ff 	movzbl -0x83(%rdx),%r8d
  10c006:	ff 
  10c007:	62 91 75 48 db c9    	vpandd %zmm25,%zmm1,%zmm1
  10c00d:	62 f2 55 40 00 ff    	vpshufb %zmm7,%zmm21,%zmm7
  10c013:	62 e1 7d 48 70 ff dd 	vpshufd $0xdd,%zmm7,%zmm23
  10c01a:	c4 e3 6d 02 52 e0 f0 	vpblendd $0xf0,-0x20(%rdx),%ymm2,%ymm2
  10c021:	62 f3 dd 48 3a e2 01 	vinserti64x4 $0x1,%ymm2,%zmm4,%zmm4
  10c028:	62 91 7d 48 db d1    	vpandd %zmm25,%zmm0,%zmm2
  10c02e:	62 f1 7d 48 71 d0 04 	vpsrlw $0x4,%zmm0,%zmm0
  10c035:	62 71 7d 48 70 d7 88 	vpshufd $0x88,%zmm7,%zmm10
  10c03c:	62 91 5d 48 db f1    	vpandd %zmm25,%zmm4,%zmm6
  10c042:	62 f1 5d 48 71 d4 04 	vpsrlw $0x4,%zmm4,%zmm4
  10c049:	62 f2 55 40 00 f6    	vpshufb %zmm6,%zmm21,%zmm6
  10c04f:	62 91 5d 48 db e1    	vpandd %zmm25,%zmm4,%zmm4
  10c055:	62 91 7d 48 db c1    	vpandd %zmm25,%zmm0,%zmm0
  10c05b:	62 61 7d 48 70 c6 dd 	vpshufd $0xdd,%zmm6,%zmm24
  10c062:	c4 c1 f9 6e ff       	vmovq  %r15,%xmm7
  10c067:	44 0f b6 ba 79 ff ff 	movzbl -0x87(%rdx),%r15d
  10c06e:	ff 
  10c06f:	62 71 7d 48 70 de 88 	vpshufd $0x88,%zmm6,%zmm11
  10c076:	c4 c1 f9 6e f0       	vmovq  %r8,%xmm6
  10c07b:	44 0f b6 82 7e ff ff 	movzbl -0x82(%rdx),%r8d
  10c082:	ff 
  10c083:	62 f2 55 40 00 e4    	vpshufb %zmm4,%zmm21,%zmm4
  10c089:	62 61 7d 48 70 ec dd 	vpshufd $0xdd,%zmm4,%zmm29
  10c090:	62 71 7d 48 70 ec 88 	vpshufd $0x88,%zmm4,%zmm13
  10c097:	62 f2 55 40 00 c0    	vpshufb %zmm0,%zmm21,%zmm0
  10c09d:	c4 c1 f9 6e e0       	vmovq  %r8,%xmm4
  10c0a2:	44 0f b6 82 7f ff ff 	movzbl -0x81(%rdx),%r8d
  10c0a9:	ff 
  10c0aa:	62 e1 7d 48 70 c0 88 	vpshufd $0x88,%zmm0,%zmm16
  10c0b1:	62 f1 7d 48 70 c0 dd 	vpshufd $0xdd,%zmm0,%zmm0
  10c0b8:	62 f1 7d 48 7f 44 24 	vmovdqa32 %zmm0,0x440(%rsp)
  10c0bf:	11 
  10c0c0:	c4 e1 f9 6e c3       	vmovq  %rbx,%xmm0
  10c0c5:	0f b6 99 7d ff ff ff 	movzbl -0x83(%rcx),%ebx
  10c0cc:	62 c1 fd 08 6e d0    	vmovq  %r8,%xmm18
  10c0d2:	c4 c1 f9 7e e0       	vmovq  %xmm4,%r8
  10c0d7:	c4 a1 7a 10 24 80    	vmovss (%rax,%r8,4),%xmm4
  10c0dd:	62 c1 fd 08 7e d0    	vmovq  %xmm18,%r8
  10c0e3:	62 f2 55 40 00 c9    	vpshufb %zmm1,%zmm21,%zmm1
  10c0e9:	c4 a3 59 21 24 80 10 	vinsertps $0x10,(%rax,%r8,4),%xmm4,%xmm4
  10c0f0:	c4 c1 f9 7e e8       	vmovq  %xmm5,%r8
  10c0f5:	c4 a1 7a 10 2c 80    	vmovss (%rax,%r8,4),%xmm5
  10c0fb:	c4 c1 f9 7e f0       	vmovq  %xmm6,%r8
  10c100:	62 e1 7d 48 70 c9 88 	vpshufd $0x88,%zmm1,%zmm17
  10c107:	c4 a3 51 21 2c 80 10 	vinsertps $0x10,(%rax,%r8,4),%xmm5,%xmm5
  10c10e:	c4 41 f9 7e c8       	vmovq  %xmm9,%r8
  10c113:	62 f2 55 40 00 db    	vpshufb %zmm3,%zmm21,%zmm3
  10c119:	62 71 7d 48 70 fb 88 	vpshufd $0x88,%zmm3,%zmm15
  10c120:	62 f2 55 40 00 d2    	vpshufb %zmm2,%zmm21,%zmm2
  10c126:	62 71 7d 48 70 f2 88 	vpshufd $0x88,%zmm2,%zmm14
  10c12d:	62 f1 7d 48 70 c9 dd 	vpshufd $0xdd,%zmm1,%zmm1
  10c134:	c5 d0 16 f4          	vmovlhps %xmm4,%xmm5,%xmm6
  10c138:	c4 a1 7a 10 24 80    	vmovss (%rax,%r8,4),%xmm4
  10c13e:	c4 41 f9 7e c0       	vmovq  %xmm8,%r8
  10c143:	c4 a3 59 21 24 80 10 	vinsertps $0x10,(%rax,%r8,4),%xmm4,%xmm4
  10c14a:	c4 c1 f9 7e f8       	vmovq  %xmm7,%r8
  10c14f:	c4 a1 7a 10 3c 88    	vmovss (%rax,%r9,4),%xmm7
  10c155:	62 f1 7d 48 70 db dd 	vpshufd $0xdd,%zmm3,%zmm3
  10c15c:	c4 a1 7a 10 2c 80    	vmovss (%rax,%r8,4),%xmm5
  10c162:	c4 a3 51 21 2c b8 10 	vinsertps $0x10,(%rax,%r15,4),%xmm5,%xmm5
  10c169:	62 f1 7d 48 70 d2 dd 	vpshufd $0xdd,%zmm2,%zmm2
  10c170:	c5 d0 16 ec          	vmovlhps %xmm4,%xmm5,%xmm5
  10c174:	c4 a1 7a 10 24 a0    	vmovss (%rax,%r12,4),%xmm4
  10c17a:	c4 e3 59 21 24 98 10 	vinsertps $0x10,(%rax,%rbx,4),%xmm4,%xmm4
  10c181:	c4 e1 f9 7e c3       	vmovq  %xmm0,%rbx
  10c186:	c4 e3 55 18 ee 01    	vinsertf128 $0x1,%xmm6,%ymm5,%ymm5
  10c18c:	c4 a1 7a 10 34 b0    	vmovss (%rax,%r14,4),%xmm6
  10c192:	c4 a3 49 21 34 a8 10 	vinsertps $0x10,(%rax,%r13,4),%xmm6,%xmm6
  10c199:	c4 e3 41 21 3c 98 10 	vinsertps $0x10,(%rax,%rbx,4),%xmm7,%xmm7
  10c1a0:	c5 d8 16 e6          	vmovlhps %xmm6,%xmm4,%xmm4
  10c1a4:	c4 a1 7a 10 34 98    	vmovss (%rax,%r11,4),%xmm6
  10c1aa:	c4 a3 49 21 34 90 10 	vinsertps $0x10,(%rax,%r10,4),%xmm6,%xmm6
  10c1b1:	c5 c0 16 fe          	vmovlhps %xmm6,%xmm7,%xmm7
  10c1b5:	c4 e3 45 18 fc 01    	vinsertf128 $0x1,%xmm4,%ymm7,%ymm7
  10c1bb:	c5 fe 6f 66 e0       	vmovdqu -0x20(%rsi),%ymm4
  10c1c0:	62 e3 45 48 1a e5 01 	vinsertf32x8 $0x1,%ymm5,%zmm7,%zmm20
  10c1c7:	c5 fe 6f 7e 80       	vmovdqu -0x80(%rsi),%ymm7
  10c1cc:	c4 63 5d 46 cc 00    	vperm2i128 $0x0,%ymm4,%ymm4,%ymm9
  10c1d2:	c4 e3 5d 46 e4 11    	vperm2i128 $0x11,%ymm4,%ymm4,%ymm4
  10c1d8:	c4 e3 45 46 f7 00    	vperm2i128 $0x0,%ymm7,%ymm7,%ymm6
  10c1de:	62 53 b5 48 3a c9 01 	vinserti64x4 $0x1,%ymm9,%zmm9,%zmm9
  10c1e5:	62 f3 dd 48 3a e4 01 	vinserti64x4 $0x1,%ymm4,%zmm4,%zmm4
  10c1ec:	c4 e3 45 46 ff 11    	vperm2i128 $0x11,%ymm7,%ymm7,%ymm7
  10c1f2:	62 f3 c5 48 3a ff 01 	vinserti64x4 $0x1,%ymm7,%zmm7,%zmm7
  10c1f9:	62 e1 7d 28 6f de    	vmovdqa32 %ymm6,%ymm19
  10c1ff:	c5 fe 6f 76 a0       	vmovdqu -0x60(%rsi),%ymm6
  10c204:	62 f1 7d 48 70 c4 a0 	vpshufd $0xa0,%zmm4,%zmm0
  10c20b:	62 a3 65 40 3a db 01 	vinserti32x8 $0x1,%ymm19,%zmm19,%zmm19
  10c212:	62 f1 7d 48 70 e4 f5 	vpshufd $0xf5,%zmm4,%zmm4
  10c219:	62 f1 7d 48 7f 64 24 	vmovdqa32 %zmm4,0x200(%rsp)
  10c220:	08 
  10c221:	62 b1 fd 48 6f e1    	vmovdqa64 %zmm17,%zmm4
  10c227:	62 61 7d 48 70 f7 a0 	vpshufd $0xa0,%zmm7,%zmm30
  10c22e:	c4 e3 4d 46 ee 00    	vperm2i128 $0x0,%ymm6,%ymm6,%ymm5
  10c234:	62 61 7d 48 7f 74 24 	vmovdqa32 %zmm30,0x3c0(%rsp)
  10c23b:	0f 
  10c23c:	c4 e3 4d 46 f6 11    	vperm2i128 $0x11,%ymm6,%ymm6,%ymm6
  10c242:	62 e1 7d 28 6f d5    	vmovdqa32 %ymm5,%ymm18
  10c248:	c5 fe 6f 6e c0       	vmovdqu -0x40(%rsi),%ymm5
  10c24d:	62 f3 cd 48 3a f6 01 	vinserti64x4 $0x1,%ymm6,%zmm6,%zmm6
  10c254:	62 61 7d 48 70 fe a0 	vpshufd $0xa0,%zmm6,%zmm31
  10c25b:	62 a3 6d 40 3a d2 01 	vinserti32x8 $0x1,%ymm18,%zmm18,%zmm18
  10c262:	62 61 7d 48 7f 7c 24 	vmovdqa32 %zmm31,0x380(%rsp)
  10c269:	0e 
  10c26a:	62 f1 7d 48 70 f6 f5 	vpshufd $0xf5,%zmm6,%zmm6
  10c271:	62 f1 7d 48 7f 74 24 	vmovdqa32 %zmm6,0x280(%rsp)
  10c278:	0a 
  10c279:	62 21 7d 48 70 e3 a0 	vpshufd $0xa0,%zmm19,%zmm28
  10c280:	c4 63 55 46 c5 00    	vperm2i128 $0x0,%ymm5,%ymm5,%ymm8
  10c286:	62 a1 7d 48 70 db f5 	vpshufd $0xf5,%zmm19,%zmm19
  10c28d:	c4 e3 55 46 ed 11    	vperm2i128 $0x11,%ymm5,%ymm5,%ymm5
  10c293:	62 f3 d5 48 3a ed 01 	vinserti64x4 $0x1,%ymm5,%zmm5,%zmm5
  10c29a:	62 e1 7d 48 7f 54 24 	vmovdqa32 %zmm18,0x400(%rsp)
  10c2a1:	10 
  10c2a2:	62 53 bd 48 3a c0 01 	vinserti64x4 $0x1,%ymm8,%zmm8,%zmm8
  10c2a9:	62 61 7d 48 70 fd a0 	vpshufd $0xa0,%zmm5,%zmm31
  10c2b0:	62 61 7d 48 7f 7c 24 	vmovdqa32 %zmm31,0x340(%rsp)
  10c2b7:	0d 
  10c2b8:	62 41 7d 48 70 f9 a0 	vpshufd $0xa0,%zmm9,%zmm31
  10c2bf:	62 92 7e 48 29 ef    	vpmovb2m %zmm31,%k5
  10c2c5:	62 41 7d 48 70 f0 a0 	vpshufd $0xa0,%zmm8,%zmm30
  10c2cc:	62 f1 7d 48 70 ed f5 	vpshufd $0xf5,%zmm5,%zmm5
  10c2d3:	62 f1 7d 48 7f 6c 24 	vmovdqa32 %zmm5,0x240(%rsp)
  10c2da:	09 
  10c2db:	62 92 7d 48 1c ef    	vpabsb %zmm31,%zmm5
  10c2e1:	62 b1 4d 45 f8 e1    	vpsubb %zmm17,%zmm22,%zmm4{%k5}
  10c2e7:	62 e1 7d 48 7f 5c 24 	vmovdqa32 %zmm19,0x300(%rsp)
  10c2ee:	0c 
  10c2ef:	62 f1 fd 48 6f f4    	vmovdqa64 %zmm4,%zmm6
  10c2f5:	62 91 7d 48 6f e2    	vmovdqa32 %zmm26,%zmm4
  10c2fb:	62 92 7e 48 29 e6    	vpmovb2m %zmm30,%k4
  10c301:	62 e1 7d 48 70 df f5 	vpshufd $0xf5,%zmm7,%zmm19
  10c308:	62 f1 7d 48 70 7c 24 	vpshufd $0xf5,0x400(%rsp),%zmm7
  10c30f:	10 f5 
  10c311:	62 f1 7d 48 7f 7c 24 	vmovdqa32 %zmm7,0x400(%rsp)
  10c318:	10 
  10c319:	62 b1 fd 48 6f f8    	vmovdqa64 %zmm16,%zmm7
  10c31f:	62 a1 7d 48 70 d2 a0 	vpshufd $0xa0,%zmm18,%zmm18
  10c326:	62 f2 55 48 50 e6    	vpdpbusd %zmm6,%zmm5,%zmm4
  10c32c:	62 92 7d 48 1c f6    	vpabsb %zmm30,%zmm6
  10c332:	62 b1 4d 44 f8 f8    	vpsubb %zmm16,%zmm22,%zmm7{%k4}
  10c338:	62 e1 7d 48 7f 5c 24 	vmovdqa32 %zmm19,0x2c0(%rsp)
  10c33f:	0b 
  10c340:	62 51 7d 48 70 c9 f5 	vpshufd $0xf5,%zmm9,%zmm9
  10c347:	62 51 7d 48 70 c0 f5 	vpshufd $0xf5,%zmm8,%zmm8
  10c34e:	62 b2 7e 48 29 da    	vpmovb2m %zmm18,%k3
  10c354:	62 f2 4d 48 50 e7    	vpdpbusd %zmm7,%zmm6,%zmm4
  10c35a:	62 b2 7d 48 1c fa    	vpabsb %zmm18,%zmm7
  10c360:	62 c1 fd 48 6f d7    	vmovdqa64 %zmm15,%zmm18
  10c366:	62 92 7e 48 29 d4    	vpmovb2m %zmm28,%k2
  10c36c:	62 c1 4d 43 f8 d7    	vpsubb %zmm15,%zmm22,%zmm18{%k3}
  10c372:	62 b2 45 48 50 e2    	vpdpbusd %zmm18,%zmm7,%zmm4
  10c378:	62 82 7d 48 1c d4    	vpabsb %zmm28,%zmm18
  10c37e:	62 41 fd 48 6f e6    	vmovdqa64 %zmm14,%zmm28
  10c384:	62 41 4d 42 f8 e6    	vpsubb %zmm14,%zmm22,%zmm28{%k2}
  10c38a:	62 81 fd 48 6f dc    	vmovdqa64 %zmm28,%zmm19
  10c390:	62 61 7d 48 6f e4    	vmovdqa32 %zmm4,%zmm28
  10c396:	62 d1 fd 48 6f e5    	vmovdqa64 %zmm13,%zmm4
  10c39c:	62 22 6d 40 50 e3    	vpdpbusd %zmm19,%zmm18,%zmm28
  10c3a2:	62 d1 4d 45 f8 e5    	vpsubb %zmm13,%zmm22,%zmm4{%k5}
  10c3a8:	62 e1 fd 48 6f dc    	vmovdqa64 %zmm4,%zmm19
  10c3ae:	62 91 7d 48 6f e2    	vmovdqa32 %zmm26,%zmm4
  10c3b4:	62 f2 7e 48 29 e8    	vpmovb2m %zmm0,%k5
  10c3ba:	62 b2 55 48 50 e3    	vpdpbusd %zmm19,%zmm5,%zmm4
  10c3c0:	62 d1 fd 48 6f ec    	vmovdqa64 %zmm12,%zmm5
  10c3c6:	62 a1 4d 45 f8 c9    	vpsubb %zmm17,%zmm22,%zmm17{%k5}
  10c3cc:	62 51 4d 45 f8 ed    	vpsubb %zmm13,%zmm22,%zmm13{%k5}
  10c3d2:	62 d1 4d 44 f8 ec    	vpsubb %zmm12,%zmm22,%zmm5{%k4}
  10c3d8:	62 f2 4d 48 50 e5    	vpdpbusd %zmm5,%zmm6,%zmm4
  10c3de:	62 d1 fd 48 6f eb    	vmovdqa64 %zmm11,%zmm5
  10c3e4:	62 d1 4d 43 f8 eb    	vpsubb %zmm11,%zmm22,%zmm5{%k3}
  10c3ea:	62 f2 45 48 50 e5    	vpdpbusd %zmm5,%zmm7,%zmm4
  10c3f0:	62 d1 fd 48 6f ea    	vmovdqa64 %zmm10,%zmm5
  10c3f6:	62 f1 7d 48 6f 7c 24 	vmovdqa32 0x340(%rsp),%zmm7
  10c3fd:	0d 
  10c3fe:	62 d1 4d 42 f8 ea    	vpsubb %zmm10,%zmm22,%zmm5{%k2}
  10c404:	62 61 7d 48 6f 7c 24 	vmovdqa32 0x380(%rsp),%zmm31
  10c40b:	0e 
  10c40c:	62 f2 6d 40 50 e5    	vpdpbusd %zmm5,%zmm18,%zmm4
  10c412:	62 91 7d 48 6f ea    	vmovdqa32 %zmm26,%zmm5
  10c418:	62 61 7d 48 6f 74 24 	vmovdqa32 0x3c0(%rsp),%zmm30
  10c41f:	0f 
  10c420:	62 e1 7d 48 6f dc    	vmovdqa32 %zmm4,%zmm19
  10c426:	62 d2 7e 48 29 e9    	vpmovb2m %zmm9,%k5
  10c42c:	62 f2 7d 48 1c e0    	vpabsb %zmm0,%zmm4
  10c432:	62 f2 7d 48 1c f7    	vpabsb %zmm7,%zmm6
  10c438:	62 f2 7e 48 29 e7    	vpmovb2m %zmm7,%k4
  10c43e:	62 92 7d 48 1c ff    	vpabsb %zmm31,%zmm7
  10c444:	62 b2 5d 48 50 e9    	vpdpbusd %zmm17,%zmm4,%zmm5
  10c44a:	62 a1 4d 44 f8 c0    	vpsubb %zmm16,%zmm22,%zmm16{%k4}
  10c450:	62 51 4d 44 f8 e4    	vpsubb %zmm12,%zmm22,%zmm12{%k4}
  10c456:	62 92 7e 48 29 df    	vpmovb2m %zmm31,%k3
  10c45c:	62 b2 4d 48 50 e8    	vpdpbusd %zmm16,%zmm6,%zmm5
  10c462:	62 51 4d 43 f8 ff    	vpsubb %zmm15,%zmm22,%zmm15{%k3}
  10c468:	62 51 4d 43 f8 db    	vpsubb %zmm11,%zmm22,%zmm11{%k3}
  10c46e:	62 92 7e 48 29 d6    	vpmovb2m %zmm30,%k2
  10c474:	62 d2 45 48 50 ef    	vpdpbusd %zmm15,%zmm7,%zmm5
  10c47a:	62 51 4d 42 f8 f6    	vpsubb %zmm14,%zmm22,%zmm14{%k2}
  10c480:	62 12 7d 48 1c fe    	vpabsb %zmm30,%zmm15
  10c486:	62 51 4d 42 f8 d2    	vpsubb %zmm10,%zmm22,%zmm10{%k2}
  10c48c:	62 d2 05 48 50 ee    	vpdpbusd %zmm14,%zmm15,%zmm5
  10c492:	62 11 7d 48 6f f2    	vmovdqa32 %zmm26,%zmm14
  10c498:	62 52 5d 48 50 f5    	vpdpbusd %zmm13,%zmm4,%zmm14
  10c49e:	62 91 7d 48 6f e2    	vmovdqa32 %zmm26,%zmm4
  10c4a4:	62 52 4d 48 50 f4    	vpdpbusd %zmm12,%zmm6,%zmm14
  10c4aa:	62 f1 7d 48 6f 44 24 	vmovdqa32 0x440(%rsp),%zmm0
  10c4b1:	11 
  10c4b2:	62 71 7d 48 6f 64 24 	vmovdqa32 0x400(%rsp),%zmm12
  10c4b9:	10 
  10c4ba:	62 52 45 48 50 f3    	vpdpbusd %zmm11,%zmm7,%zmm14
  10c4c0:	62 d2 7d 48 1c f9    	vpabsb %zmm9,%zmm7
  10c4c6:	62 11 fd 48 6f dd    	vmovdqa64 %zmm29,%zmm11
  10c4cc:	62 52 7d 48 1c c8    	vpabsb %zmm8,%zmm9
  10c4d2:	62 52 05 48 50 f2    	vpdpbusd %zmm10,%zmm15,%zmm14
  10c4d8:	62 71 fd 48 6f f9    	vmovdqa64 %zmm1,%zmm15
  10c4de:	62 11 4d 45 f8 dd    	vpsubb %zmm29,%zmm22,%zmm11{%k5}
  10c4e4:	62 d2 7e 48 29 e0    	vpmovb2m %zmm8,%k4
  10c4ea:	62 71 4d 45 f8 f9    	vpsubb %zmm1,%zmm22,%zmm15{%k5}
  10c4f0:	62 52 7d 48 1c c4    	vpabsb %zmm12,%zmm8
  10c4f6:	62 f1 7d 48 6f 74 24 	vmovdqa32 0x300(%rsp),%zmm6
  10c4fd:	0c 
  10c4fe:	62 d2 45 48 50 e7    	vpdpbusd %zmm15,%zmm7,%zmm4
  10c504:	62 71 fd 48 6f f8    	vmovdqa64 %zmm0,%zmm15
  10c50a:	62 d2 7e 48 29 dc    	vpmovb2m %zmm12,%k3
  10c510:	62 71 4d 44 f8 f8    	vpsubb %zmm0,%zmm22,%zmm15{%k4}
  10c516:	62 71 fd 48 6f e0    	vmovdqa64 %zmm0,%zmm12
  10c51c:	62 72 7d 48 1c d6    	vpabsb %zmm6,%zmm10
  10c522:	62 d2 35 48 50 e7    	vpdpbusd %zmm15,%zmm9,%zmm4
  10c528:	62 71 fd 48 6f fb    	vmovdqa64 %zmm3,%zmm15
  10c52e:	62 f2 7e 48 29 d6    	vpmovb2m %zmm6,%k2
  10c534:	62 71 4d 43 f8 fb    	vpsubb %zmm3,%zmm22,%zmm15{%k3}
  10c53a:	62 f1 fd 48 6f f2    	vmovdqa64 %zmm2,%zmm6
  10c540:	62 d2 3d 48 50 e7    	vpdpbusd %zmm15,%zmm8,%zmm4
  10c546:	62 f1 4d 42 f8 f2    	vpsubb %zmm2,%zmm22,%zmm6{%k2}
  10c54c:	62 f2 2d 48 50 e6    	vpdpbusd %zmm6,%zmm10,%zmm4
  10c552:	62 91 7d 48 6f f2    	vmovdqa32 %zmm26,%zmm6
  10c558:	62 d2 45 48 50 f3    	vpdpbusd %zmm11,%zmm7,%zmm6
  10c55e:	62 91 fd 48 6f fb    	vmovdqa64 %zmm27,%zmm7
  10c564:	62 61 1d 40 fe e4    	vpaddd %zmm4,%zmm28,%zmm28
  10c56a:	62 91 4d 44 f8 fb    	vpsubb %zmm27,%zmm22,%zmm7{%k4}
  10c570:	62 71 7d 48 6f 5c 24 	vmovdqa32 0x200(%rsp),%zmm11
  10c577:	08 
  10c578:	62 f2 35 48 50 f7    	vpdpbusd %zmm7,%zmm9,%zmm6
  10c57e:	62 91 fd 48 6f f8    	vmovdqa64 %zmm24,%zmm7
  10c584:	62 91 4d 43 f8 f8    	vpsubb %zmm24,%zmm22,%zmm7{%k3}
  10c58a:	62 f2 3d 48 50 f7    	vpdpbusd %zmm7,%zmm8,%zmm6
  10c590:	62 b1 fd 48 6f ff    	vmovdqa64 %zmm23,%zmm7
  10c596:	62 11 7d 48 6f c2    	vmovdqa32 %zmm26,%zmm8
  10c59c:	62 b1 4d 42 f8 ff    	vpsubb %zmm23,%zmm22,%zmm7{%k2}
  10c5a2:	62 f2 2d 48 50 f7    	vpdpbusd %zmm7,%zmm10,%zmm6
  10c5a8:	62 d2 7d 48 1c fb    	vpabsb %zmm11,%zmm7
  10c5ae:	62 71 7d 48 6f 54 24 	vmovdqa32 0x240(%rsp),%zmm10
  10c5b5:	09 
  10c5b6:	62 e1 65 40 fe de    	vpaddd %zmm6,%zmm19,%zmm19
  10c5bc:	c5 f9 6f b4 24 90 05 	vmovdqa 0x590(%rsp),%xmm6
  10c5c3:	00 00 
  10c5c5:	62 d2 7e 48 29 eb    	vpmovb2m %zmm11,%k5
  10c5cb:	62 f1 4d 45 f8 c9    	vpsubb %zmm1,%zmm22,%zmm1{%k5}
  10c5d1:	62 72 45 48 50 c1    	vpdpbusd %zmm1,%zmm7,%zmm8
  10c5d7:	62 d2 7d 48 1c ca    	vpabsb %zmm10,%zmm1
  10c5dd:	62 d2 7e 48 29 e2    	vpmovb2m %zmm10,%k4
  10c5e3:	62 71 4d 44 f8 e0    	vpsubb %zmm0,%zmm22,%zmm12{%k4}
  10c5e9:	62 f1 7d 48 6f 44 24 	vmovdqa32 0x2c0(%rsp),%zmm0
  10c5f0:	0b 
  10c5f1:	62 52 75 48 50 c4    	vpdpbusd %zmm12,%zmm1,%zmm8
  10c5f7:	62 71 7d 48 6f 64 24 	vmovdqa32 0x280(%rsp),%zmm12
  10c5fe:	0a 
  10c5ff:	62 f2 7e 48 29 d0    	vpmovb2m %zmm0,%k2
  10c605:	62 f1 4d 42 f8 d2    	vpsubb %zmm2,%zmm22,%zmm2{%k2}
  10c60b:	62 d2 7e 48 29 dc    	vpmovb2m %zmm12,%k3
  10c611:	62 52 7d 48 1c cc    	vpabsb %zmm12,%zmm9
  10c617:	62 f1 4d 43 f8 db    	vpsubb %zmm3,%zmm22,%zmm3{%k3}
  10c61d:	62 72 35 48 50 c3    	vpdpbusd %zmm3,%zmm9,%zmm8
  10c623:	62 f2 7d 48 1c d8    	vpabsb %zmm0,%zmm3
  10c629:	62 91 7d 48 6f c2    	vmovdqa32 %zmm26,%zmm0
  10c62f:	62 72 65 48 50 c2    	vpdpbusd %zmm2,%zmm3,%zmm8
  10c635:	62 91 fd 48 6f d5    	vmovdqa64 %zmm29,%zmm2
  10c63b:	62 91 4d 45 f8 d5    	vpsubb %zmm29,%zmm22,%zmm2{%k5}
  10c641:	62 d1 55 48 fe e8    	vpaddd %zmm8,%zmm5,%zmm5
  10c647:	62 f2 45 48 50 c2    	vpdpbusd %zmm2,%zmm7,%zmm0
  10c64d:	62 91 fd 48 6f d3    	vmovdqa64 %zmm27,%zmm2
  10c653:	62 91 4d 44 f8 d3    	vpsubb %zmm27,%zmm22,%zmm2{%k4}
  10c659:	62 f2 75 48 50 c2    	vpdpbusd %zmm2,%zmm1,%zmm0
  10c65f:	62 91 fd 48 6f c8    	vmovdqa64 %zmm24,%zmm1
  10c665:	62 f1 7d 48 6f d5    	vmovdqa32 %zmm5,%zmm2
  10c66b:	62 f1 7d 48 70 ed 4e 	vpshufd $0x4e,%zmm5,%zmm5
  10c672:	62 91 4d 43 f8 c8    	vpsubb %zmm24,%zmm22,%zmm1{%k3}
  10c678:	62 f2 35 48 50 c1    	vpdpbusd %zmm1,%zmm9,%zmm0
  10c67e:	62 b1 fd 48 6f cf    	vmovdqa64 %zmm23,%zmm1
  10c684:	62 b1 4d 42 f8 cf    	vpsubb %zmm23,%zmm22,%zmm1{%k2}
  10c68a:	62 f2 65 48 50 c1    	vpdpbusd %zmm1,%zmm3,%zmm0
  10c690:	62 91 7d 48 6f dc    	vmovdqa32 %zmm28,%zmm3
  10c696:	62 01 7d 48 70 e4 4e 	vpshufd $0x4e,%zmm28,%zmm28
  10c69d:	62 21 7d 49 6f e3    	vmovdqa32 %zmm19,%zmm28{%k1}
  10c6a3:	62 71 0d 48 fe f0    	vpaddd %zmm0,%zmm14,%zmm14
  10c6a9:	c4 e2 49 8c 86 78 ff 	vpmaskmovd -0x88(%rsi),%xmm6,%xmm0
  10c6b0:	ff ff 
  10c6b2:	62 d1 7d 49 6f ee    	vmovdqa32 %zmm14,%zmm5{%k1}
  10c6b8:	62 b1 7d 49 70 db 4e 	vpshufd $0x4e,%zmm19,%zmm3{%k1}
  10c6bf:	62 f1 7c 48 5b db    	vcvtdq2ps %zmm3,%zmm3
  10c6c5:	62 01 7c 48 5b e4    	vcvtdq2ps %zmm28,%zmm28
  10c6cb:	c5 f9 70 c0 44       	vpshufd $0x44,%xmm0,%xmm0
  10c6d0:	c4 e3 7d 38 c0 01    	vinserti128 $0x1,%xmm0,%ymm0,%ymm0
  10c6d6:	62 f2 7d 48 13 c0    	vcvtph2ps %ymm0,%zmm0
  10c6dc:	62 d1 7d 49 70 d6 4e 	vpshufd $0x4e,%zmm14,%zmm2{%k1}
  10c6e3:	62 f1 7c 48 5b d2    	vcvtdq2ps %zmm2,%zmm2
  10c6e9:	62 f1 7c 48 5b ed    	vcvtdq2ps %zmm5,%zmm5
  10c6ef:	62 f3 7d 48 04 c8 00 	vpermilps $0x0,%zmm0,%zmm1
  10c6f6:	62 b1 74 48 59 cc    	vmulps %zmm20,%zmm1,%zmm1
  10c6fc:	62 f2 75 48 a8 5c 24 	vfmadd213ps 0x480(%rsp),%zmm1,%zmm3
  10c703:	12 
  10c704:	62 f1 7c 48 29 5c 24 	vmovaps %zmm3,0x480(%rsp)
  10c70b:	12 
  10c70c:	62 f3 7d 48 04 c8 55 	vpermilps $0x55,%zmm0,%zmm1
  10c713:	62 b1 74 48 59 cc    	vmulps %zmm20,%zmm1,%zmm1
  10c719:	62 62 75 48 a8 64 24 	vfmadd213ps 0x540(%rsp),%zmm1,%zmm28
  10c720:	15 
  10c721:	62 f3 7d 48 04 c8 aa 	vpermilps $0xaa,%zmm0,%zmm1
  10c728:	62 f3 7d 48 04 c0 ff 	vpermilps $0xff,%zmm0,%zmm0
  10c72f:	62 b1 74 48 59 cc    	vmulps %zmm20,%zmm1,%zmm1
  10c735:	62 61 7c 48 29 64 24 	vmovaps %zmm28,0x540(%rsp)
  10c73c:	15 
  10c73d:	62 f2 75 48 a8 54 24 	vfmadd213ps 0x4c0(%rsp),%zmm1,%zmm2
  10c744:	13 
  10c745:	62 b1 7c 48 59 c4    	vmulps %zmm20,%zmm0,%zmm0
  10c74b:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x4c0(%rsp)
  10c752:	13 
  10c753:	62 f2 7d 48 a8 6c 24 	vfmadd213ps 0x500(%rsp),%zmm0,%zmm5
  10c75a:	14 
  10c75b:	62 f1 7c 48 29 6c 24 	vmovaps %zmm5,0x500(%rsp)
  10c762:	14 
  10c763:	48 39 bc 24 e0 01 00 	cmp    %rdi,0x1e0(%rsp)
  10c76a:	00 
  10c76b:	0f 85 4f f7 ff ff    	jne    10bec0 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1170>
  10c771:	4c 8b ac 24 c0 01 00 	mov    0x1c0(%rsp),%r13
  10c778:	00 
  10c779:	4c 8b 94 24 a0 01 00 	mov    0x1a0(%rsp),%r10
  10c780:	00 
  10c781:	4c 8b 84 24 90 01 00 	mov    0x190(%rsp),%r8
  10c788:	00 
  10c789:	62 f1 7c 48 28 cb    	vmovaps %zmm3,%zmm1
  10c78f:	4c 8b 9c 24 88 01 00 	mov    0x188(%rsp),%r11
  10c796:	00 
  10c797:	4c 8b 8c 24 80 01 00 	mov    0x180(%rsp),%r9
  10c79e:	00 
  10c79f:	62 91 7c 48 28 ec    	vmovaps %zmm28,%zmm5
  10c7a5:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  10c7ac:	00 
  10c7ad:	62 d1 7c 48 11 09    	vmovups %zmm1,(%r9)
  10c7b3:	62 f1 7c 48 28 64 24 	vmovaps 0x500(%rsp),%zmm4
  10c7ba:	14 
  10c7bb:	49 83 c5 02          	add    $0x2,%r13
  10c7bf:	4d 01 c3             	add    %r8,%r11
  10c7c2:	4d 01 c2             	add    %r8,%r10
  10c7c5:	62 d1 7c 48 11 2c 81 	vmovups %zmm5,(%r9,%rax,4)
  10c7cc:	48 8b 84 24 58 01 00 	mov    0x158(%rsp),%rax
  10c7d3:	00 
  10c7d4:	62 d1 7c 48 11 14 81 	vmovups %zmm2,(%r9,%rax,4)
  10c7db:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  10c7e2:	00 
  10c7e3:	62 d1 7c 48 11 24 81 	vmovups %zmm4,(%r9,%rax,4)
  10c7ea:	48 8b 84 24 98 01 00 	mov    0x198(%rsp),%rax
  10c7f1:	00 
  10c7f2:	49 83 c1 40          	add    $0x40,%r9
  10c7f6:	49 39 c5             	cmp    %rax,%r13
  10c7f9:	0f 8c 41 f6 ff ff    	jl     10be40 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x10f0>
  10c7ff:	48 8b 94 24 38 01 00 	mov    0x138(%rsp),%rdx
  10c806:	00 
  10c807:	8b bc 24 30 01 00 00 	mov    0x130(%rsp),%edi
  10c80e:	48 8b 8c 24 28 01 00 	mov    0x128(%rsp),%rcx
  10c815:	00 
  10c816:	62 21 fd 48 6f c5    	vmovdqa64 %zmm21,%zmm24
  10c81c:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
  10c823:	00 
  10c824:	48 8b 84 24 48 01 00 	mov    0x148(%rsp),%rax
  10c82b:	00 
  10c82c:	48 8b 9c 24 50 01 00 	mov    0x150(%rsp),%rbx
  10c833:	00 
  10c834:	48 ff c1             	inc    %rcx
  10c837:	48 01 9c 24 70 01 00 	add    %rbx,0x170(%rsp)
  10c83e:	00 
  10c83f:	48 01 c6             	add    %rax,%rsi
  10c842:	48 8b 84 24 40 01 00 	mov    0x140(%rsp),%rax
  10c849:	00 
  10c84a:	48 01 c2             	add    %rax,%rdx
  10c84d:	48 8b 84 24 d0 00 00 	mov    0xd0(%rsp),%rax
  10c854:	00 
  10c855:	48 39 c1             	cmp    %rax,%rcx
  10c858:	0f 85 6d f5 ff ff    	jne    10bdcb <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x107b>
  10c85e:	44 8b 4d 10          	mov    0x10(%rbp),%r9d
  10c862:	62 e1 fd 28 6f 44 24 	vmovdqa64 0xe0(%rsp),%ymm16
  10c869:	07 
  10c86a:	c5 7a 7e 94 24 48 01 	vmovq  0x148(%rsp),%xmm10
  10c871:	00 00 
  10c873:	44 8b 94 24 10 01 00 	mov    0x110(%rsp),%r10d
  10c87a:	00 
  10c87b:	4c 8b b4 24 08 01 00 	mov    0x108(%rsp),%r14
  10c882:	00 
  10c883:	4c 8b bc 24 e0 01 00 	mov    0x1e0(%rsp),%r15
  10c88a:	00 
  10c88b:	41 39 f9             	cmp    %edi,%r9d
  10c88e:	0f 84 1c 17 00 00    	je     10dfb0 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x3260>
  10c894:	31 c9                	xor    %ecx,%ecx
  10c896:	41 83 fa 03          	cmp    $0x3,%r10d
  10c89a:	0f 8e 76 0c 00 00    	jle    10d516 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x27c6>
  10c8a0:	49 69 c7 88 00 00 00 	imul   $0x88,%r15,%rax
  10c8a7:	45 85 c9             	test   %r9d,%r9d
  10c8aa:	48 8b 9c 24 78 01 00 	mov    0x178(%rsp),%rbx
  10c8b1:	00 
  10c8b2:	c4 41 f9 7e d0       	vmovq  %xmm10,%r8
  10c8b7:	4c 8b a4 24 b8 00 00 	mov    0xb8(%rsp),%r12
  10c8be:	00 
  10c8bf:	c4 41 f9 7e d3       	vmovq  %xmm10,%r11
  10c8c4:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
  10c8c9:	4c 89 b4 24 88 00 00 	mov    %r14,0x88(%rsp)
  10c8d0:	00 
  10c8d1:	62 e1 fd 28 7f 04 24 	vmovdqa64 %ymm16,(%rsp)
  10c8d8:	62 b1 fd 28 6f e8    	vmovdqa64 %ymm16,%ymm5
  10c8de:	44 89 4d 10          	mov    %r9d,0x10(%rbp)
  10c8e2:	4c 89 bc 24 40 02 00 	mov    %r15,0x240(%rsp)
  10c8e9:	00 
  10c8ea:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  10c8ef:	41 8d 41 07          	lea    0x7(%r9),%eax
  10c8f3:	c5 79 d6 94 24 48 01 	vmovq  %xmm10,0x148(%rsp)
  10c8fa:	00 00 
  10c8fc:	41 0f 49 c1          	cmovns %r9d,%eax
  10c900:	4c 0f af d9          	imul   %rcx,%r11
  10c904:	c1 f8 03             	sar    $0x3,%eax
  10c907:	48 98                	cltq
  10c909:	49 89 da             	mov    %rbx,%r10
  10c90c:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
  10c913:	00 
  10c914:	4c 89 f8             	mov    %r15,%rax
  10c917:	48 0f af c1          	imul   %rcx,%rax
  10c91b:	49 c1 e2 04          	shl    $0x4,%r10
  10c91f:	48 69 c0 88 00 00 00 	imul   $0x88,%rax,%rax
  10c926:	49 8d 34 06          	lea    (%r14,%rax,1),%rsi
  10c92a:	48 8d 04 8d 00 00 00 	lea    0x0(,%rcx,4),%rax
  10c931:	00 
  10c932:	c4 c1 f9 7e f6       	vmovq  %xmm6,%r14
  10c937:	48 8d 50 05          	lea    0x5(%rax),%rdx
  10c93b:	48 8d 78 01          	lea    0x1(%rax),%rdi
  10c93f:	48 0f af d3          	imul   %rbx,%rdx
  10c943:	48 0f af fb          	imul   %rbx,%rdi
  10c947:	48 89 94 24 60 01 00 	mov    %rdx,0x160(%rsp)
  10c94e:	00 
  10c94f:	48 8d 51 02          	lea    0x2(%rcx),%rdx
  10c953:	49 0f af d0          	imul   %r8,%rdx
  10c957:	4c 8b 84 24 98 01 00 	mov    0x198(%rsp),%r8
  10c95e:	00 
  10c95f:	48 89 94 24 68 01 00 	mov    %rdx,0x168(%rsp)
  10c966:	00 
  10c967:	48 8d 50 09          	lea    0x9(%rax),%rdx
  10c96b:	48 0f af d3          	imul   %rbx,%rdx
  10c96f:	48 89 94 24 70 01 00 	mov    %rdx,0x170(%rsp)
  10c976:	00 
  10c977:	48 8d 50 0d          	lea    0xd(%rax),%rdx
  10c97b:	48 0f af d3          	imul   %rbx,%rdx
  10c97f:	48 89 94 24 80 01 00 	mov    %rdx,0x180(%rsp)
  10c986:	00 
  10c987:	48 89 da             	mov    %rbx,%rdx
  10c98a:	48 c1 e2 06          	shl    $0x6,%rdx
  10c98e:	48 89 94 24 c0 00 00 	mov    %rdx,0xc0(%rsp)
  10c995:	00 
  10c996:	48 89 da             	mov    %rbx,%rdx
  10c999:	48 0f af d1          	imul   %rcx,%rdx
  10c99d:	4a 8d 14 42          	lea    (%rdx,%r8,2),%rdx
  10c9a1:	48 c1 e2 04          	shl    $0x4,%rdx
  10c9a5:	4c 01 e2             	add    %r12,%rdx
  10c9a8:	48 89 94 24 c8 00 00 	mov    %rdx,0xc8(%rsp)
  10c9af:	00 
  10c9b0:	48 8d 50 03          	lea    0x3(%rax),%rdx
  10c9b4:	48 0f af d3          	imul   %rbx,%rdx
  10c9b8:	4c 29 da             	sub    %r11,%rdx
  10c9bb:	48 89 94 24 e0 00 00 	mov    %rdx,0xe0(%rsp)
  10c9c2:	00 
  10c9c3:	48 8d 50 07          	lea    0x7(%rax),%rdx
  10c9c7:	48 0f af d3          	imul   %rbx,%rdx
  10c9cb:	4c 29 da             	sub    %r11,%rdx
  10c9ce:	48 89 94 24 08 01 00 	mov    %rdx,0x108(%rsp)
  10c9d5:	00 
  10c9d6:	48 8d 50 0b          	lea    0xb(%rax),%rdx
  10c9da:	48 83 c0 0f          	add    $0xf,%rax
  10c9de:	48 0f af c3          	imul   %rbx,%rax
  10c9e2:	48 0f af d3          	imul   %rbx,%rdx
  10c9e6:	48 8b 9c 24 80 05 00 	mov    0x580(%rsp),%rbx
  10c9ed:	00 
  10c9ee:	4c 29 d8             	sub    %r11,%rax
  10c9f1:	49 89 c5             	mov    %rax,%r13
  10c9f4:	4c 89 c0             	mov    %r8,%rax
  10c9f7:	4c 29 da             	sub    %r11,%rdx
  10c9fa:	4d 89 d8             	mov    %r11,%r8
  10c9fd:	49 0f af c7          	imul   %r15,%rax
  10ca01:	48 89 94 24 10 01 00 	mov    %rdx,0x110(%rsp)
  10ca08:	00 
  10ca09:	4c 89 ac 24 00 01 00 	mov    %r13,0x100(%rsp)
  10ca10:	00 
  10ca11:	48 69 c0 88 00 00 00 	imul   $0x88,%rax,%rax
  10ca18:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
  10ca1f:	00 
  10ca20:	48 01 d8             	add    %rbx,%rax
  10ca23:	48 89 cb             	mov    %rcx,%rbx
  10ca26:	48 89 f9             	mov    %rdi,%rcx
  10ca29:	48 f7 d8             	neg    %rax
  10ca2c:	48 89 df             	mov    %rbx,%rdi
  10ca2f:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
  10ca36:	00 
  10ca37:	48 8d 84 24 a0 05 00 	lea    0x5a0(%rsp),%rax
  10ca3e:	00 
  10ca3f:	48 89 84 24 00 02 00 	mov    %rax,0x200(%rsp)
  10ca46:	00 
  10ca47:	48 b8 0f 0f 0f 0f 0f 	movabs $0xf0f0f0f0f0f0f0f,%rax
  10ca4e:	0f 0f 0f 
  10ca51:	62 f2 fd 28 7c f8    	vpbroadcastq %rax,%ymm7
  10ca57:	c5 fd 7f bc 24 e0 01 	vmovdqa %ymm7,0x1e0(%rsp)
  10ca5e:	00 00 
  10ca60:	49 8d 04 36          	lea    (%r14,%rsi,1),%rax
  10ca64:	48 89 b4 24 a0 05 00 	mov    %rsi,0x5a0(%rsp)
  10ca6b:	00 
  10ca6c:	48 8b b4 24 98 01 00 	mov    0x198(%rsp),%rsi
  10ca73:	00 
  10ca74:	48 89 84 24 a8 05 00 	mov    %rax,0x5a8(%rsp)
  10ca7b:	00 
  10ca7c:	4c 01 f0             	add    %r14,%rax
  10ca7f:	48 89 84 24 b0 05 00 	mov    %rax,0x5b0(%rsp)
  10ca86:	00 
  10ca87:	4c 01 f0             	add    %r14,%rax
  10ca8a:	48 89 84 24 b8 05 00 	mov    %rax,0x5b8(%rsp)
  10ca91:	00 
  10ca92:	48 39 b4 24 58 01 00 	cmp    %rsi,0x158(%rsp)
  10ca99:	00 
  10ca9a:	0f 8e ea 09 00 00    	jle    10d48a <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x273a>
  10caa0:	48 8b 9c 24 78 01 00 	mov    0x178(%rsp),%rbx
  10caa7:	00 
  10caa8:	48 8b 94 24 48 01 00 	mov    0x148(%rsp),%rdx
  10caaf:	00 
  10cab0:	4c 8b ac 24 90 00 00 	mov    0x90(%rsp),%r13
  10cab7:	00 
  10cab8:	48 89 bc 24 b0 00 00 	mov    %rdi,0xb0(%rsp)
  10cabf:	00 
  10cac0:	4c 8b 9c 24 98 00 00 	mov    0x98(%rsp),%r11
  10cac7:	00 
  10cac8:	48 89 8c 24 a0 00 00 	mov    %rcx,0xa0(%rsp)
  10cacf:	00 
  10cad0:	4c 8d 8c 24 c0 07 00 	lea    0x7c0(%rsp),%r9
  10cad7:	00 
  10cad8:	62 01 3d 00 ef c0    	vpxord %xmm24,%xmm24,%xmm24
  10cade:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
  10cae5:	00 
  10cae6:	4c 89 54 24 40       	mov    %r10,0x40(%rsp)
  10caeb:	4c 89 b4 24 50 01 00 	mov    %r14,0x150(%rsp)
  10caf2:	00 
  10caf3:	48 8d 34 0b          	lea    (%rbx,%rcx,1),%rsi
  10caf7:	4c 89 ef             	mov    %r13,%rdi
  10cafa:	48 89 b4 24 40 01 00 	mov    %rsi,0x140(%rsp)
  10cb01:	00 
  10cb02:	48 8b b4 24 60 01 00 	mov    0x160(%rsp),%rsi
  10cb09:	00 
  10cb0a:	48 01 de             	add    %rbx,%rsi
  10cb0d:	48 89 b4 24 38 01 00 	mov    %rsi,0x138(%rsp)
  10cb14:	00 
  10cb15:	48 8b b4 24 70 01 00 	mov    0x170(%rsp),%rsi
  10cb1c:	00 
  10cb1d:	48 01 de             	add    %rbx,%rsi
  10cb20:	48 89 b4 24 30 01 00 	mov    %rsi,0x130(%rsp)
  10cb27:	00 
  10cb28:	48 8b b4 24 68 01 00 	mov    0x168(%rsp),%rsi
  10cb2f:	00 
  10cb30:	48 01 d6             	add    %rdx,%rsi
  10cb33:	48 8b 94 24 c8 00 00 	mov    0xc8(%rsp),%rdx
  10cb3a:	00 
  10cb3b:	48 89 b4 24 28 01 00 	mov    %rsi,0x128(%rsp)
  10cb42:	00 
  10cb43:	48 8b b4 24 80 01 00 	mov    0x180(%rsp),%rsi
  10cb4a:	00 
  10cb4b:	48 01 f3             	add    %rsi,%rbx
  10cb4e:	48 8b b4 24 98 01 00 	mov    0x198(%rsp),%rsi
  10cb55:	00 
  10cb56:	48 89 9c 24 20 01 00 	mov    %rbx,0x120(%rsp)
  10cb5d:	00 
  10cb5e:	48 8d 9c 24 c0 05 00 	lea    0x5c0(%rsp),%rbx
  10cb65:	00 
  10cb66:	48 89 9c 24 80 02 00 	mov    %rbx,0x280(%rsp)
  10cb6d:	00 
  10cb6e:	48 89 cb             	mov    %rcx,%rbx
  10cb71:	4c 29 c3             	sub    %r8,%rbx
  10cb74:	48 89 9c 24 18 01 00 	mov    %rbx,0x118(%rsp)
  10cb7b:	00 
  10cb7c:	48 89 f1             	mov    %rsi,%rcx
  10cb7f:	90                   	nop
  10cb80:	83 bc 24 8c 05 00 00 	cmpl   $0x1f,0x58c(%rsp)
  10cb87:	1f 
  10cb88:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  10cb8c:	c5 fc 29 8c 24 c0 05 	vmovaps %ymm1,0x5c0(%rsp)
  10cb93:	00 00 
  10cb95:	c5 fc 29 8c 24 e0 05 	vmovaps %ymm1,0x5e0(%rsp)
  10cb9c:	00 00 
  10cb9e:	c5 fc 29 8c 24 00 06 	vmovaps %ymm1,0x600(%rsp)
  10cba5:	00 00 
  10cba7:	c5 fc 29 8c 24 20 06 	vmovaps %ymm1,0x620(%rsp)
  10cbae:	00 00 
  10cbb0:	c5 fc 29 8c 24 40 06 	vmovaps %ymm1,0x640(%rsp)
  10cbb7:	00 00 
  10cbb9:	c5 fc 29 8c 24 60 06 	vmovaps %ymm1,0x660(%rsp)
  10cbc0:	00 00 
  10cbc2:	c5 fc 29 8c 24 80 06 	vmovaps %ymm1,0x680(%rsp)
  10cbc9:	00 00 
  10cbcb:	c5 fc 29 8c 24 a0 06 	vmovaps %ymm1,0x6a0(%rsp)
  10cbd2:	00 00 
  10cbd4:	c5 fc 29 8c 24 c0 06 	vmovaps %ymm1,0x6c0(%rsp)
  10cbdb:	00 00 
  10cbdd:	c5 fc 29 8c 24 e0 06 	vmovaps %ymm1,0x6e0(%rsp)
  10cbe4:	00 00 
  10cbe6:	c5 fc 29 8c 24 00 07 	vmovaps %ymm1,0x700(%rsp)
  10cbed:	00 00 
  10cbef:	c5 fc 29 8c 24 20 07 	vmovaps %ymm1,0x720(%rsp)
  10cbf6:	00 00 
  10cbf8:	c5 fc 29 8c 24 40 07 	vmovaps %ymm1,0x740(%rsp)
  10cbff:	00 00 
  10cc01:	c5 fc 29 8c 24 60 07 	vmovaps %ymm1,0x760(%rsp)
  10cc08:	00 00 
  10cc0a:	c5 fc 29 8c 24 80 07 	vmovaps %ymm1,0x780(%rsp)
  10cc11:	00 00 
  10cc13:	c5 fc 29 8c 24 a0 07 	vmovaps %ymm1,0x7a0(%rsp)
  10cc1a:	00 00 
  10cc1c:	0f 8e a1 12 00 00    	jle    10dec3 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x3173>
  10cc22:	48 8b 84 24 80 05 00 	mov    0x580(%rsp),%rax
  10cc29:	00 
  10cc2a:	4c 8b 15 ef f2 03 00 	mov    0x3f2ef(%rip),%r10        # 14bf20 <ggml_table_f32_e8m0_half@@Base-0x1420>
  10cc31:	45 31 e4             	xor    %r12d,%r12d
  10cc34:	c5 fd 6f f5          	vmovdqa %ymm5,%ymm6
  10cc38:	48 89 8c 24 c0 01 00 	mov    %rcx,0x1c0(%rsp)
  10cc3f:	00 
  10cc40:	4c 89 84 24 a0 01 00 	mov    %r8,0x1a0(%rsp)
  10cc47:	00 
  10cc48:	48 89 94 24 90 01 00 	mov    %rdx,0x190(%rsp)
  10cc4f:	00 
  10cc50:	48 89 bc 24 88 01 00 	mov    %rdi,0x188(%rsp)
  10cc57:	00 
  10cc58:	48 8d 34 38          	lea    (%rax,%rdi,1),%rsi
  10cc5c:	0f 1f 40 00          	nopl   0x0(%rax)
  10cc60:	c5 fd 6f 0d f8 23 02 	vmovdqa 0x223f8(%rip),%ymm1        # 12f060 <_ZL11iq2xxs_grid+0xb20>
  10cc67:	00 
  10cc68:	c5 fe 6f 6e 08       	vmovdqu 0x8(%rsi),%ymm5
  10cc6d:	c4 e2 75 36 5e 28    	vpermd 0x28(%rsi),%ymm1,%ymm3
  10cc73:	4e 8d 34 1e          	lea    (%rsi,%r11,1),%r14
  10cc77:	0f b6 7e 04          	movzbl 0x4(%rsi),%edi
  10cc7b:	0f b6 56 05          	movzbl 0x5(%rsi),%edx
  10cc7f:	0f b6 5e 06          	movzbl 0x6(%rsi),%ebx
  10cc83:	44 0f b6 46 07       	movzbl 0x7(%rsi),%r8d
  10cc88:	c4 e2 75 36 56 68    	vpermd 0x68(%rsi),%ymm1,%ymm2
  10cc8e:	0f b6 46 01          	movzbl 0x1(%rsi),%eax
  10cc92:	0f b6 0e             	movzbl (%rsi),%ecx
  10cc95:	44 0f b6 7e 02       	movzbl 0x2(%rsi),%r15d
  10cc9a:	44 0f b6 6e 03       	movzbl 0x3(%rsi),%r13d
  10cc9f:	c5 fd 6f bc 24 e0 01 	vmovdqa 0x1e0(%rsp),%ymm7
  10cca6:	00 00 
  10cca8:	c4 e3 55 02 db f0    	vpblendd $0xf0,%ymm3,%ymm5,%ymm3
  10ccae:	c4 e2 75 36 c5       	vpermd %ymm5,%ymm1,%ymm0
  10ccb3:	c4 41 7a 10 24 9a    	vmovss (%r10,%rbx,4),%xmm12
  10ccb9:	c5 fe 6f 6e 48       	vmovdqu 0x48(%rsi),%ymm5
  10ccbe:	c4 03 19 21 24 82 10 	vinsertps $0x10,(%r10,%r8,4),%xmm12,%xmm12
  10ccc5:	c4 41 7a 10 1c ba    	vmovss (%r10,%rdi,4),%xmm11
  10cccb:	c4 43 21 21 1c 92 10 	vinsertps $0x10,(%r10,%rdx,4),%xmm11,%xmm11
  10ccd2:	c4 e3 7d 02 46 28 f0 	vpblendd $0xf0,0x28(%rsi),%ymm0,%ymm0
  10ccd9:	c4 01 7a 10 2c ba    	vmovss (%r10,%r15,4),%xmm13
  10ccdf:	c4 03 11 21 2c aa 10 	vinsertps $0x10,(%r10,%r13,4),%xmm13,%xmm13
  10cce6:	c5 65 db c7          	vpand  %ymm7,%ymm3,%ymm8
  10ccea:	c5 fd db e7          	vpand  %ymm7,%ymm0,%ymm4
  10ccee:	c5 e5 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm3
  10ccf3:	c5 fd 71 d0 04       	vpsrlw $0x4,%ymm0,%ymm0
  10ccf8:	c5 e5 db df          	vpand  %ymm7,%ymm3,%ymm3
  10ccfc:	c5 fd db c7          	vpand  %ymm7,%ymm0,%ymm0
  10cd00:	c4 42 4d 00 c0       	vpshufb %ymm8,%ymm6,%ymm8
  10cd05:	c4 e2 4d 00 e4       	vpshufb %ymm4,%ymm6,%ymm4
  10cd0a:	c4 e2 75 36 cd       	vpermd %ymm5,%ymm1,%ymm1
  10cd0f:	c4 e3 55 02 d2 f0    	vpblendd $0xf0,%ymm2,%ymm5,%ymm2
  10cd15:	c4 e2 4d 00 db       	vpshufb %ymm3,%ymm6,%ymm3
  10cd1a:	c4 e2 4d 00 c0       	vpshufb %ymm0,%ymm6,%ymm0
  10cd1f:	c4 41 20 16 dc       	vmovlhps %xmm12,%xmm11,%xmm11
  10cd24:	c4 41 7a 10 24 8a    	vmovss (%r10,%rcx,4),%xmm12
  10cd2a:	c4 43 19 21 24 82 10 	vinsertps $0x10,(%r10,%rax,4),%xmm12,%xmm12
  10cd31:	c4 e3 75 02 4e 68 f0 	vpblendd $0xf0,0x68(%rsi),%ymm1,%ymm1
  10cd38:	c5 6d db cf          	vpand  %ymm7,%ymm2,%ymm9
  10cd3c:	c5 f5 db ef          	vpand  %ymm7,%ymm1,%ymm5
  10cd40:	c5 ed 71 d2 04       	vpsrlw $0x4,%ymm2,%ymm2
  10cd45:	c5 f5 71 d1 04       	vpsrlw $0x4,%ymm1,%ymm1
  10cd4a:	c5 ed db d7          	vpand  %ymm7,%ymm2,%ymm2
  10cd4e:	c5 f5 db cf          	vpand  %ymm7,%ymm1,%ymm1
  10cd52:	c4 42 4d 00 c9       	vpshufb %ymm9,%ymm6,%ymm9
  10cd57:	c4 e2 4d 00 ed       	vpshufb %ymm5,%ymm6,%ymm5
  10cd5c:	c4 e2 4d 00 d2       	vpshufb %ymm2,%ymm6,%ymm2
  10cd61:	c4 e2 4d 00 c9       	vpshufb %ymm1,%ymm6,%ymm1
  10cd66:	c5 fd 70 fa dd       	vpshufd $0xdd,%ymm2,%ymm7
  10cd6b:	62 e1 7d 28 70 f4 88 	vpshufd $0x88,%ymm4,%ymm22
  10cd72:	c5 fd 7f bc 24 00 04 	vmovdqa %ymm7,0x400(%rsp)
  10cd79:	00 00 
  10cd7b:	62 61 7d 28 70 f2 88 	vpshufd $0x88,%ymm2,%ymm30
  10cd82:	62 61 7d 28 70 eb 88 	vpshufd $0x88,%ymm3,%ymm29
  10cd89:	c5 fd 70 d3 dd       	vpshufd $0xdd,%ymm3,%ymm2
  10cd8e:	62 61 7d 28 70 d1 88 	vpshufd $0x88,%ymm1,%ymm26
  10cd95:	c5 fd 7f 94 24 c0 03 	vmovdqa %ymm2,0x3c0(%rsp)
  10cd9c:	00 00 
  10cd9e:	c5 fd 70 c9 dd       	vpshufd $0xdd,%ymm1,%ymm1
  10cda3:	c4 c1 7d 70 f9 dd    	vpshufd $0xdd,%ymm9,%ymm7
  10cda9:	c5 fd 7f 8c 24 c0 04 	vmovdqa %ymm1,0x4c0(%rsp)
  10cdb0:	00 00 
  10cdb2:	c4 c1 7d 70 d0 dd    	vpshufd $0xdd,%ymm8,%ymm2
  10cdb8:	c5 fd 70 c8 dd       	vpshufd $0xdd,%ymm0,%ymm1
  10cdbd:	c5 fd 70 dd dd       	vpshufd $0xdd,%ymm5,%ymm3
  10cdc2:	c5 fd 7f bc 24 40 05 	vmovdqa %ymm7,0x540(%rsp)
  10cdc9:	00 00 
  10cdcb:	c5 fd 70 e4 dd       	vpshufd $0xdd,%ymm4,%ymm4
  10cdd0:	62 c1 1c 08 16 cd    	vmovlhps %xmm13,%xmm12,%xmm17
  10cdd6:	62 41 7d 28 70 e1 88 	vpshufd $0x88,%ymm9,%ymm28
  10cddd:	c5 fd 7f 94 24 00 05 	vmovdqa %ymm2,0x500(%rsp)
  10cde4:	00 00 
  10cde6:	62 c3 75 20 18 cb 01 	vinsertf32x4 $0x1,%xmm11,%ymm17,%ymm17
  10cded:	62 41 7d 28 70 d8 88 	vpshufd $0x88,%ymm8,%ymm27
  10cdf4:	62 61 7d 28 70 c8 88 	vpshufd $0x88,%ymm0,%ymm25
  10cdfb:	c5 fd 7f 8c 24 80 03 	vmovdqa %ymm1,0x380(%rsp)
  10ce02:	00 00 
  10ce04:	62 e1 7d 28 70 fd 88 	vpshufd $0x88,%ymm5,%ymm23
  10ce0b:	c5 fd 7f 9c 24 40 03 	vmovdqa %ymm3,0x340(%rsp)
  10ce12:	00 00 
  10ce14:	c5 fd 7f a4 24 00 03 	vmovdqa %ymm4,0x300(%rsp)
  10ce1b:	00 00 
  10ce1d:	4c 8b ac 24 00 02 00 	mov    0x200(%rsp),%r13
  10ce24:	00 
  10ce25:	48 8b 84 24 80 02 00 	mov    0x280(%rsp),%rax
  10ce2c:	00 
  10ce2d:	62 e1 fd 28 7f 74 24 	vmovdqa64 %ymm22,0x480(%rsp)
  10ce34:	24 
  10ce35:	c5 fd 7f b4 24 c0 02 	vmovdqa %ymm6,0x2c0(%rsp)
  10ce3c:	00 00 
  10ce3e:	49 8b 55 00          	mov    0x0(%r13),%rdx
  10ce42:	62 81 7d 28 6f c0    	vmovdqa32 %ymm24,%ymm16
  10ce48:	62 01 7d 28 6f f8    	vmovdqa32 %ymm24,%ymm31
  10ce4e:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  10ce52:	49 83 c5 08          	add    $0x8,%r13
  10ce56:	4c 01 f2             	add    %r14,%rdx
  10ce59:	c5 fe 6f 42 08       	vmovdqu 0x8(%rdx),%ymm0
  10ce5e:	c5 fe 6f 5a 68       	vmovdqu 0x68(%rdx),%ymm3
  10ce63:	c5 fe 6f 52 48       	vmovdqu 0x48(%rdx),%ymm2
  10ce68:	c5 fe 6f 4a 28       	vmovdqu 0x28(%rdx),%ymm1
  10ce6d:	c4 e3 7d 46 e0 00    	vperm2i128 $0x0,%ymm0,%ymm0,%ymm4
  10ce73:	c4 e3 65 46 fb 00    	vperm2i128 $0x0,%ymm3,%ymm3,%ymm7
  10ce79:	c4 e3 7d 46 c0 11    	vperm2i128 $0x11,%ymm0,%ymm0,%ymm0
  10ce7f:	c4 e3 65 46 db 11    	vperm2i128 $0x11,%ymm3,%ymm3,%ymm3
  10ce85:	c5 7d 70 c0 a0       	vpshufd $0xa0,%ymm0,%ymm8
  10ce8a:	c5 7d 70 f7 a0       	vpshufd $0xa0,%ymm7,%ymm14
  10ce8f:	c5 fd 70 c0 f5       	vpshufd $0xf5,%ymm0,%ymm0
  10ce94:	c5 fd 7f 84 24 40 04 	vmovdqa %ymm0,0x440(%rsp)
  10ce9b:	00 00 
  10ce9d:	c4 c2 0d 08 c6       	vpsignb %ymm14,%ymm14,%ymm0
  10cea2:	62 e1 fd 28 6f e8    	vmovdqa64 %ymm0,%ymm21
  10cea8:	62 91 fd 28 6f c6    	vmovdqa64 %ymm30,%ymm0
  10ceae:	c4 e3 6d 46 f2 00    	vperm2i128 $0x0,%ymm2,%ymm2,%ymm6
  10ceb4:	c4 c2 7d 08 c6       	vpsignb %ymm14,%ymm0,%ymm0
  10ceb9:	62 e2 55 20 50 c0    	vpdpbusd %ymm0,%ymm21,%ymm16
  10cebf:	c5 7d 70 d6 a0       	vpshufd $0xa0,%ymm6,%ymm10
  10cec4:	c4 c2 2d 08 c2       	vpsignb %ymm10,%ymm10,%ymm0
  10cec9:	62 e1 fd 28 6f e0    	vmovdqa64 %ymm0,%ymm20
  10cecf:	62 91 fd 28 6f c5    	vmovdqa64 %ymm29,%ymm0
  10ced5:	c4 e3 75 46 e9 00    	vperm2i128 $0x0,%ymm1,%ymm1,%ymm5
  10cedb:	c5 7d 70 ec a0       	vpshufd $0xa0,%ymm4,%ymm13
  10cee0:	c4 c2 7d 08 c2       	vpsignb %ymm10,%ymm0,%ymm0
  10cee5:	c5 7d 70 fd a0       	vpshufd $0xa0,%ymm5,%ymm15
  10ceea:	c5 7d 70 e3 a0       	vpshufd $0xa0,%ymm3,%ymm12
  10ceef:	c4 e3 6d 46 d2 11    	vperm2i128 $0x11,%ymm2,%ymm2,%ymm2
  10cef5:	c5 7d 70 da a0       	vpshufd $0xa0,%ymm2,%ymm11
  10cefa:	c4 e3 75 46 c9 11    	vperm2i128 $0x11,%ymm1,%ymm1,%ymm1
  10cf00:	c5 fd 70 ff f5       	vpshufd $0xf5,%ymm7,%ymm7
  10cf05:	c5 fd 70 f6 f5       	vpshufd $0xf5,%ymm6,%ymm6
  10cf0a:	c5 7d 70 c9 a0       	vpshufd $0xa0,%ymm1,%ymm9
  10cf0f:	c5 fd 70 ed f5       	vpshufd $0xf5,%ymm5,%ymm5
  10cf14:	c5 fd 70 e4 f5       	vpshufd $0xf5,%ymm4,%ymm4
  10cf19:	c5 fd 70 db f5       	vpshufd $0xf5,%ymm3,%ymm3
  10cf1e:	62 e2 5d 20 50 c0    	vpdpbusd %ymm0,%ymm20,%ymm16
  10cf24:	c4 c2 05 08 c7       	vpsignb %ymm15,%ymm15,%ymm0
  10cf29:	c5 fd 70 d2 f5       	vpshufd $0xf5,%ymm2,%ymm2
  10cf2e:	c5 fd 70 c9 f5       	vpshufd $0xf5,%ymm1,%ymm1
  10cf33:	62 e1 fd 28 6f d8    	vmovdqa64 %ymm0,%ymm19
  10cf39:	62 91 fd 28 6f c4    	vmovdqa64 %ymm28,%ymm0
  10cf3f:	c4 c2 7d 08 c7       	vpsignb %ymm15,%ymm0,%ymm0
  10cf44:	62 e2 65 20 50 c0    	vpdpbusd %ymm0,%ymm19,%ymm16
  10cf4a:	c4 c2 15 08 c5       	vpsignb %ymm13,%ymm13,%ymm0
  10cf4f:	62 e1 fd 28 6f d0    	vmovdqa64 %ymm0,%ymm18
  10cf55:	62 91 fd 28 6f c3    	vmovdqa64 %ymm27,%ymm0
  10cf5b:	c4 c2 7d 08 c5       	vpsignb %ymm13,%ymm0,%ymm0
  10cf60:	62 a1 7d 28 6f f0    	vmovdqa32 %ymm16,%ymm22
  10cf66:	62 e2 6d 20 50 f0    	vpdpbusd %ymm0,%ymm18,%ymm22
  10cf6c:	62 91 fd 28 6f c2    	vmovdqa64 %ymm26,%ymm0
  10cf72:	c4 42 7d 08 f6       	vpsignb %ymm14,%ymm0,%ymm14
  10cf77:	62 91 fd 28 6f c1    	vmovdqa64 %ymm25,%ymm0
  10cf7d:	c4 42 7d 08 d2       	vpsignb %ymm10,%ymm0,%ymm10
  10cf82:	62 b1 fd 28 6f c7    	vmovdqa64 %ymm23,%ymm0
  10cf88:	c4 42 7d 08 ff       	vpsignb %ymm15,%ymm0,%ymm15
  10cf8d:	c4 c2 1d 08 c4       	vpsignb %ymm12,%ymm12,%ymm0
  10cf92:	62 42 55 20 50 fe    	vpdpbusd %ymm14,%ymm21,%ymm31
  10cf98:	62 42 5d 20 50 fa    	vpdpbusd %ymm10,%ymm20,%ymm31
  10cf9e:	c4 42 35 08 d1       	vpsignb %ymm9,%ymm9,%ymm10
  10cfa3:	c5 7d 6f b4 24 80 04 	vmovdqa 0x480(%rsp),%ymm14
  10cfaa:	00 00 
  10cfac:	62 42 65 20 50 ff    	vpdpbusd %ymm15,%ymm19,%ymm31
  10cfb2:	62 e1 fd 28 6f d8    	vmovdqa64 %ymm0,%ymm19
  10cfb8:	62 91 fd 28 6f c6    	vmovdqa64 %ymm30,%ymm0
  10cfbe:	c4 42 7d 08 fc       	vpsignb %ymm12,%ymm0,%ymm15
  10cfc3:	62 91 fd 28 6f c5    	vmovdqa64 %ymm29,%ymm0
  10cfc9:	c4 42 0d 08 ed       	vpsignb %ymm13,%ymm14,%ymm13
  10cfce:	62 42 6d 20 50 fd    	vpdpbusd %ymm13,%ymm18,%ymm31
  10cfd4:	c4 42 25 08 f3       	vpsignb %ymm11,%ymm11,%ymm14
  10cfd9:	62 11 7d 28 6f e8    	vmovdqa32 %ymm24,%ymm13
  10cfdf:	62 52 65 20 50 ef    	vpdpbusd %ymm15,%ymm19,%ymm13
  10cfe5:	c4 42 7d 08 fb       	vpsignb %ymm11,%ymm0,%ymm15
  10cfea:	62 91 fd 28 6f c4    	vmovdqa64 %ymm28,%ymm0
  10cff0:	c4 42 0d 50 ef       	{vex} vpdpbusd %ymm15,%ymm14,%ymm13
  10cff5:	c4 42 7d 08 f9       	vpsignb %ymm9,%ymm0,%ymm15
  10cffa:	62 91 fd 28 6f c3    	vmovdqa64 %ymm27,%ymm0
  10d000:	c4 c2 7d 08 c0       	vpsignb %ymm8,%ymm0,%ymm0
  10d005:	c4 42 2d 50 ef       	{vex} vpdpbusd %ymm15,%ymm10,%ymm13
  10d00a:	c4 42 3d 08 f8       	vpsignb %ymm8,%ymm8,%ymm15
  10d00f:	c4 62 05 50 e8       	{vex} vpdpbusd %ymm0,%ymm15,%ymm13
  10d014:	62 91 fd 28 6f c2    	vmovdqa64 %ymm26,%ymm0
  10d01a:	c4 42 7d 08 e4       	vpsignb %ymm12,%ymm0,%ymm12
  10d01f:	62 91 fd 28 6f c1    	vmovdqa64 %ymm25,%ymm0
  10d025:	c4 42 7d 08 db       	vpsignb %ymm11,%ymm0,%ymm11
  10d02a:	62 b1 fd 28 6f c7    	vmovdqa64 %ymm23,%ymm0
  10d030:	c4 42 7d 08 c9       	vpsignb %ymm9,%ymm0,%ymm9
  10d035:	62 c1 7d 28 6f ed    	vmovdqa32 %ymm13,%ymm21
  10d03b:	62 11 7d 28 6f e8    	vmovdqa32 %ymm24,%ymm13
  10d041:	c5 fd 6f 84 24 00 03 	vmovdqa 0x300(%rsp),%ymm0
  10d048:	00 00 
  10d04a:	62 52 65 20 50 ec    	vpdpbusd %ymm12,%ymm19,%ymm13
  10d050:	c4 62 55 08 e5       	vpsignb %ymm5,%ymm5,%ymm12
  10d055:	c4 42 0d 50 eb       	{vex} vpdpbusd %ymm11,%ymm14,%ymm13
  10d05a:	c5 7d 6f b4 24 80 04 	vmovdqa 0x480(%rsp),%ymm14
  10d061:	00 00 
  10d063:	c5 7d 6f 9c 24 40 05 	vmovdqa 0x540(%rsp),%ymm11
  10d06a:	00 00 
  10d06c:	c4 42 2d 50 e9       	{vex} vpdpbusd %ymm9,%ymm10,%ymm13
  10d071:	c5 7d 6f 94 24 00 04 	vmovdqa 0x400(%rsp),%ymm10
  10d078:	00 00 
  10d07a:	c4 42 0d 08 c0       	vpsignb %ymm8,%ymm14,%ymm8
  10d07f:	c5 7d 6f b4 24 c0 03 	vmovdqa 0x3c0(%rsp),%ymm14
  10d086:	00 00 
  10d088:	c4 42 05 50 e8       	{vex} vpdpbusd %ymm8,%ymm15,%ymm13
  10d08d:	c4 62 45 08 ff       	vpsignb %ymm7,%ymm7,%ymm15
  10d092:	62 11 7d 28 6f c0    	vmovdqa32 %ymm24,%ymm8
  10d098:	c4 62 2d 08 cf       	vpsignb %ymm7,%ymm10,%ymm9
  10d09d:	c4 42 05 50 c1       	{vex} vpdpbusd %ymm9,%ymm15,%ymm8
  10d0a2:	c4 62 0d 08 ce       	vpsignb %ymm6,%ymm14,%ymm9
  10d0a7:	62 c1 7d 28 6f c5    	vmovdqa32 %ymm13,%ymm16
  10d0ad:	c4 62 4d 08 ee       	vpsignb %ymm6,%ymm6,%ymm13
  10d0b2:	c4 42 15 50 c1       	{vex} vpdpbusd %ymm9,%ymm13,%ymm8
  10d0b7:	c4 62 25 08 cd       	vpsignb %ymm5,%ymm11,%ymm9
  10d0bc:	c4 62 5d 08 dc       	vpsignb %ymm4,%ymm4,%ymm11
  10d0c1:	c4 42 1d 50 c1       	{vex} vpdpbusd %ymm9,%ymm12,%ymm8
  10d0c6:	c5 7d 6f 8c 24 00 05 	vmovdqa 0x500(%rsp),%ymm9
  10d0cd:	00 00 
  10d0cf:	c4 62 35 08 cc       	vpsignb %ymm4,%ymm9,%ymm9
  10d0d4:	c4 42 25 50 c1       	{vex} vpdpbusd %ymm9,%ymm11,%ymm8
  10d0d9:	c4 e2 7d 08 e4       	vpsignb %ymm4,%ymm0,%ymm4
  10d0de:	c5 7d 6f 8c 24 c0 04 	vmovdqa 0x4c0(%rsp),%ymm9
  10d0e5:	00 00 
  10d0e7:	62 51 4d 20 fe c0    	vpaddd %ymm8,%ymm22,%ymm8
  10d0ed:	c4 e2 35 08 ff       	vpsignb %ymm7,%ymm9,%ymm7
  10d0f2:	62 11 7d 28 6f c8    	vmovdqa32 %ymm24,%ymm9
  10d0f8:	c4 62 05 50 cf       	{vex} vpdpbusd %ymm7,%ymm15,%ymm9
  10d0fd:	c4 e2 75 08 f9       	vpsignb %ymm1,%ymm1,%ymm7
  10d102:	c5 7d 6f bc 24 80 03 	vmovdqa 0x380(%rsp),%ymm15
  10d109:	00 00 
  10d10b:	c4 e2 05 08 f6       	vpsignb %ymm6,%ymm15,%ymm6
  10d110:	c4 62 15 50 ce       	{vex} vpdpbusd %ymm6,%ymm13,%ymm9
  10d115:	c5 7d 6f ac 24 40 03 	vmovdqa 0x340(%rsp),%ymm13
  10d11c:	00 00 
  10d11e:	c5 fd 6f b4 24 40 05 	vmovdqa 0x540(%rsp),%ymm6
  10d125:	00 00 
  10d127:	c4 e2 15 08 ed       	vpsignb %ymm5,%ymm13,%ymm5
  10d12c:	c4 62 1d 50 cd       	{vex} vpdpbusd %ymm5,%ymm12,%ymm9
  10d131:	c4 62 65 08 e3       	vpsignb %ymm3,%ymm3,%ymm12
  10d136:	62 91 7d 28 6f e8    	vmovdqa32 %ymm24,%ymm5
  10d13c:	c4 62 25 50 cc       	{vex} vpdpbusd %ymm4,%ymm11,%ymm9
  10d141:	c4 e2 2d 08 e3       	vpsignb %ymm3,%ymm10,%ymm4
  10d146:	c4 62 6d 08 da       	vpsignb %ymm2,%ymm2,%ymm11
  10d14b:	c4 e2 1d 50 ec       	{vex} vpdpbusd %ymm4,%ymm12,%ymm5
  10d150:	c4 e2 0d 08 e2       	vpsignb %ymm2,%ymm14,%ymm4
  10d155:	c4 e2 05 08 d2       	vpsignb %ymm2,%ymm15,%ymm2
  10d15a:	62 51 05 20 fe c9    	vpaddd %ymm9,%ymm31,%ymm9
  10d160:	c4 e2 25 50 ec       	{vex} vpdpbusd %ymm4,%ymm11,%ymm5
  10d165:	c4 e2 4d 08 e1       	vpsignb %ymm1,%ymm6,%ymm4
  10d16a:	c4 e2 15 08 c9       	vpsignb %ymm1,%ymm13,%ymm1
  10d16f:	c5 7d 6f 94 24 40 04 	vmovdqa 0x440(%rsp),%ymm10
  10d176:	00 00 
  10d178:	c4 e2 45 50 ec       	{vex} vpdpbusd %ymm4,%ymm7,%ymm5
  10d17d:	c5 fd 6f a4 24 00 05 	vmovdqa 0x500(%rsp),%ymm4
  10d184:	00 00 
  10d186:	c4 c2 2d 08 f2       	vpsignb %ymm10,%ymm10,%ymm6
  10d18b:	c4 c2 7d 08 c2       	vpsignb %ymm10,%ymm0,%ymm0
  10d190:	c4 c2 5d 08 e2       	vpsignb %ymm10,%ymm4,%ymm4
  10d195:	c4 e2 4d 50 ec       	{vex} vpdpbusd %ymm4,%ymm6,%ymm5
  10d19a:	c5 fd 6f a4 24 c0 04 	vmovdqa 0x4c0(%rsp),%ymm4
  10d1a1:	00 00 
  10d1a3:	c4 e2 5d 08 db       	vpsignb %ymm3,%ymm4,%ymm3
  10d1a8:	62 91 7d 28 6f e0    	vmovdqa32 %ymm24,%ymm4
  10d1ae:	c4 e2 1d 50 e3       	{vex} vpdpbusd %ymm3,%ymm12,%ymm4
  10d1b3:	c4 c1 7d 70 d9 4e    	vpshufd $0x4e,%ymm9,%ymm3
  10d1b9:	c4 e3 3d 02 db cc    	vpblendd $0xcc,%ymm3,%ymm8,%ymm3
  10d1bf:	c4 41 7d 70 c0 4e    	vpshufd $0x4e,%ymm8,%ymm8
  10d1c5:	c4 43 3d 02 c1 cc    	vpblendd $0xcc,%ymm9,%ymm8,%ymm8
  10d1cb:	c5 fc 5b db          	vcvtdq2ps %ymm3,%ymm3
  10d1cf:	c4 41 7c 5b c0       	vcvtdq2ps %ymm8,%ymm8
  10d1d4:	c4 e2 25 50 e2       	{vex} vpdpbusd %ymm2,%ymm11,%ymm4
  10d1d9:	c4 e2 45 50 e1       	{vex} vpdpbusd %ymm1,%ymm7,%ymm4
  10d1de:	c4 e2 4d 50 e0       	{vex} vpdpbusd %ymm0,%ymm6,%ymm4
  10d1e3:	62 f1 55 20 fe c5    	vpaddd %ymm5,%ymm21,%ymm0
  10d1e9:	62 f1 7d 20 fe e4    	vpaddd %ymm4,%ymm16,%ymm4
  10d1ef:	c5 f9 6f b4 24 90 05 	vmovdqa 0x590(%rsp),%xmm6
  10d1f6:	00 00 
  10d1f8:	c5 fd 70 d4 4e       	vpshufd $0x4e,%ymm4,%ymm2
  10d1fd:	c4 e3 7d 02 d2 cc    	vpblendd $0xcc,%ymm2,%ymm0,%ymm2
  10d203:	c5 fd 70 c0 4e       	vpshufd $0x4e,%ymm0,%ymm0
  10d208:	c4 e3 7d 02 cc cc    	vpblendd $0xcc,%ymm4,%ymm0,%ymm1
  10d20e:	c4 e2 49 8c 02       	vpmaskmovd (%rdx),%xmm6,%xmm0
  10d213:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  10d217:	62 e1 7c 28 5b e9    	vcvtdq2ps %ymm1,%ymm21
  10d21d:	c5 f9 70 c0 44       	vpshufd $0x44,%xmm0,%xmm0
  10d222:	c4 e2 7d 13 c0       	vcvtph2ps %xmm0,%ymm0
  10d227:	c4 e3 7d 04 e0 00    	vpermilps $0x0,%ymm0,%ymm4
  10d22d:	62 b1 5c 28 59 e1    	vmulps %ymm17,%ymm4,%ymm4
  10d233:	c4 e2 5d a8 58 80    	vfmadd213ps -0x80(%rax),%ymm4,%ymm3
  10d239:	c5 fc 29 58 80       	vmovaps %ymm3,-0x80(%rax)
  10d23e:	c4 e3 7d 04 d8 55    	vpermilps $0x55,%ymm0,%ymm3
  10d244:	62 b1 64 28 59 d9    	vmulps %ymm17,%ymm3,%ymm3
  10d24a:	c4 62 65 a8 40 a0    	vfmadd213ps -0x60(%rax),%ymm3,%ymm8
  10d250:	c4 e3 7d 04 d8 aa    	vpermilps $0xaa,%ymm0,%ymm3
  10d256:	c4 e3 7d 04 c0 ff    	vpermilps $0xff,%ymm0,%ymm0
  10d25c:	62 b1 64 28 59 d9    	vmulps %ymm17,%ymm3,%ymm3
  10d262:	62 b1 7c 28 59 c1    	vmulps %ymm17,%ymm0,%ymm0
  10d268:	c4 e2 65 a8 50 c0    	vfmadd213ps -0x40(%rax),%ymm3,%ymm2
  10d26e:	62 f2 55 20 a8 40 ff 	vfmadd213ps -0x20(%rax),%ymm21,%ymm0
  10d275:	c5 7c 29 40 a0       	vmovaps %ymm8,-0x60(%rax)
  10d27a:	c5 fc 29 50 c0       	vmovaps %ymm2,-0x40(%rax)
  10d27f:	c5 fc 29 40 e0       	vmovaps %ymm0,-0x20(%rax)
  10d284:	4c 39 c8             	cmp    %r9,%rax
  10d287:	0f 85 b1 fb ff ff    	jne    10ce3e <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x20ee>
  10d28d:	c5 fd 6f b4 24 c0 02 	vmovdqa 0x2c0(%rsp),%ymm6
  10d294:	00 00 
  10d296:	49 ff c4             	inc    %r12
  10d299:	48 81 c6 88 00 00 00 	add    $0x88,%rsi
  10d2a0:	4c 39 a4 24 40 02 00 	cmp    %r12,0x240(%rsp)
  10d2a7:	00 
  10d2a8:	0f 85 b2 f9 ff ff    	jne    10cc60 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1f10>
  10d2ae:	c5 fd 6f ee          	vmovdqa %ymm6,%ymm5
  10d2b2:	62 e1 7c 28 28 54 24 	vmovaps 0x5c0(%rsp),%ymm18
  10d2b9:	2e 
  10d2ba:	62 e1 7c 28 28 4c 24 	vmovaps 0x5e0(%rsp),%ymm17
  10d2c1:	2f 
  10d2c2:	62 e1 7c 28 28 44 24 	vmovaps 0x600(%rsp),%ymm16
  10d2c9:	30 
  10d2ca:	c5 7c 28 bc 24 20 06 	vmovaps 0x620(%rsp),%ymm15
  10d2d1:	00 00 
  10d2d3:	c5 7c 28 b4 24 40 06 	vmovaps 0x640(%rsp),%ymm14
  10d2da:	00 00 
  10d2dc:	c5 7c 28 ac 24 60 06 	vmovaps 0x660(%rsp),%ymm13
  10d2e3:	00 00 
  10d2e5:	c5 7c 28 a4 24 80 06 	vmovaps 0x680(%rsp),%ymm12
  10d2ec:	00 00 
  10d2ee:	c5 7c 28 9c 24 a0 06 	vmovaps 0x6a0(%rsp),%ymm11
  10d2f5:	00 00 
  10d2f7:	c5 7c 28 8c 24 c0 06 	vmovaps 0x6c0(%rsp),%ymm9
  10d2fe:	00 00 
  10d300:	c5 7c 28 84 24 e0 06 	vmovaps 0x6e0(%rsp),%ymm8
  10d307:	00 00 
  10d309:	c5 fc 28 bc 24 00 07 	vmovaps 0x700(%rsp),%ymm7
  10d310:	00 00 
  10d312:	c5 fc 28 b4 24 20 07 	vmovaps 0x720(%rsp),%ymm6
  10d319:	00 00 
  10d31b:	c5 fc 28 a4 24 40 07 	vmovaps 0x740(%rsp),%ymm4
  10d322:	00 00 
  10d324:	c5 fc 28 94 24 60 07 	vmovaps 0x760(%rsp),%ymm2
  10d32b:	00 00 
  10d32d:	c5 fc 28 8c 24 80 07 	vmovaps 0x780(%rsp),%ymm1
  10d334:	00 00 
  10d336:	c5 fc 28 84 24 a0 07 	vmovaps 0x7a0(%rsp),%ymm0
  10d33d:	00 00 
  10d33f:	48 8b 8c 24 c0 01 00 	mov    0x1c0(%rsp),%rcx
  10d346:	00 
  10d347:	4c 8b 84 24 a0 01 00 	mov    0x1a0(%rsp),%r8
  10d34e:	00 
  10d34f:	48 8b 94 24 90 01 00 	mov    0x190(%rsp),%rdx
  10d356:	00 
  10d357:	48 8b bc 24 88 01 00 	mov    0x188(%rsp),%rdi
  10d35e:	00 
  10d35f:	48 8b 84 24 18 01 00 	mov    0x118(%rsp),%rax
  10d366:	00 
  10d367:	62 e1 7c 28 11 12    	vmovups %ymm18,(%rdx)
  10d36d:	48 ff c1             	inc    %rcx
  10d370:	62 e1 7c 28 11 0c 82 	vmovups %ymm17,(%rdx,%rax,4)
  10d377:	48 8b 84 24 40 01 00 	mov    0x140(%rsp),%rax
  10d37e:	00 
  10d37f:	4c 29 c0             	sub    %r8,%rax
  10d382:	62 e1 7c 28 11 04 82 	vmovups %ymm16,(%rdx,%rax,4)
  10d389:	48 8b 84 24 e0 00 00 	mov    0xe0(%rsp),%rax
  10d390:	00 
  10d391:	c5 7c 11 3c 82       	vmovups %ymm15,(%rdx,%rax,4)
  10d396:	48 8b 84 24 48 01 00 	mov    0x148(%rsp),%rax
  10d39d:	00 
  10d39e:	c5 7c 11 34 82       	vmovups %ymm14,(%rdx,%rax,4)
  10d3a3:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  10d3aa:	00 
  10d3ab:	4c 29 c0             	sub    %r8,%rax
  10d3ae:	c5 7c 11 2c 82       	vmovups %ymm13,(%rdx,%rax,4)
  10d3b3:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  10d3ba:	00 
  10d3bb:	4c 29 c0             	sub    %r8,%rax
  10d3be:	c5 7c 11 24 82       	vmovups %ymm12,(%rdx,%rax,4)
  10d3c3:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  10d3ca:	00 
  10d3cb:	c5 7c 11 1c 82       	vmovups %ymm11,(%rdx,%rax,4)
  10d3d0:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  10d3d7:	00 
  10d3d8:	4c 29 c0             	sub    %r8,%rax
  10d3db:	c5 7c 11 0c 82       	vmovups %ymm9,(%rdx,%rax,4)
  10d3e0:	48 8b 84 24 70 01 00 	mov    0x170(%rsp),%rax
  10d3e7:	00 
  10d3e8:	4c 29 c0             	sub    %r8,%rax
  10d3eb:	c5 7c 11 04 82       	vmovups %ymm8,(%rdx,%rax,4)
  10d3f0:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
  10d3f7:	00 
  10d3f8:	4c 29 c0             	sub    %r8,%rax
  10d3fb:	c5 fc 11 3c 82       	vmovups %ymm7,(%rdx,%rax,4)
  10d400:	48 8b 84 24 10 01 00 	mov    0x110(%rsp),%rax
  10d407:	00 
  10d408:	c5 fc 11 34 82       	vmovups %ymm6,(%rdx,%rax,4)
  10d40d:	48 8b 84 24 28 01 00 	mov    0x128(%rsp),%rax
  10d414:	00 
  10d415:	4c 29 c0             	sub    %r8,%rax
  10d418:	c5 fc 11 24 82       	vmovups %ymm4,(%rdx,%rax,4)
  10d41d:	48 8b 84 24 80 01 00 	mov    0x180(%rsp),%rax
  10d424:	00 
  10d425:	4c 29 c0             	sub    %r8,%rax
  10d428:	c5 fc 11 14 82       	vmovups %ymm2,(%rdx,%rax,4)
  10d42d:	48 8b 84 24 20 01 00 	mov    0x120(%rsp),%rax
  10d434:	00 
  10d435:	4c 29 c0             	sub    %r8,%rax
  10d438:	c5 fc 11 0c 82       	vmovups %ymm1,(%rdx,%rax,4)
  10d43d:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  10d444:	00 
  10d445:	c5 fc 11 04 82       	vmovups %ymm0,(%rdx,%rax,4)
  10d44a:	48 8b 84 24 50 01 00 	mov    0x150(%rsp),%rax
  10d451:	00 
  10d452:	48 83 c2 20          	add    $0x20,%rdx
  10d456:	49 29 c3             	sub    %rax,%r11
  10d459:	48 01 c7             	add    %rax,%rdi
  10d45c:	48 39 8c 24 58 01 00 	cmp    %rcx,0x158(%rsp)
  10d463:	00 
  10d464:	0f 85 16 f7 ff ff    	jne    10cb80 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1e30>
  10d46a:	49 89 c6             	mov    %rax,%r14
  10d46d:	48 8b bc 24 b0 00 00 	mov    0xb0(%rsp),%rdi
  10d474:	00 
  10d475:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  10d47c:	00 
  10d47d:	48 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%rcx
  10d484:	00 
  10d485:	4c 8b 54 24 40       	mov    0x40(%rsp),%r10
  10d48a:	4a 8d 34 30          	lea    (%rax,%r14,1),%rsi
  10d48e:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  10d495:	00 
  10d496:	48 8b 9c 24 c0 00 00 	mov    0xc0(%rsp),%rbx
  10d49d:	00 
  10d49e:	48 83 c7 04          	add    $0x4,%rdi
  10d4a2:	4d 01 d0             	add    %r10,%r8
  10d4a5:	4c 01 d1             	add    %r10,%rcx
  10d4a8:	4c 01 94 24 60 01 00 	add    %r10,0x160(%rsp)
  10d4af:	00 
  10d4b0:	4c 01 94 24 68 01 00 	add    %r10,0x168(%rsp)
  10d4b7:	00 
  10d4b8:	4c 01 94 24 70 01 00 	add    %r10,0x170(%rsp)
  10d4bf:	00 
  10d4c0:	4c 01 94 24 80 01 00 	add    %r10,0x180(%rsp)
  10d4c7:	00 
  10d4c8:	48 01 9c 24 c8 00 00 	add    %rbx,0xc8(%rsp)
  10d4cf:	00 
  10d4d0:	48 39 c7             	cmp    %rax,%rdi
  10d4d3:	0f 8c 87 f5 ff ff    	jl     10ca60 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1d10>
  10d4d9:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  10d4de:	62 e1 fd 28 6f 04 24 	vmovdqa64 (%rsp),%ymm16
  10d4e5:	4c 8b b4 24 88 00 00 	mov    0x88(%rsp),%r14
  10d4ec:	00 
  10d4ed:	48 ff c8             	dec    %rax
  10d4f0:	44 8b 4d 10          	mov    0x10(%rbp),%r9d
  10d4f4:	4c 8b bc 24 40 02 00 	mov    0x240(%rsp),%r15
  10d4fb:	00 
  10d4fc:	48 29 c8             	sub    %rcx,%rax
  10d4ff:	48 83 e0 fc          	and    $0xfffffffffffffffc,%rax
  10d503:	48 8d 4c 01 04       	lea    0x4(%rcx,%rax,1),%rcx
  10d508:	48 39 8c 24 d0 00 00 	cmp    %rcx,0xd0(%rsp)
  10d50f:	00 
  10d510:	0f 8e 56 09 00 00    	jle    10de6c <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x311c>
  10d516:	45 85 c9             	test   %r9d,%r9d
  10d519:	41 8d 41 07          	lea    0x7(%r9),%eax
  10d51d:	4c 8b 94 24 98 01 00 	mov    0x198(%rsp),%r10
  10d524:	00 
  10d525:	41 0f 49 c1          	cmovns %r9d,%eax
  10d529:	c1 f8 03             	sar    $0x3,%eax
  10d52c:	48 98                	cltq
  10d52e:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
  10d535:	00 
  10d536:	4c 39 d0             	cmp    %r10,%rax
  10d539:	0f 8e 2d 09 00 00    	jle    10de6c <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x311c>
  10d53f:	48 8b 9c 24 78 01 00 	mov    0x178(%rsp),%rbx
  10d546:	00 
  10d547:	48 8d 04 8d 01 00 00 	lea    0x1(,%rcx,4),%rax
  10d54e:	00 
  10d54f:	62 61 7d 28 6f 1d 07 	vmovdqa32 0x21b07(%rip),%ymm27        # 12f060 <_ZL11iq2xxs_grid+0xb20>
  10d556:	1b 02 00 
  10d559:	4d 69 df 88 00 00 00 	imul   $0x88,%r15,%r11
  10d560:	62 a1 fd 28 6f d8    	vmovdqa64 %ymm16,%ymm19
  10d566:	62 a1 45 00 ef ff    	vpxord %xmm23,%xmm23,%xmm23
  10d56c:	4c 89 9c 24 70 01 00 	mov    %r11,0x170(%rsp)
  10d573:	00 
  10d574:	48 89 df             	mov    %rbx,%rdi
  10d577:	48 0f af c3          	imul   %rbx,%rax
  10d57b:	48 89 da             	mov    %rbx,%rdx
  10d57e:	4c 8d 0c 1b          	lea    (%rbx,%rbx,1),%r9
  10d582:	48 c1 e7 04          	shl    $0x4,%rdi
  10d586:	4c 89 8c 24 18 01 00 	mov    %r9,0x118(%rsp)
  10d58d:	00 
  10d58e:	48 0f af d1          	imul   %rcx,%rdx
  10d592:	48 89 bc 24 28 01 00 	mov    %rdi,0x128(%rsp)
  10d599:	00 
  10d59a:	48 8b bc 24 b8 00 00 	mov    0xb8(%rsp),%rdi
  10d5a1:	00 
  10d5a2:	48 c1 e2 02          	shl    $0x2,%rdx
  10d5a6:	4a 8d 34 d2          	lea    (%rdx,%r10,8),%rsi
  10d5aa:	48 8d 34 b7          	lea    (%rdi,%rsi,4),%rsi
  10d5ae:	4c 89 ff             	mov    %r15,%rdi
  10d5b1:	48 0f af f9          	imul   %rcx,%rdi
  10d5b5:	48 69 ff 88 00 00 00 	imul   $0x88,%rdi,%rdi
  10d5bc:	4d 8d 2c 3e          	lea    (%r14,%rdi,1),%r13
  10d5c0:	48 89 d7             	mov    %rdx,%rdi
  10d5c3:	49 89 de             	mov    %rbx,%r14
  10d5c6:	48 29 c7             	sub    %rax,%rdi
  10d5c9:	48 89 bc 24 20 01 00 	mov    %rdi,0x120(%rsp)
  10d5d0:	00 
  10d5d1:	4c 89 d7             	mov    %r10,%rdi
  10d5d4:	49 0f af ff          	imul   %r15,%rdi
  10d5d8:	48 69 df 88 00 00 00 	imul   $0x88,%rdi,%rbx
  10d5df:	48 bf 0f 0f 0f 0f 0f 	movabs $0xf0f0f0f0f0f0f0f,%rdi
  10d5e6:	0f 0f 0f 
  10d5e9:	62 62 fd 28 7c c7    	vpbroadcastq %rdi,%ymm24
  10d5ef:	48 89 9c 24 30 01 00 	mov    %rbx,0x130(%rsp)
  10d5f6:	00 
  10d5f7:	48 89 c3             	mov    %rax,%rbx
  10d5fa:	48 29 d3             	sub    %rdx,%rbx
  10d5fd:	48 89 c2             	mov    %rax,%rdx
  10d600:	4e 8d 24 33          	lea    (%rbx,%r14,1),%r12
  10d604:	48 89 9c 24 50 01 00 	mov    %rbx,0x150(%rsp)
  10d60b:	00 
  10d60c:	48 8b 84 24 20 01 00 	mov    0x120(%rsp),%rax
  10d613:	00 
  10d614:	48 8b 9c 24 78 01 00 	mov    0x178(%rsp),%rbx
  10d61b:	00 
  10d61c:	4c 8b 8c 24 30 01 00 	mov    0x130(%rsp),%r9
  10d623:	00 
  10d624:	48 89 f7             	mov    %rsi,%rdi
  10d627:	4c 8b 84 24 98 01 00 	mov    0x198(%rsp),%r8
  10d62e:	00 
  10d62f:	48 89 8c 24 48 01 00 	mov    %rcx,0x148(%rsp)
  10d636:	00 
  10d637:	48 89 b4 24 38 01 00 	mov    %rsi,0x138(%rsp)
  10d63e:	00 
  10d63f:	4c 89 ac 24 68 01 00 	mov    %r13,0x168(%rsp)
  10d646:	00 
  10d647:	48 01 d0             	add    %rdx,%rax
  10d64a:	48 8d 14 5a          	lea    (%rdx,%rbx,2),%rdx
  10d64e:	49 89 d2             	mov    %rdx,%r10
  10d651:	48 89 94 24 40 01 00 	mov    %rdx,0x140(%rsp)
  10d658:	00 
  10d659:	49 29 c2             	sub    %rax,%r10
  10d65c:	4c 89 94 24 60 01 00 	mov    %r10,0x160(%rsp)
  10d663:	00 
  10d664:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  10d66b:	00 00 00 00 
  10d66f:	90                   	nop
  10d670:	83 bc 24 8c 05 00 00 	cmpl   $0x1f,0x58c(%rsp)
  10d677:	1f 
  10d678:	0f 8e 14 08 00 00    	jle    10de92 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x3142>
  10d67e:	48 8b 84 24 80 05 00 	mov    0x580(%rsp),%rax
  10d685:	00 
  10d686:	48 8b 8c 24 68 01 00 	mov    0x168(%rsp),%rcx
  10d68d:	00 
  10d68e:	48 8b 15 8b e8 03 00 	mov    0x3e88b(%rip),%rdx        # 14bf20 <ggml_table_f32_e8m0_half@@Base-0x1420>
  10d695:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
  10d699:	31 f6                	xor    %esi,%esi
  10d69b:	c5 fc 29 a4 24 c0 04 	vmovaps %ymm4,0x4c0(%rsp)
  10d6a2:	00 00 
  10d6a4:	4c 89 84 24 90 01 00 	mov    %r8,0x190(%rsp)
  10d6ab:	00 
  10d6ac:	48 89 bc 24 88 01 00 	mov    %rdi,0x188(%rsp)
  10d6b3:	00 
  10d6b4:	c5 fc 29 a4 24 00 05 	vmovaps %ymm4,0x500(%rsp)
  10d6bb:	00 00 
  10d6bd:	4c 89 8c 24 80 01 00 	mov    %r9,0x180(%rsp)
  10d6c4:	00 
  10d6c5:	c5 fc 29 a4 24 40 04 	vmovaps %ymm4,0x440(%rsp)
  10d6cc:	00 00 
  10d6ce:	c5 fc 29 a4 24 80 04 	vmovaps %ymm4,0x480(%rsp)
  10d6d5:	00 00 
  10d6d7:	4c 01 c8             	add    %r9,%rax
  10d6da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  10d6e0:	c5 fe 6f 70 08       	vmovdqu 0x8(%rax),%ymm6
  10d6e5:	62 f2 25 20 36 98 28 	vpermd 0x28(%rax),%ymm27,%ymm3
  10d6ec:	00 00 00 
  10d6ef:	62 f2 25 20 36 90 68 	vpermd 0x68(%rax),%ymm27,%ymm2
  10d6f6:	00 00 00 
  10d6f9:	62 21 7d 28 6f ff    	vmovdqa32 %ymm23,%ymm31
  10d6ff:	44 0f b6 70 04       	movzbl 0x4(%rax),%r14d
  10d704:	44 0f b6 68 05       	movzbl 0x5(%rax),%r13d
  10d709:	44 0f b6 58 06       	movzbl 0x6(%rax),%r11d
  10d70e:	48 ff c6             	inc    %rsi
  10d711:	0f b6 78 07          	movzbl 0x7(%rax),%edi
  10d715:	44 0f b6 08          	movzbl (%rax),%r9d
  10d719:	0f b6 58 02          	movzbl 0x2(%rax),%ebx
  10d71d:	48 81 c1 88 00 00 00 	add    $0x88,%rcx
  10d724:	44 0f b6 50 03       	movzbl 0x3(%rax),%r10d
  10d729:	44 0f b6 40 01       	movzbl 0x1(%rax),%r8d
  10d72e:	48 05 88 00 00 00    	add    $0x88,%rax
  10d734:	c4 e3 4d 02 db f0    	vpblendd $0xf0,%ymm3,%ymm6,%ymm3
  10d73a:	62 f2 25 20 36 ce    	vpermd %ymm6,%ymm27,%ymm1
  10d740:	c5 fe 6f 70 c0       	vmovdqu -0x40(%rax),%ymm6
  10d745:	c4 e3 75 02 48 a0 f0 	vpblendd $0xf0,-0x60(%rax),%ymm1,%ymm1
  10d74c:	62 91 e5 28 db f8    	vpandq %ymm24,%ymm3,%ymm7
  10d752:	c5 e5 71 d3 04       	vpsrlw $0x4,%ymm3,%ymm3
  10d757:	62 91 e5 28 db d8    	vpandq %ymm24,%ymm3,%ymm3
  10d75d:	62 f2 65 20 00 db    	vpshufb %ymm3,%ymm19,%ymm3
  10d763:	62 f2 65 20 00 ff    	vpshufb %ymm7,%ymm19,%ymm7
  10d769:	62 e1 7d 28 70 e7 88 	vpshufd $0x88,%ymm7,%ymm20
  10d770:	c5 fd 70 ff dd       	vpshufd $0xdd,%ymm7,%ymm7
  10d775:	c5 fd 7f bc 24 80 03 	vmovdqa %ymm7,0x380(%rsp)
  10d77c:	00 00 
  10d77e:	62 e1 7d 28 70 f3 88 	vpshufd $0x88,%ymm3,%ymm22
  10d785:	62 f2 25 20 36 c6    	vpermd %ymm6,%ymm27,%ymm0
  10d78b:	c4 e3 4d 02 d2 f0    	vpblendd $0xf0,%ymm2,%ymm6,%ymm2
  10d791:	62 91 f5 28 db f0    	vpandq %ymm24,%ymm1,%ymm6
  10d797:	c5 f5 71 d1 04       	vpsrlw $0x4,%ymm1,%ymm1
  10d79c:	c4 e3 7d 02 40 e0 f0 	vpblendd $0xf0,-0x20(%rax),%ymm0,%ymm0
  10d7a3:	62 91 f5 28 db c8    	vpandq %ymm24,%ymm1,%ymm1
  10d7a9:	62 91 ed 28 db e8    	vpandq %ymm24,%ymm2,%ymm5
  10d7af:	c5 ed 71 d2 04       	vpsrlw $0x4,%ymm2,%ymm2
  10d7b4:	62 91 fd 28 db e0    	vpandq %ymm24,%ymm0,%ymm4
  10d7ba:	c5 fd 71 d0 04       	vpsrlw $0x4,%ymm0,%ymm0
  10d7bf:	62 91 fd 28 db c0    	vpandq %ymm24,%ymm0,%ymm0
  10d7c5:	62 f2 65 20 00 c9    	vpshufb %ymm1,%ymm19,%ymm1
  10d7cb:	62 f2 65 20 00 c0    	vpshufb %ymm0,%ymm19,%ymm0
  10d7d1:	c5 7d 70 e1 88       	vpshufd $0x88,%ymm1,%ymm12
  10d7d6:	62 e1 7d 28 70 c8 88 	vpshufd $0x88,%ymm0,%ymm17
  10d7dd:	62 e1 7d 28 70 c1 dd 	vpshufd $0xdd,%ymm1,%ymm16
  10d7e4:	c5 fd 70 c8 dd       	vpshufd $0xdd,%ymm0,%ymm1
  10d7e9:	c4 a1 7a 10 04 b2    	vmovss (%rdx,%r14,4),%xmm0
  10d7ef:	c5 fd 7f 8c 24 e0 01 	vmovdqa %ymm1,0x1e0(%rsp)
  10d7f6:	00 00 
  10d7f8:	c4 a3 79 21 04 aa 10 	vinsertps $0x10,(%rdx,%r13,4),%xmm0,%xmm0
  10d7ff:	c4 a1 7a 10 0c 9a    	vmovss (%rdx,%r11,4),%xmm1
  10d805:	c4 e3 71 21 0c ba 10 	vinsertps $0x10,(%rdx,%rdi,4),%xmm1,%xmm1
  10d80c:	62 f2 65 20 00 e4    	vpshufb %ymm4,%ymm19,%ymm4
  10d812:	c5 7d 7f a4 24 40 05 	vmovdqa %ymm12,0x540(%rsp)
  10d819:	00 00 
  10d81b:	62 61 7d 28 70 ec 88 	vpshufd $0x88,%ymm4,%ymm29
  10d822:	c5 7d 70 fc dd       	vpshufd $0xdd,%ymm4,%ymm15
  10d827:	62 61 7d 28 7f 6c 24 	vmovdqa32 %ymm29,0x3c0(%rsp)
  10d82e:	1e 
  10d82f:	62 91 ed 28 db d0    	vpandq %ymm24,%ymm2,%ymm2
  10d835:	62 f2 65 20 00 d2    	vpshufb %ymm2,%ymm19,%ymm2
  10d83b:	c5 fd 70 e3 dd       	vpshufd $0xdd,%ymm3,%ymm4
  10d840:	62 e1 7d 28 70 d2 dd 	vpshufd $0xdd,%ymm2,%ymm18
  10d847:	c5 7d 7f bc 24 c0 02 	vmovdqa %ymm15,0x2c0(%rsp)
  10d84e:	00 00 
  10d850:	c5 fd 7f a4 24 80 02 	vmovdqa %ymm4,0x280(%rsp)
  10d857:	00 00 
  10d859:	62 f2 65 20 00 f6    	vpshufb %ymm6,%ymm19,%ymm6
  10d85f:	62 f2 65 20 00 ed    	vpshufb %ymm5,%ymm19,%ymm5
  10d865:	62 61 7d 28 70 f6 88 	vpshufd $0x88,%ymm6,%ymm30
  10d86c:	62 e1 7d 28 70 ed 88 	vpshufd $0x88,%ymm5,%ymm21
  10d873:	62 61 7d 28 7f 74 24 	vmovdqa32 %ymm30,0x400(%rsp)
  10d87a:	20 
  10d87b:	c5 fd 70 f6 dd       	vpshufd $0xdd,%ymm6,%ymm6
  10d880:	c5 fd 70 ed dd       	vpshufd $0xdd,%ymm5,%ymm5
  10d885:	c5 fd 7f b4 24 40 03 	vmovdqa %ymm6,0x340(%rsp)
  10d88c:	00 00 
  10d88e:	62 61 7d 28 70 ca 88 	vpshufd $0x88,%ymm2,%ymm25
  10d895:	c5 f8 16 c1          	vmovlhps %xmm1,%xmm0,%xmm0
  10d899:	c5 fd 7f ac 24 00 03 	vmovdqa %ymm5,0x300(%rsp)
  10d8a0:	00 00 
  10d8a2:	c5 fa 10 0c 9a       	vmovss (%rdx,%rbx,4),%xmm1
  10d8a7:	c4 a1 7a 10 2c 8a    	vmovss (%rdx,%r9,4),%xmm5
  10d8ad:	c4 a3 71 21 0c 92 10 	vinsertps $0x10,(%rdx,%r10,4),%xmm1,%xmm1
  10d8b4:	62 e1 7d 28 7f 44 24 	vmovdqa32 %ymm16,0x240(%rsp)
  10d8bb:	12 
  10d8bc:	62 e1 7d 28 7f 54 24 	vmovdqa32 %ymm18,0x200(%rsp)
  10d8c3:	10 
  10d8c4:	c4 a3 51 21 2c 82 10 	vinsertps $0x10,(%rdx,%r8,4),%xmm5,%xmm5
  10d8cb:	c5 fe 6f 61 e0       	vmovdqu -0x20(%rcx),%ymm4
  10d8d0:	c5 fe 6f 51 c0       	vmovdqu -0x40(%rcx),%ymm2
  10d8d5:	c5 d0 16 e9          	vmovlhps %xmm1,%xmm5,%xmm5
  10d8d9:	c4 63 5d 46 cc 00    	vperm2i128 $0x0,%ymm4,%ymm4,%ymm9
  10d8df:	c5 fe 6f 49 a0       	vmovdqu -0x60(%rcx),%ymm1
  10d8e4:	c4 63 6d 46 c2 00    	vperm2i128 $0x0,%ymm2,%ymm2,%ymm8
  10d8ea:	c4 e3 55 18 e8 01    	vinsertf128 $0x1,%xmm0,%ymm5,%ymm5
  10d8f0:	c5 fe 6f 41 80       	vmovdqu -0x80(%rcx),%ymm0
  10d8f5:	c4 c1 7d 70 d9 a0    	vpshufd $0xa0,%ymm9,%ymm3
  10d8fb:	c4 41 7d 70 f8 a0    	vpshufd $0xa0,%ymm8,%ymm15
  10d901:	c4 e3 5d 46 e4 11    	vperm2i128 $0x11,%ymm4,%ymm4,%ymm4
  10d907:	c4 e3 6d 46 d2 11    	vperm2i128 $0x11,%ymm2,%ymm2,%ymm2
  10d90d:	c4 41 7d 70 c9 f5    	vpshufd $0xf5,%ymm9,%ymm9
  10d913:	c4 41 7d 70 c0 f5    	vpshufd $0xf5,%ymm8,%ymm8
  10d919:	62 e1 7d 28 70 d4 a0 	vpshufd $0xa0,%ymm4,%ymm18
  10d920:	c5 7d 70 e2 a0       	vpshufd $0xa0,%ymm2,%ymm12
  10d925:	c5 fd 70 e4 f5       	vpshufd $0xf5,%ymm4,%ymm4
  10d92a:	c5 fd 70 d2 f5       	vpshufd $0xf5,%ymm2,%ymm2
  10d92f:	c4 e3 75 46 f9 00    	vperm2i128 $0x0,%ymm1,%ymm1,%ymm7
  10d935:	c4 e3 75 46 c9 11    	vperm2i128 $0x11,%ymm1,%ymm1,%ymm1
  10d93b:	c4 e3 7d 46 f0 00    	vperm2i128 $0x0,%ymm0,%ymm0,%ymm6
  10d941:	c4 e3 7d 46 c0 11    	vperm2i128 $0x11,%ymm0,%ymm0,%ymm0
  10d947:	c5 7d 70 f7 a0       	vpshufd $0xa0,%ymm7,%ymm14
  10d94c:	c5 7d 70 d9 a0       	vpshufd $0xa0,%ymm1,%ymm11
  10d951:	c5 7d 70 d0 a0       	vpshufd $0xa0,%ymm0,%ymm10
  10d956:	c5 fd 70 c0 f5       	vpshufd $0xf5,%ymm0,%ymm0
  10d95b:	c5 fd 7f 84 24 c0 01 	vmovdqa %ymm0,0x1c0(%rsp)
  10d962:	00 00 
  10d964:	c4 e2 65 08 c3       	vpsignb %ymm3,%ymm3,%ymm0
  10d969:	62 61 fd 28 6f f0    	vmovdqa64 %ymm0,%ymm30
  10d96f:	62 91 fd 28 6f c1    	vmovdqa64 %ymm25,%ymm0
  10d975:	c5 7d 70 ee a0       	vpshufd $0xa0,%ymm6,%ymm13
  10d97a:	c5 fd 70 ff f5       	vpshufd $0xf5,%ymm7,%ymm7
  10d97f:	c4 e2 7d 08 c3       	vpsignb %ymm3,%ymm0,%ymm0
  10d984:	62 62 0d 20 50 f8    	vpdpbusd %ymm0,%ymm30,%ymm31
  10d98a:	c4 c2 05 08 c7       	vpsignb %ymm15,%ymm15,%ymm0
  10d98f:	c5 fd 70 f6 f5       	vpshufd $0xf5,%ymm6,%ymm6
  10d994:	62 61 fd 28 6f e8    	vmovdqa64 %ymm0,%ymm29
  10d99a:	62 b1 fd 28 6f c6    	vmovdqa64 %ymm22,%ymm0
  10d9a0:	c5 fd 70 c9 f5       	vpshufd $0xf5,%ymm1,%ymm1
  10d9a5:	c4 c2 7d 08 c7       	vpsignb %ymm15,%ymm0,%ymm0
  10d9aa:	62 81 7d 28 6f c7    	vmovdqa32 %ymm31,%ymm16
  10d9b0:	62 e2 15 20 50 c0    	vpdpbusd %ymm0,%ymm29,%ymm16
  10d9b6:	c4 c2 0d 08 c6       	vpsignb %ymm14,%ymm14,%ymm0
  10d9bb:	62 61 fd 28 6f e0    	vmovdqa64 %ymm0,%ymm28
  10d9c1:	62 b1 fd 28 6f c5    	vmovdqa64 %ymm21,%ymm0
  10d9c7:	c4 c2 7d 08 c6       	vpsignb %ymm14,%ymm0,%ymm0
  10d9cc:	62 e2 1d 20 50 c0    	vpdpbusd %ymm0,%ymm28,%ymm16
  10d9d2:	c4 c2 15 08 c5       	vpsignb %ymm13,%ymm13,%ymm0
  10d9d7:	62 61 fd 28 6f d0    	vmovdqa64 %ymm0,%ymm26
  10d9dd:	62 b1 fd 28 6f c4    	vmovdqa64 %ymm20,%ymm0
  10d9e3:	c4 c2 7d 08 c5       	vpsignb %ymm13,%ymm0,%ymm0
  10d9e8:	62 61 fd 28 6f f8    	vmovdqa64 %ymm0,%ymm31
  10d9ee:	62 b1 7d 28 6f c0    	vmovdqa32 %ymm16,%ymm0
  10d9f4:	62 92 2d 20 50 c7    	vpdpbusd %ymm31,%ymm26,%ymm0
  10d9fa:	62 21 7d 28 6f ff    	vmovdqa32 %ymm23,%ymm31
  10da00:	c5 fd 7f 84 24 a0 01 	vmovdqa %ymm0,0x1a0(%rsp)
  10da07:	00 00 
  10da09:	62 b1 fd 28 6f c1    	vmovdqa64 %ymm17,%ymm0
  10da0f:	c4 e2 7d 08 db       	vpsignb %ymm3,%ymm0,%ymm3
  10da14:	62 62 0d 20 50 fb    	vpdpbusd %ymm3,%ymm30,%ymm31
  10da1a:	62 b1 fd 28 6f c2    	vmovdqa64 %ymm18,%ymm0
  10da20:	c5 fd 6f 9c 24 40 05 	vmovdqa 0x540(%rsp),%ymm3
  10da27:	00 00 
  10da29:	62 61 7d 28 6f 74 24 	vmovdqa32 0x400(%rsp),%ymm30
  10da30:	20 
  10da31:	c4 42 65 08 ff       	vpsignb %ymm15,%ymm3,%ymm15
  10da36:	62 42 15 20 50 ff    	vpdpbusd %ymm15,%ymm29,%ymm31
  10da3c:	62 61 7d 28 6f 6c 24 	vmovdqa32 0x3c0(%rsp),%ymm29
  10da43:	1e 
  10da44:	62 91 fd 28 6f dd    	vmovdqa64 %ymm29,%ymm3
  10da4a:	c4 42 65 08 f6       	vpsignb %ymm14,%ymm3,%ymm14
  10da4f:	62 42 1d 20 50 fe    	vpdpbusd %ymm14,%ymm28,%ymm31
  10da55:	62 91 fd 28 6f de    	vmovdqa64 %ymm30,%ymm3
  10da5b:	c4 42 65 08 ed       	vpsignb %ymm13,%ymm3,%ymm13
  10da60:	62 b1 fd 28 6f da    	vmovdqa64 %ymm18,%ymm3
  10da66:	c4 62 65 08 f3       	vpsignb %ymm3,%ymm3,%ymm14
  10da6b:	62 91 fd 28 6f d9    	vmovdqa64 %ymm25,%ymm3
  10da71:	c4 62 65 08 f8       	vpsignb %ymm0,%ymm3,%ymm15
  10da76:	62 b1 fd 28 6f de    	vmovdqa64 %ymm22,%ymm3
  10da7c:	62 b1 fd 28 6f c4    	vmovdqa64 %ymm20,%ymm0
  10da82:	62 01 7d 28 6f e7    	vmovdqa32 %ymm31,%ymm28
  10da88:	c4 c2 65 08 dc       	vpsignb %ymm12,%ymm3,%ymm3
  10da8d:	c4 c2 7d 08 c2       	vpsignb %ymm10,%ymm0,%ymm0
  10da92:	62 e1 7d 28 6f 64 24 	vmovdqa32 0x280(%rsp),%ymm20
  10da99:	14 
  10da9a:	62 42 2d 20 50 e5    	vpdpbusd %ymm13,%ymm26,%ymm28
  10daa0:	62 31 7d 28 6f ef    	vmovdqa32 %ymm23,%ymm13
  10daa6:	c4 42 0d 50 ef       	{vex} vpdpbusd %ymm15,%ymm14,%ymm13
  10daab:	c4 42 1d 08 fc       	vpsignb %ymm12,%ymm12,%ymm15
  10dab0:	c4 62 05 50 eb       	{vex} vpdpbusd %ymm3,%ymm15,%ymm13
  10dab5:	c4 c2 25 08 db       	vpsignb %ymm11,%ymm11,%ymm3
  10daba:	62 e1 fd 28 6f c3    	vmovdqa64 %ymm3,%ymm16
  10dac0:	62 b1 fd 28 6f dd    	vmovdqa64 %ymm21,%ymm3
  10dac6:	c4 c2 65 08 db       	vpsignb %ymm11,%ymm3,%ymm3
  10dacb:	62 72 7d 20 50 eb    	vpdpbusd %ymm3,%ymm16,%ymm13
  10dad1:	c4 c2 2d 08 da       	vpsignb %ymm10,%ymm10,%ymm3
  10dad6:	c4 62 65 50 e8       	{vex} vpdpbusd %ymm0,%ymm3,%ymm13
  10dadb:	62 b1 fd 28 6f c1    	vmovdqa64 %ymm17,%ymm0
  10dae1:	62 c1 7d 28 6f f5    	vmovdqa32 %ymm13,%ymm22
  10dae7:	62 31 fd 28 6f ea    	vmovdqa64 %ymm18,%ymm13
  10daed:	62 e1 7d 28 6f 54 24 	vmovdqa32 0x200(%rsp),%ymm18
  10daf4:	10 
  10daf5:	62 e1 7d 28 6f 6c 24 	vmovdqa32 0x300(%rsp),%ymm21
  10dafc:	18 
  10dafd:	c4 c2 7d 08 c5       	vpsignb %ymm13,%ymm0,%ymm0
  10db02:	62 e1 7d 28 6f 4c 24 	vmovdqa32 0x380(%rsp),%ymm17
  10db09:	1c 
  10db0a:	62 31 7d 28 6f ef    	vmovdqa32 %ymm23,%ymm13
  10db10:	c4 62 0d 50 e8       	{vex} vpdpbusd %ymm0,%ymm14,%ymm13
  10db15:	c4 62 45 08 f7       	vpsignb %ymm7,%ymm7,%ymm14
  10db1a:	c5 fd 6f 84 24 40 05 	vmovdqa 0x540(%rsp),%ymm0
  10db21:	00 00 
  10db23:	c4 42 7d 08 e4       	vpsignb %ymm12,%ymm0,%ymm12
  10db28:	c4 42 05 50 ec       	{vex} vpdpbusd %ymm12,%ymm15,%ymm13
  10db2d:	c4 42 3d 08 f8       	vpsignb %ymm8,%ymm8,%ymm15
  10db32:	c4 62 4d 08 e6       	vpsignb %ymm6,%ymm6,%ymm12
  10db37:	62 91 fd 28 6f c5    	vmovdqa64 %ymm29,%ymm0
  10db3d:	c4 42 7d 08 db       	vpsignb %ymm11,%ymm0,%ymm11
  10db42:	62 91 fd 28 6f c6    	vmovdqa64 %ymm30,%ymm0
  10db48:	c4 42 7d 08 d2       	vpsignb %ymm10,%ymm0,%ymm10
  10db4d:	62 b1 fd 28 6f c2    	vmovdqa64 %ymm18,%ymm0
  10db53:	62 52 7d 20 50 eb    	vpdpbusd %ymm11,%ymm16,%ymm13
  10db59:	c4 42 7d 08 d9       	vpsignb %ymm9,%ymm0,%ymm11
  10db5e:	62 b1 fd 28 6f c4    	vmovdqa64 %ymm20,%ymm0
  10db64:	c4 42 65 50 ea       	{vex} vpdpbusd %ymm10,%ymm3,%ymm13
  10db69:	c4 c2 35 08 d9       	vpsignb %ymm9,%ymm9,%ymm3
  10db6e:	62 31 7d 28 6f d7    	vmovdqa32 %ymm23,%ymm10
  10db74:	c4 42 65 50 d3       	{vex} vpdpbusd %ymm11,%ymm3,%ymm10
  10db79:	c4 42 7d 08 d8       	vpsignb %ymm8,%ymm0,%ymm11
  10db7e:	62 b1 fd 28 6f c5    	vmovdqa64 %ymm21,%ymm0
  10db84:	c4 42 05 50 d3       	{vex} vpdpbusd %ymm11,%ymm15,%ymm10
  10db89:	c4 62 7d 08 df       	vpsignb %ymm7,%ymm0,%ymm11
  10db8e:	62 b1 fd 28 6f c1    	vmovdqa64 %ymm17,%ymm0
  10db94:	62 61 7d 28 6f 74 24 	vmovdqa32 0x1e0(%rsp),%ymm30
  10db9b:	0f 
  10db9c:	62 e1 7d 28 6f 44 24 	vmovdqa32 0x240(%rsp),%ymm16
  10dba3:	12 
  10dba4:	c4 42 0d 50 d3       	{vex} vpdpbusd %ymm11,%ymm14,%ymm10
  10dba9:	c4 62 7d 08 de       	vpsignb %ymm6,%ymm0,%ymm11
  10dbae:	62 91 fd 28 6f c6    	vmovdqa64 %ymm30,%ymm0
  10dbb4:	c4 42 1d 50 d3       	{vex} vpdpbusd %ymm11,%ymm12,%ymm10
  10dbb9:	c4 42 7d 08 c9       	vpsignb %ymm9,%ymm0,%ymm9
  10dbbe:	62 31 7d 28 6f df    	vmovdqa32 %ymm23,%ymm11
  10dbc4:	c4 42 65 50 d9       	{vex} vpdpbusd %ymm9,%ymm3,%ymm11
  10dbc9:	c4 62 6d 08 ca       	vpsignb %ymm2,%ymm2,%ymm9
  10dbce:	62 b1 fd 28 6f d8    	vmovdqa64 %ymm16,%ymm3
  10dbd4:	c4 42 65 08 c0       	vpsignb %ymm8,%ymm3,%ymm8
  10dbd9:	62 b1 fd 28 6f d9    	vmovdqa64 %ymm17,%ymm3
  10dbdf:	c4 42 05 50 d8       	{vex} vpdpbusd %ymm8,%ymm15,%ymm11
  10dbe4:	c4 62 5d 08 c4       	vpsignb %ymm4,%ymm4,%ymm8
  10dbe9:	c5 7d 6f bc 24 c0 02 	vmovdqa 0x2c0(%rsp),%ymm15
  10dbf0:	00 00 
  10dbf2:	c5 fd 6f 84 24 40 03 	vmovdqa 0x340(%rsp),%ymm0
  10dbf9:	00 00 
  10dbfb:	c4 e2 05 08 ff       	vpsignb %ymm7,%ymm15,%ymm7
  10dc00:	c4 62 0d 50 df       	{vex} vpdpbusd %ymm7,%ymm14,%ymm11
  10dc05:	c4 e2 7d 08 f6       	vpsignb %ymm6,%ymm0,%ymm6
  10dc0a:	62 b1 7d 28 6f ff    	vmovdqa32 %ymm23,%ymm7
  10dc10:	c4 62 1d 50 de       	{vex} vpdpbusd %ymm6,%ymm12,%ymm11
  10dc15:	c4 62 75 08 e1       	vpsignb %ymm1,%ymm1,%ymm12
  10dc1a:	62 b1 fd 28 6f f2    	vmovdqa64 %ymm18,%ymm6
  10dc20:	c4 e2 4d 08 f4       	vpsignb %ymm4,%ymm6,%ymm6
  10dc25:	c4 e2 3d 50 fe       	{vex} vpdpbusd %ymm6,%ymm8,%ymm7
  10dc2a:	62 b1 fd 28 6f f4    	vmovdqa64 %ymm20,%ymm6
  10dc30:	62 51 1d 20 fe db    	vpaddd %ymm11,%ymm28,%ymm11
  10dc36:	c4 e2 4d 08 f2       	vpsignb %ymm2,%ymm6,%ymm6
  10dc3b:	c5 7d 6f b4 24 c0 01 	vmovdqa 0x1c0(%rsp),%ymm14
  10dc42:	00 00 
  10dc44:	c4 e2 35 50 fe       	{vex} vpdpbusd %ymm6,%ymm9,%ymm7
  10dc49:	62 b1 fd 28 6f f5    	vmovdqa64 %ymm21,%ymm6
  10dc4f:	c4 e2 4d 08 f1       	vpsignb %ymm1,%ymm6,%ymm6
  10dc54:	62 c1 7d 28 6f ce    	vmovdqa32 %ymm14,%ymm17
  10dc5a:	c4 e2 1d 50 fe       	{vex} vpdpbusd %ymm6,%ymm12,%ymm7
  10dc5f:	c4 c2 0d 08 f6       	vpsignb %ymm14,%ymm14,%ymm6
  10dc64:	c4 42 65 08 f6       	vpsignb %ymm14,%ymm3,%ymm14
  10dc69:	62 91 fd 28 6f de    	vmovdqa64 %ymm30,%ymm3
  10dc6f:	c4 e2 65 08 e4       	vpsignb %ymm4,%ymm3,%ymm4
  10dc74:	62 b1 7d 28 6f df    	vmovdqa32 %ymm23,%ymm3
  10dc7a:	c4 e2 3d 50 dc       	{vex} vpdpbusd %ymm4,%ymm8,%ymm3
  10dc7f:	62 b1 fd 28 6f e0    	vmovdqa64 %ymm16,%ymm4
  10dc85:	c4 e2 5d 08 e2       	vpsignb %ymm2,%ymm4,%ymm4
  10dc8a:	c5 fd 6f d3          	vmovdqa %ymm3,%ymm2
  10dc8e:	c4 e2 05 08 d9       	vpsignb %ymm1,%ymm15,%ymm3
  10dc93:	c4 e2 35 50 d4       	{vex} vpdpbusd %ymm4,%ymm9,%ymm2
  10dc98:	62 b1 fd 28 6f e1    	vmovdqa64 %ymm17,%ymm4
  10dc9e:	c4 e2 7d 08 c4       	vpsignb %ymm4,%ymm0,%ymm0
  10dca3:	c4 c1 7d 70 e3 4e    	vpshufd $0x4e,%ymm11,%ymm4
  10dca9:	c4 e2 1d 50 d3       	{vex} vpdpbusd %ymm3,%ymm12,%ymm2
  10dcae:	c5 fd 6f ca          	vmovdqa %ymm2,%ymm1
  10dcb2:	c5 ad fe 94 24 a0 01 	vpaddd 0x1a0(%rsp),%ymm10,%ymm2
  10dcb9:	00 00 
  10dcbb:	c4 e2 4d 50 c8       	{vex} vpdpbusd %ymm0,%ymm6,%ymm1
  10dcc0:	c4 c2 4d 50 fe       	{vex} vpdpbusd %ymm14,%ymm6,%ymm7
  10dcc5:	c5 15 fe e9          	vpaddd %ymm1,%ymm13,%ymm13
  10dcc9:	c4 e3 6d 02 e4 cc    	vpblendd $0xcc,%ymm4,%ymm2,%ymm4
  10dccf:	c5 fd 70 d2 4e       	vpshufd $0x4e,%ymm2,%ymm2
  10dcd4:	c4 c3 6d 02 d3 cc    	vpblendd $0xcc,%ymm11,%ymm2,%ymm2
  10dcda:	c4 c1 7d 70 dd 4e    	vpshufd $0x4e,%ymm13,%ymm3
  10dce0:	c5 fc 5b e4          	vcvtdq2ps %ymm4,%ymm4
  10dce4:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  10dce8:	62 f1 4d 20 fe c7    	vpaddd %ymm7,%ymm22,%ymm0
  10dcee:	c5 f9 6f b4 24 90 05 	vmovdqa 0x590(%rsp),%xmm6
  10dcf5:	00 00 
  10dcf7:	c4 e3 7d 02 db cc    	vpblendd $0xcc,%ymm3,%ymm0,%ymm3
  10dcfd:	c5 fd 70 c0 4e       	vpshufd $0x4e,%ymm0,%ymm0
  10dd02:	c4 43 7d 02 ed cc    	vpblendd $0xcc,%ymm13,%ymm0,%ymm13
  10dd08:	c4 e2 49 8c 81 78 ff 	vpmaskmovd -0x88(%rcx),%xmm6,%xmm0
  10dd0f:	ff ff 
  10dd11:	62 c1 7c 28 5b f5    	vcvtdq2ps %ymm13,%ymm22
  10dd17:	c5 f9 70 c0 44       	vpshufd $0x44,%xmm0,%xmm0
  10dd1c:	c4 e2 7d 13 c0       	vcvtph2ps %xmm0,%ymm0
  10dd21:	c4 e3 7d 04 f0 00    	vpermilps $0x0,%ymm0,%ymm6
  10dd27:	c5 cc 59 f5          	vmulps %ymm5,%ymm6,%ymm6
  10dd2b:	c4 e2 4d a8 a4 24 80 	vfmadd213ps 0x480(%rsp),%ymm6,%ymm4
  10dd32:	04 00 00 
  10dd35:	c5 fc 29 a4 24 80 04 	vmovaps %ymm4,0x480(%rsp)
  10dd3c:	00 00 
  10dd3e:	c4 e3 7d 04 e0 55    	vpermilps $0x55,%ymm0,%ymm4
  10dd44:	c5 dc 59 e5          	vmulps %ymm5,%ymm4,%ymm4
  10dd48:	c4 e2 5d a8 94 24 40 	vfmadd213ps 0x440(%rsp),%ymm4,%ymm2
  10dd4f:	04 00 00 
  10dd52:	c5 fc 29 94 24 40 04 	vmovaps %ymm2,0x440(%rsp)
  10dd59:	00 00 
  10dd5b:	c5 fc 5b d3          	vcvtdq2ps %ymm3,%ymm2
  10dd5f:	c4 e3 7d 04 d8 aa    	vpermilps $0xaa,%ymm0,%ymm3
  10dd65:	c4 e3 7d 04 c0 ff    	vpermilps $0xff,%ymm0,%ymm0
  10dd6b:	c5 e4 59 dd          	vmulps %ymm5,%ymm3,%ymm3
  10dd6f:	c5 fc 59 c5          	vmulps %ymm5,%ymm0,%ymm0
  10dd73:	c4 e2 65 a8 94 24 00 	vfmadd213ps 0x500(%rsp),%ymm3,%ymm2
  10dd7a:	05 00 00 
  10dd7d:	62 e2 7d 28 a8 74 24 	vfmadd213ps 0x4c0(%rsp),%ymm0,%ymm22
  10dd84:	26 
  10dd85:	c5 fc 29 94 24 00 05 	vmovaps %ymm2,0x500(%rsp)
  10dd8c:	00 00 
  10dd8e:	62 e1 7c 28 29 74 24 	vmovaps %ymm22,0x4c0(%rsp)
  10dd95:	26 
  10dd96:	49 39 f7             	cmp    %rsi,%r15
  10dd99:	0f 85 41 f9 ff ff    	jne    10d6e0 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x2990>
  10dd9f:	4c 8b 84 24 90 01 00 	mov    0x190(%rsp),%r8
  10dda6:	00 
  10dda7:	48 8b bc 24 88 01 00 	mov    0x188(%rsp),%rdi
  10ddae:	00 
  10ddaf:	4c 8b 8c 24 80 01 00 	mov    0x180(%rsp),%r9
  10ddb6:	00 
  10ddb7:	c5 fc 28 ea          	vmovaps %ymm2,%ymm5
  10ddbb:	c5 fc 28 9c 24 80 04 	vmovaps 0x480(%rsp),%ymm3
  10ddc2:	00 00 
  10ddc4:	c5 fc 28 8c 24 40 04 	vmovaps 0x440(%rsp),%ymm1
  10ddcb:	00 00 
  10ddcd:	48 8b 84 24 50 01 00 	mov    0x150(%rsp),%rax
  10ddd4:	00 
  10ddd5:	49 ff c0             	inc    %r8
  10ddd8:	c5 fc 28 94 24 c0 04 	vmovaps 0x4c0(%rsp),%ymm2
  10dddf:	00 00 
  10dde1:	c5 fc 11 1f          	vmovups %ymm3,(%rdi)
  10dde5:	c5 fc 11 0c 87       	vmovups %ymm1,(%rdi,%rax,4)
  10ddea:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  10ddf1:	00 
  10ddf2:	c4 a1 7c 11 2c a7    	vmovups %ymm5,(%rdi,%r12,4)
  10ddf8:	c5 fc 11 14 87       	vmovups %ymm2,(%rdi,%rax,4)
  10ddfd:	48 8b 84 24 70 01 00 	mov    0x170(%rsp),%rax
  10de04:	00 
  10de05:	48 83 c7 20          	add    $0x20,%rdi
  10de09:	49 01 c1             	add    %rax,%r9
  10de0c:	4c 39 84 24 58 01 00 	cmp    %r8,0x158(%rsp)
  10de13:	00 
  10de14:	0f 85 56 f8 ff ff    	jne    10d670 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x2920>
  10de1a:	48 8b 84 24 18 01 00 	mov    0x118(%rsp),%rax
  10de21:	00 
  10de22:	48 8b 94 24 40 01 00 	mov    0x140(%rsp),%rdx
  10de29:	00 
  10de2a:	48 8b b4 24 38 01 00 	mov    0x138(%rsp),%rsi
  10de31:	00 
  10de32:	48 8b 8c 24 48 01 00 	mov    0x148(%rsp),%rcx
  10de39:	00 
  10de3a:	4c 8b ac 24 68 01 00 	mov    0x168(%rsp),%r13
  10de41:	00 
  10de42:	48 01 c2             	add    %rax,%rdx
  10de45:	48 8b 84 24 28 01 00 	mov    0x128(%rsp),%rax
  10de4c:	00 
  10de4d:	48 ff c1             	inc    %rcx
  10de50:	48 01 c6             	add    %rax,%rsi
  10de53:	48 8b 84 24 70 01 00 	mov    0x170(%rsp),%rax
  10de5a:	00 
  10de5b:	49 01 c5             	add    %rax,%r13
  10de5e:	48 39 8c 24 d0 00 00 	cmp    %rcx,0xd0(%rsp)
  10de65:	00 
  10de66:	0f 85 a0 f7 ff ff    	jne    10d60c <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x28bc>
  10de6c:	48 8b 84 24 f8 09 00 	mov    0x9f8(%rsp),%rax
  10de73:	00 
  10de74:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
  10de7b:	00 00 
  10de7d:	0f 85 d0 01 00 00    	jne    10e053 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x3303>
  10de83:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  10de87:	5b                   	pop    %rbx
  10de88:	41 5c                	pop    %r12
  10de8a:	41 5d                	pop    %r13
  10de8c:	41 5e                	pop    %r14
  10de8e:	41 5f                	pop    %r15
  10de90:	5d                   	pop    %rbp
  10de91:	c3                   	ret
  10de92:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
  10de96:	c5 fc 29 bc 24 c0 04 	vmovaps %ymm7,0x4c0(%rsp)
  10de9d:	00 00 
  10de9f:	c5 fc 28 ef          	vmovaps %ymm7,%ymm5
  10dea3:	c5 fc 29 bc 24 00 05 	vmovaps %ymm7,0x500(%rsp)
  10deaa:	00 00 
  10deac:	c5 fc 29 bc 24 40 04 	vmovaps %ymm7,0x440(%rsp)
  10deb3:	00 00 
  10deb5:	c5 fc 29 bc 24 80 04 	vmovaps %ymm7,0x480(%rsp)
  10debc:	00 00 
  10debe:	e9 f8 fe ff ff       	jmp    10ddbb <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x306b>
  10dec3:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  10dec7:	c5 fc 28 c8          	vmovaps %ymm0,%ymm1
  10decb:	c5 fc 28 d0          	vmovaps %ymm0,%ymm2
  10decf:	c5 fc 28 e0          	vmovaps %ymm0,%ymm4
  10ded3:	c5 fc 28 f0          	vmovaps %ymm0,%ymm6
  10ded7:	c5 fc 28 f8          	vmovaps %ymm0,%ymm7
  10dedb:	c5 7c 28 c0          	vmovaps %ymm0,%ymm8
  10dedf:	c5 7c 28 c8          	vmovaps %ymm0,%ymm9
  10dee3:	c5 7c 28 d8          	vmovaps %ymm0,%ymm11
  10dee7:	c5 7c 28 e0          	vmovaps %ymm0,%ymm12
  10deeb:	c5 7c 28 e8          	vmovaps %ymm0,%ymm13
  10deef:	c5 7c 28 f0          	vmovaps %ymm0,%ymm14
  10def3:	c5 7c 28 f8          	vmovaps %ymm0,%ymm15
  10def7:	62 e1 7c 28 28 c0    	vmovaps %ymm0,%ymm16
  10defd:	62 e1 7c 28 28 c8    	vmovaps %ymm0,%ymm17
  10df03:	62 e1 7c 28 28 d0    	vmovaps %ymm0,%ymm18
  10df09:	e9 51 f4 ff ff       	jmp    10d35f <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x260f>
  10df0e:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
  10df12:	62 f1 7c 48 29 6c 24 	vmovaps %zmm5,0x4c0(%rsp)
  10df19:	13 
  10df1a:	62 f1 7c 48 28 cd    	vmovaps %zmm5,%zmm1
  10df20:	62 f1 7c 48 28 54 24 	vmovaps 0x4c0(%rsp),%zmm2
  10df27:	13 
  10df28:	62 f1 7c 48 29 6c 24 	vmovaps %zmm5,0x500(%rsp)
  10df2f:	14 
  10df30:	62 f1 7c 48 29 6c 24 	vmovaps %zmm5,0x540(%rsp)
  10df37:	15 
  10df38:	62 f1 7c 48 29 6c 24 	vmovaps %zmm5,0x480(%rsp)
  10df3f:	12 
  10df40:	62 f1 7c 48 28 6c 24 	vmovaps 0x540(%rsp),%zmm5
  10df47:	15 
  10df48:	e9 58 e8 ff ff       	jmp    10c7a5 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1a55>
  10df4d:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  10df51:	62 f1 7c 48 28 c8    	vmovaps %zmm0,%zmm1
  10df57:	62 f1 7c 48 28 d0    	vmovaps %zmm0,%zmm2
  10df5d:	62 f1 7c 48 28 d8    	vmovaps %zmm0,%zmm3
  10df63:	62 f1 7c 48 28 e0    	vmovaps %zmm0,%zmm4
  10df69:	62 f1 7c 48 28 e8    	vmovaps %zmm0,%zmm5
  10df6f:	62 f1 7c 48 28 f0    	vmovaps %zmm0,%zmm6
  10df75:	62 f1 7c 48 28 f8    	vmovaps %zmm0,%zmm7
  10df7b:	62 71 7c 48 28 c0    	vmovaps %zmm0,%zmm8
  10df81:	62 71 7c 48 28 c8    	vmovaps %zmm0,%zmm9
  10df87:	62 71 7c 48 28 d0    	vmovaps %zmm0,%zmm10
  10df8d:	62 71 7c 48 28 d8    	vmovaps %zmm0,%zmm11
  10df93:	62 71 7c 48 28 e8    	vmovaps %zmm0,%zmm13
  10df99:	62 71 7c 48 28 f8    	vmovaps %zmm0,%zmm15
  10df9f:	62 e1 7c 48 28 c8    	vmovaps %zmm0,%zmm17
  10dfa5:	62 e1 7c 48 28 d0    	vmovaps %zmm0,%zmm18
  10dfab:	e9 0e db ff ff       	jmp    10babe <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0xd6e>
  10dfb0:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  10dfb7:	00 
  10dfb8:	48 c7 84 24 98 01 00 	movq   $0x0,0x198(%rsp)
  10dfbf:	00 00 00 00 00 
  10dfc4:	48 39 c1             	cmp    %rax,%rcx
  10dfc7:	0f 8c d3 e8 ff ff    	jl     10c8a0 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1b50>
  10dfcd:	e9 9a fe ff ff       	jmp    10de6c <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x311c>
  10dfd2:	41 39 f9             	cmp    %edi,%r9d
  10dfd5:	74 48                	je     10e01f <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x32cf>
  10dfd7:	48 8b 84 24 78 01 00 	mov    0x178(%rsp),%rax
  10dfde:	00 
  10dfdf:	31 c9                	xor    %ecx,%ecx
  10dfe1:	48 8d 04 85 00 00 00 	lea    0x0(,%rax,4),%rax
  10dfe8:	00 
  10dfe9:	c4 61 f9 6e d0       	vmovq  %rax,%xmm10
  10dfee:	e9 ad e8 ff ff       	jmp    10c8a0 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1b50>
  10dff3:	45 85 c0             	test   %r8d,%r8d
  10dff6:	41 8d 40 03          	lea    0x3(%r8),%eax
  10dffa:	41 0f 49 c0          	cmovns %r8d,%eax
  10dffe:	c1 f8 02             	sar    $0x2,%eax
  10e001:	48 98                	cltq
  10e003:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  10e00a:	00 
  10e00b:	41 83 f8 03          	cmp    $0x3,%r8d
  10e00f:	0f 8e 57 fe ff ff    	jle    10de6c <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x311c>
  10e015:	31 c9                	xor    %ecx,%ecx
  10e017:	8d 5f 07             	lea    0x7(%rdi),%ebx
  10e01a:	e9 b0 dc ff ff       	jmp    10bccf <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0xf7f>
  10e01f:	48 8b 84 24 78 01 00 	mov    0x178(%rsp),%rax
  10e026:	00 
  10e027:	48 c7 84 24 98 01 00 	movq   $0x0,0x198(%rsp)
  10e02e:	00 00 00 00 00 
  10e033:	48 8d 04 85 00 00 00 	lea    0x0(,%rax,4),%rax
  10e03a:	00 
  10e03b:	c4 61 f9 6e d0       	vmovq  %rax,%xmm10
  10e040:	48 39 8c 24 d8 00 00 	cmp    %rcx,0xd8(%rsp)
  10e047:	00 
  10e048:	0f 8f 52 e8 ff ff    	jg     10c8a0 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x1b50>
  10e04e:	e9 b5 f4 ff ff       	jmp    10d508 <_Z28gemm_q4_b32_8x8_q8_0_lut_avxI13block_mxfp4x8EviPfmPKvS3_iiDv4_x+0x27b8>
  10e053:	c5 f8 77             	vzeroupper
  10e056:	e8 a5 78 f0 ff       	call   15900 <__stack_chk_fail@plt>
  10e05b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
