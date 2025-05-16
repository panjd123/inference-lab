	.file	"main.cpp"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB8898:
	.cfi_startproc
	endbr64
	movl	%esi, %eax
	ret
	.cfi_endproc
.LFE8898:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.section	.text._Z25native_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z25native_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z25native_cube_root_templateILi2EEvPKfPfi
	.type	_Z25native_cube_root_templateILi2EEvPKfPfi, @function
_Z25native_cube_root_templateILi2EEvPKfPfi:
.LFB12739:
	.cfi_startproc
	endbr64
	movq	%rsi, %rcx
	testl	%edx, %edx
	jle	.L23
	movslq	%edx, %rdx
	vmovss	.LC0(%rip), %xmm0
	xorl	%eax, %eax
	salq	$2, %rdx
	leaq	-4(%rdx), %rsi
	shrq	$2, %rsi
	addq	$1, %rsi
	andl	$3, %esi
	je	.L5
	cmpq	$1, %rsi
	je	.L17
	cmpq	$2, %rsi
	je	.L18
	vmovss	(%rdi), %xmm2
	movl	$4, %eax
	vmulss	%xmm0, %xmm2, %xmm3
	vmulss	%xmm3, %xmm3, %xmm1
	vaddss	%xmm3, %xmm3, %xmm5
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm5, %xmm4, %xmm6
	vmulss	%xmm0, %xmm6, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm2, %xmm9
	vaddss	%xmm10, %xmm9, %xmm11
	vmulss	%xmm0, %xmm11, %xmm12
	vmovss	%xmm12, (%rcx)
.L18:
	vmovss	(%rdi,%rax), %xmm13
	vmulss	%xmm0, %xmm13, %xmm14
	vmulss	%xmm14, %xmm14, %xmm15
	vaddss	%xmm14, %xmm14, %xmm3
	vdivss	%xmm15, %xmm13, %xmm2
	vaddss	%xmm3, %xmm2, %xmm1
	vmulss	%xmm0, %xmm1, %xmm4
	vmulss	%xmm4, %xmm4, %xmm5
	vaddss	%xmm4, %xmm4, %xmm7
	vdivss	%xmm5, %xmm13, %xmm6
	vaddss	%xmm7, %xmm6, %xmm8
	vmulss	%xmm0, %xmm8, %xmm9
	vmovss	%xmm9, (%rcx,%rax)
	addq	$4, %rax
.L17:
	vmovss	(%rdi,%rax), %xmm10
	vmulss	%xmm0, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm10, %xmm13
	vaddss	%xmm14, %xmm13, %xmm15
	vmulss	%xmm0, %xmm15, %xmm1
	vmulss	%xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm1, %xmm4
	vdivss	%xmm2, %xmm10, %xmm3
	vaddss	%xmm4, %xmm3, %xmm5
	vmulss	%xmm0, %xmm5, %xmm6
	vmovss	%xmm6, (%rcx,%rax)
	addq	$4, %rax
	cmpq	%rax, %rdx
	je	.L24
.L5:
	vmovss	(%rdi,%rax), %xmm7
	vmulss	%xmm0, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm9, %xmm7, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm0, %xmm12, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm1
	vdivss	%xmm14, %xmm7, %xmm15
	vaddss	%xmm1, %xmm15, %xmm2
	vmulss	%xmm0, %xmm2, %xmm3
	vmovss	%xmm3, (%rcx,%rax)
	vmovss	4(%rdi,%rax), %xmm4
	vmulss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm5, %xmm5, %xmm6
	vaddss	%xmm5, %xmm5, %xmm8
	vdivss	%xmm6, %xmm4, %xmm7
	vaddss	%xmm8, %xmm7, %xmm9
	vmulss	%xmm0, %xmm9, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm13
	vdivss	%xmm11, %xmm4, %xmm12
	vaddss	%xmm13, %xmm12, %xmm14
	vmulss	%xmm0, %xmm14, %xmm15
	vmovss	%xmm15, 4(%rcx,%rax)
	vmovss	8(%rdi,%rax), %xmm2
	vmulss	%xmm0, %xmm2, %xmm3
	vmulss	%xmm3, %xmm3, %xmm1
	vaddss	%xmm3, %xmm3, %xmm5
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm5, %xmm4, %xmm6
	vmulss	%xmm0, %xmm6, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm2, %xmm9
	vaddss	%xmm10, %xmm9, %xmm11
	vmulss	%xmm0, %xmm11, %xmm12
	vmovss	%xmm12, 8(%rcx,%rax)
	vmovss	12(%rdi,%rax), %xmm13
	vmulss	%xmm0, %xmm13, %xmm14
	vmulss	%xmm14, %xmm14, %xmm15
	vaddss	%xmm14, %xmm14, %xmm3
	vdivss	%xmm15, %xmm13, %xmm2
	vaddss	%xmm3, %xmm2, %xmm1
	vmulss	%xmm0, %xmm1, %xmm4
	vmulss	%xmm4, %xmm4, %xmm5
	vaddss	%xmm4, %xmm4, %xmm7
	vdivss	%xmm5, %xmm13, %xmm6
	vaddss	%xmm7, %xmm6, %xmm8
	vmulss	%xmm0, %xmm8, %xmm9
	vmovss	%xmm9, 12(%rcx,%rax)
	addq	$16, %rax
	cmpq	%rax, %rdx
	jne	.L5
.L23:
	ret
.L24:
	ret
	.cfi_endproc
.LFE12739:
	.size	_Z25native_cube_root_templateILi2EEvPKfPfi, .-_Z25native_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z22opt_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z22opt_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z22opt_cube_root_templateILi2EEvPKfPfi
	.type	_Z22opt_cube_root_templateILi2EEvPKfPfi, @function
_Z22opt_cube_root_templateILi2EEvPKfPfi:
.LFB12740:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	movq	%rdi, %rcx
	testl	%eax, %eax
	jle	.L66
	leal	-1(%rax), %edx
	movl	%eax, %r10d
	cmpl	$2, %edx
	jbe	.L27
	leaq	4(%rdi), %r9
	movq	%rsi, %rdi
	subq	%r9, %rdi
	cmpq	$24, %rdi
	ja	.L67
.L27:
	leaq	0(,%rax,4), %r9
	vmovss	.LC0(%rip), %xmm5
	xorl	%eax, %eax
	leaq	-4(%r9), %rdx
	shrq	$2, %rdx
	addq	$1, %rdx
	andl	$3, %edx
	je	.L35
	cmpq	$1, %rdx
	je	.L54
	cmpq	$2, %rdx
	jne	.L68
.L55:
	vmovss	(%rcx,%rax), %xmm0
	vmulss	%xmm5, %xmm0, %xmm15
	vmulss	%xmm15, %xmm15, %xmm14
	vaddss	%xmm15, %xmm15, %xmm10
	vdivss	%xmm14, %xmm0, %xmm6
	vaddss	%xmm10, %xmm6, %xmm1
	vmulss	%xmm5, %xmm1, %xmm11
	vmulss	%xmm11, %xmm11, %xmm2
	vaddss	%xmm11, %xmm11, %xmm8
	vdivss	%xmm2, %xmm0, %xmm7
	vaddss	%xmm8, %xmm7, %xmm12
	vmulss	%xmm5, %xmm12, %xmm9
	vmovss	%xmm9, (%rsi,%rax)
	addq	$4, %rax
.L54:
	vmovss	(%rcx,%rax), %xmm13
	vmulss	%xmm5, %xmm13, %xmm4
	vmulss	%xmm4, %xmm4, %xmm3
	vaddss	%xmm4, %xmm4, %xmm15
	vdivss	%xmm3, %xmm13, %xmm0
	vaddss	%xmm15, %xmm0, %xmm14
	vmulss	%xmm5, %xmm14, %xmm6
	vmulss	%xmm6, %xmm6, %xmm10
	vaddss	%xmm6, %xmm6, %xmm1
	vdivss	%xmm10, %xmm13, %xmm11
	vaddss	%xmm1, %xmm11, %xmm2
	vmulss	%xmm5, %xmm2, %xmm7
	vmovss	%xmm7, (%rsi,%rax)
	addq	$4, %rax
	cmpq	%r9, %rax
	je	.L69
.L35:
	vmovss	(%rcx,%rax), %xmm8
	vmulss	%xmm5, %xmm8, %xmm12
	vmulss	%xmm12, %xmm12, %xmm9
	vaddss	%xmm12, %xmm12, %xmm4
	vdivss	%xmm9, %xmm8, %xmm13
	vaddss	%xmm4, %xmm13, %xmm3
	vmulss	%xmm5, %xmm3, %xmm0
	vmulss	%xmm0, %xmm0, %xmm15
	vaddss	%xmm0, %xmm0, %xmm6
	vdivss	%xmm15, %xmm8, %xmm14
	vaddss	%xmm6, %xmm14, %xmm10
	vmulss	%xmm5, %xmm10, %xmm11
	vmovss	%xmm11, (%rsi,%rax)
	vmovss	4(%rcx,%rax), %xmm2
	vmulss	%xmm5, %xmm2, %xmm7
	vmulss	%xmm7, %xmm7, %xmm1
	vaddss	%xmm7, %xmm7, %xmm12
	vdivss	%xmm1, %xmm2, %xmm8
	vaddss	%xmm12, %xmm8, %xmm9
	vmulss	%xmm5, %xmm9, %xmm13
	vmulss	%xmm13, %xmm13, %xmm4
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm4, %xmm2, %xmm3
	vaddss	%xmm0, %xmm3, %xmm15
	vmulss	%xmm5, %xmm15, %xmm14
	vmovss	%xmm14, 4(%rsi,%rax)
	vmovss	8(%rcx,%rax), %xmm6
	vmulss	%xmm5, %xmm6, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm7
	vdivss	%xmm11, %xmm6, %xmm2
	vaddss	%xmm7, %xmm2, %xmm1
	vmulss	%xmm5, %xmm1, %xmm8
	vmulss	%xmm8, %xmm8, %xmm12
	vaddss	%xmm8, %xmm8, %xmm13
	vdivss	%xmm12, %xmm6, %xmm9
	vaddss	%xmm13, %xmm9, %xmm4
	vmulss	%xmm5, %xmm4, %xmm3
	vmovss	%xmm3, 8(%rsi,%rax)
	vmovss	12(%rcx,%rax), %xmm0
	vmulss	%xmm5, %xmm0, %xmm15
	vmulss	%xmm15, %xmm15, %xmm14
	vaddss	%xmm15, %xmm15, %xmm10
	vdivss	%xmm14, %xmm0, %xmm6
	vaddss	%xmm10, %xmm6, %xmm11
	vmulss	%xmm5, %xmm11, %xmm7
	vmulss	%xmm7, %xmm7, %xmm2
	vaddss	%xmm7, %xmm7, %xmm1
	vdivss	%xmm2, %xmm0, %xmm8
	vaddss	%xmm1, %xmm8, %xmm12
	vmulss	%xmm5, %xmm12, %xmm9
	vmovss	%xmm9, 12(%rsi,%rax)
	addq	$16, %rax
	cmpq	%r9, %rax
	jne	.L35
	ret
.L64:
	vzeroupper
.L66:
	ret
	.p2align 4,,10
	.p2align 3
.L67:
	cmpl	$6, %edx
	jbe	.L37
	vbroadcastss	.LC0(%rip), %ymm2
	shrl	$3, %r10d
	xorl	%r11d, %r11d
	movq	%r10, %r8
	salq	$5, %r8
	andl	$1, %r10d
	je	.L29
	vmovups	(%rcx), %ymm1
	movl	$32, %r11d
	vmulps	%ymm2, %ymm1, %ymm3
	vmulps	%ymm3, %ymm3, %ymm4
	vaddps	%ymm3, %ymm3, %ymm10
	vrcpps	%ymm4, %ymm0
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm7
	vmulps	%ymm5, %ymm0, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm1, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm5
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm3
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm3, %ymm0
	vmulps	%ymm0, %ymm1, %ymm1
	vaddps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm2, %ymm6, %ymm7
	vmovups	%ymm7, (%rsi)
	cmpq	$32, %r8
	je	.L63
	.p2align 4,,10
	.p2align 3
.L29:
	vmovups	(%rcx,%r11), %ymm8
	vmulps	%ymm2, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm3
	vrcpps	%ymm10, %ymm11
	vmulps	%ymm10, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm14
	vmulps	%ymm12, %ymm11, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm8, %ymm4
	vaddps	%ymm3, %ymm4, %ymm0
	vmulps	%ymm2, %ymm0, %ymm5
	vmulps	%ymm5, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm12
	vrcpps	%ymm1, %ymm6
	vmulps	%ymm1, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm10
	vmulps	%ymm7, %ymm6, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm8, %ymm8
	vaddps	%ymm12, %ymm8, %ymm13
	vmulps	%ymm2, %ymm13, %ymm14
	vmovups	%ymm14, (%rsi,%r11)
	vmovups	32(%rcx,%r11), %ymm15
	vmulps	%ymm2, %ymm15, %ymm3
	vmulps	%ymm3, %ymm3, %ymm4
	vaddps	%ymm3, %ymm3, %ymm10
	vrcpps	%ymm4, %ymm0
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm6
	vmulps	%ymm5, %ymm0, %ymm1
	vsubps	%ymm1, %ymm6, %ymm7
	vmulps	%ymm7, %ymm15, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm2, %ymm11, %ymm8
	vmulps	%ymm8, %ymm8, %ymm12
	vaddps	%ymm8, %ymm8, %ymm5
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm3
	vmulps	%ymm14, %ymm13, %ymm4
	vsubps	%ymm4, %ymm3, %ymm0
	vmulps	%ymm0, %ymm15, %ymm15
	vaddps	%ymm5, %ymm15, %ymm1
	vmulps	%ymm2, %ymm1, %ymm6
	vmovups	%ymm6, 32(%rsi,%r11)
	addq	$64, %r11
	cmpq	%r11, %r8
	jne	.L29
.L63:
	movl	%eax, %edx
	andl	$-8, %edx
	movl	%edx, %edi
	cmpl	%edx, %eax
	je	.L64
	movl	%eax, %r10d
	subl	%edx, %r10d
	leal	-1(%r10), %r9d
	cmpl	$2, %r9d
	jbe	.L70
	vzeroupper
.L28:
	vmovups	(%rcx,%rdi,4), %xmm7
	movl	%r10d, %r8d
	vbroadcastss	.LC0(%rip), %xmm9
	andl	$-4, %r8d
	vmulps	%xmm9, %xmm7, %xmm2
	addl	%r8d, %edx
	andl	$3, %r10d
	vmulps	%xmm2, %xmm2, %xmm10
	vaddps	%xmm2, %xmm2, %xmm3
	vrcpps	%xmm10, %xmm11
	vmulps	%xmm10, %xmm11, %xmm8
	vaddps	%xmm11, %xmm11, %xmm13
	vmulps	%xmm8, %xmm11, %xmm12
	vsubps	%xmm12, %xmm13, %xmm14
	vmulps	%xmm14, %xmm7, %xmm4
	vaddps	%xmm3, %xmm4, %xmm0
	vmulps	%xmm9, %xmm0, %xmm15
	vmulps	%xmm15, %xmm15, %xmm5
	vaddps	%xmm15, %xmm15, %xmm8
	vrcpps	%xmm5, %xmm1
	vmulps	%xmm5, %xmm1, %xmm6
	vaddps	%xmm1, %xmm1, %xmm2
	vmulps	%xmm6, %xmm1, %xmm10
	vsubps	%xmm10, %xmm2, %xmm11
	vmulps	%xmm11, %xmm7, %xmm7
	vaddps	%xmm8, %xmm7, %xmm12
	vmulps	%xmm9, %xmm12, %xmm9
	vmovups	%xmm9, (%rsi,%rdi,4)
	je	.L66
.L33:
	movslq	%edx, %r11
	vmovss	.LC0(%rip), %xmm14
	leal	1(%rdx), %r10d
	vmovss	(%rcx,%r11,4), %xmm13
	leaq	0(,%r11,4), %rdi
	vmulss	%xmm14, %xmm13, %xmm4
	vmulss	%xmm4, %xmm4, %xmm3
	vaddss	%xmm4, %xmm4, %xmm15
	vdivss	%xmm3, %xmm13, %xmm0
	vaddss	%xmm15, %xmm0, %xmm5
	vmulss	%xmm14, %xmm5, %xmm6
	vmulss	%xmm6, %xmm6, %xmm1
	vaddss	%xmm6, %xmm6, %xmm2
	vdivss	%xmm1, %xmm13, %xmm10
	vaddss	%xmm2, %xmm10, %xmm11
	vmulss	%xmm14, %xmm11, %xmm7
	vmovss	%xmm7, (%rsi,%r11,4)
	cmpl	%r10d, %eax
	jle	.L66
	vmovss	4(%rcx,%rdi), %xmm8
	addl	$2, %edx
	vmulss	%xmm14, %xmm8, %xmm12
	vmulss	%xmm12, %xmm12, %xmm9
	vaddss	%xmm12, %xmm12, %xmm4
	vdivss	%xmm9, %xmm8, %xmm13
	vaddss	%xmm4, %xmm13, %xmm3
	vmulss	%xmm14, %xmm3, %xmm0
	vmulss	%xmm0, %xmm0, %xmm15
	vaddss	%xmm0, %xmm0, %xmm6
	vdivss	%xmm15, %xmm8, %xmm5
	vaddss	%xmm6, %xmm5, %xmm1
	vmulss	%xmm14, %xmm1, %xmm10
	vmovss	%xmm10, 4(%rsi,%rdi)
	cmpl	%edx, %eax
	jle	.L66
	vmovss	8(%rcx,%rdi), %xmm2
	vmulss	%xmm14, %xmm2, %xmm11
	vmulss	%xmm11, %xmm11, %xmm7
	vaddss	%xmm11, %xmm11, %xmm12
	vdivss	%xmm7, %xmm2, %xmm8
	vaddss	%xmm12, %xmm8, %xmm9
	vmulss	%xmm14, %xmm9, %xmm13
	vmulss	%xmm13, %xmm13, %xmm4
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm4, %xmm2, %xmm3
	vaddss	%xmm0, %xmm3, %xmm15
	vmulss	%xmm14, %xmm15, %xmm14
	vmovss	%xmm14, 8(%rsi,%rdi)
	ret
.L68:
	vmovss	(%rcx), %xmm6
	movl	$4, %eax
	vmulss	%xmm5, %xmm6, %xmm10
	vmulss	%xmm10, %xmm10, %xmm1
	vaddss	%xmm10, %xmm10, %xmm11
	vdivss	%xmm1, %xmm6, %xmm2
	vaddss	%xmm11, %xmm2, %xmm7
	vmulss	%xmm5, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm12
	vaddss	%xmm8, %xmm8, %xmm13
	vdivss	%xmm12, %xmm6, %xmm9
	vaddss	%xmm13, %xmm9, %xmm4
	vmulss	%xmm5, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi)
	jmp	.L55
.L69:
	ret
.L37:
	xorl	%edi, %edi
	xorl	%edx, %edx
	jmp	.L28
.L70:
	vzeroupper
	jmp	.L33
	.cfi_endproc
.LFE12740:
	.size	_Z22opt_cube_root_templateILi2EEvPKfPfi, .-_Z22opt_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z23avx2_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z23avx2_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z23avx2_cube_root_templateILi2EEvPKfPfi
	.type	_Z23avx2_cube_root_templateILi2EEvPKfPfi, @function
_Z23avx2_cube_root_templateILi2EEvPKfPfi:
.LFB12741:
	.cfi_startproc
	endbr64
	leal	14(%rdx), %eax
	addl	$7, %edx
	movq	%rdi, %rcx
	cmovs	%eax, %edx
	andl	$-8, %edx
	jle	.L81
	vmovups	(%rcx), %ymm1
	leal	-1(%rdx), %edi
	movl	$8, %r8d
	vbroadcastss	.LC0(%rip), %ymm2
	shrl	$3, %edi
	vmulps	%ymm2, %ymm1, %ymm3
	andl	$1, %edi
	vmulps	%ymm3, %ymm3, %ymm4
	vaddps	%ymm3, %ymm3, %ymm10
	vrcpps	%ymm4, %ymm0
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm7
	vmulps	%ymm5, %ymm0, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm1, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm5
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm3
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm3, %ymm0
	vmulps	%ymm0, %ymm1, %ymm1
	vaddps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm2, %ymm6, %ymm7
	vmovups	%ymm7, (%rsi)
	cmpl	$8, %edx
	jle	.L82
	testl	%edi, %edi
	je	.L73
	vmovups	32(%rcx), %ymm8
	movl	$16, %r8d
	vmulps	%ymm2, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm3
	vrcpps	%ymm10, %ymm11
	vmulps	%ymm10, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm14
	vmulps	%ymm12, %ymm11, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm8, %ymm4
	vaddps	%ymm3, %ymm4, %ymm0
	vmulps	%ymm2, %ymm0, %ymm5
	vmulps	%ymm5, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm12
	vrcpps	%ymm1, %ymm6
	vmulps	%ymm1, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm10
	vmulps	%ymm7, %ymm6, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm8, %ymm8
	vaddps	%ymm12, %ymm8, %ymm13
	vmulps	%ymm2, %ymm13, %ymm14
	vmovups	%ymm14, 32(%rsi)
	cmpl	$16, %edx
	jle	.L82
	.p2align 4,,10
	.p2align 3
.L73:
	vmovups	(%rcx,%r8,4), %ymm15
	leaq	8(%r8), %r9
	vmulps	%ymm2, %ymm15, %ymm3
	vmulps	%ymm3, %ymm3, %ymm4
	vaddps	%ymm3, %ymm3, %ymm10
	vrcpps	%ymm4, %ymm0
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm6
	vmulps	%ymm5, %ymm0, %ymm1
	vsubps	%ymm1, %ymm6, %ymm7
	vmulps	%ymm7, %ymm15, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm2, %ymm11, %ymm8
	vmulps	%ymm8, %ymm8, %ymm12
	vaddps	%ymm8, %ymm8, %ymm5
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm3
	vmulps	%ymm14, %ymm13, %ymm4
	vsubps	%ymm4, %ymm3, %ymm0
	vmulps	%ymm0, %ymm15, %ymm15
	vaddps	%ymm5, %ymm15, %ymm1
	vmulps	%ymm2, %ymm1, %ymm6
	vmovups	%ymm6, (%rsi,%r8,4)
	vmovups	(%rcx,%r9,4), %ymm7
	addq	$16, %r8
	vmulps	%ymm2, %ymm7, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm3
	vrcpps	%ymm10, %ymm11
	vmulps	%ymm10, %ymm11, %ymm8
	vaddps	%ymm11, %ymm11, %ymm13
	vmulps	%ymm8, %ymm11, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm7, %ymm4
	vaddps	%ymm3, %ymm4, %ymm0
	vmulps	%ymm2, %ymm0, %ymm15
	vmulps	%ymm15, %ymm15, %ymm5
	vaddps	%ymm15, %ymm15, %ymm8
	vrcpps	%ymm5, %ymm1
	vmulps	%ymm5, %ymm1, %ymm6
	vaddps	%ymm1, %ymm1, %ymm10
	vmulps	%ymm6, %ymm1, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm7, %ymm7
	vaddps	%ymm8, %ymm7, %ymm12
	vmulps	%ymm2, %ymm12, %ymm13
	vmovups	%ymm13, (%rsi,%r9,4)
	cmpl	%r8d, %edx
	jg	.L73
.L82:
	vzeroupper
.L81:
	ret
	.cfi_endproc
.LFE12741:
	.size	_Z23avx2_cube_root_templateILi2EEvPKfPfi, .-_Z23avx2_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z25native_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z25native_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z25native_cube_root_templateILi16EEvPKfPfi
	.type	_Z25native_cube_root_templateILi16EEvPKfPfi, @function
_Z25native_cube_root_templateILi16EEvPKfPfi:
.LFB12771:
	.cfi_startproc
	endbr64
	testl	%edx, %edx
	jle	.L90
	movslq	%edx, %rdx
	vmovss	.LC0(%rip), %xmm3
	xorl	%r8d, %r8d
	leaq	0(,%rdx,4), %rcx
	.p2align 4,,10
	.p2align 3
.L86:
	vmovss	(%rdi,%r8), %xmm2
	movl	$16, %eax
	vmulss	%xmm3, %xmm2, %xmm6
	.p2align 4,,10
	.p2align 3
.L85:
	vmulss	%xmm6, %xmm6, %xmm1
	vaddss	%xmm6, %xmm6, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	subl	$1, %eax
	jne	.L85
	vmovss	%xmm6, (%rsi,%r8)
	addq	$4, %r8
	cmpq	%r8, %rcx
	jne	.L86
.L90:
	ret
	.cfi_endproc
.LFE12771:
	.size	_Z25native_cube_root_templateILi16EEvPKfPfi, .-_Z25native_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z22opt_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z22opt_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z22opt_cube_root_templateILi16EEvPKfPfi
	.type	_Z22opt_cube_root_templateILi16EEvPKfPfi, @function
_Z22opt_cube_root_templateILi16EEvPKfPfi:
.LFB12772:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	movq	%rdi, %rcx
	testl	%eax, %eax
	jle	.L111
	leal	-1(%rax), %edx
	movl	%eax, %r8d
	cmpl	$2, %edx
	jbe	.L93
	leaq	4(%rdi), %r9
	movq	%rsi, %rdi
	subq	%r9, %rdi
	cmpq	$24, %rdi
	ja	.L112
.L93:
	vmovss	.LC0(%rip), %xmm1
	leaq	0(,%rax,4), %r11
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L101:
	vmovss	(%rcx,%rax), %xmm9
	vmulss	%xmm1, %xmm9, %xmm8
	vmulss	%xmm8, %xmm8, %xmm2
	vaddss	%xmm8, %xmm8, %xmm10
	vdivss	%xmm2, %xmm9, %xmm4
	vaddss	%xmm10, %xmm4, %xmm11
	vmulss	%xmm1, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm6
	vdivss	%xmm13, %xmm9, %xmm14
	vaddss	%xmm6, %xmm14, %xmm0
	vmulss	%xmm1, %xmm0, %xmm15
	vmulss	%xmm15, %xmm15, %xmm7
	vaddss	%xmm15, %xmm15, %xmm3
	vdivss	%xmm7, %xmm9, %xmm5
	vaddss	%xmm3, %xmm5, %xmm8
	vmulss	%xmm1, %xmm8, %xmm2
	vmulss	%xmm2, %xmm2, %xmm4
	vaddss	%xmm2, %xmm2, %xmm11
	vdivss	%xmm4, %xmm9, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm1, %xmm12, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm14, %xmm9, %xmm6
	vaddss	%xmm0, %xmm6, %xmm15
	vmulss	%xmm1, %xmm15, %xmm7
	vmulss	%xmm7, %xmm7, %xmm5
	vaddss	%xmm7, %xmm7, %xmm8
	vdivss	%xmm5, %xmm9, %xmm3
	vaddss	%xmm8, %xmm3, %xmm2
	vmulss	%xmm1, %xmm2, %xmm4
	vmulss	%xmm4, %xmm4, %xmm10
	vaddss	%xmm4, %xmm4, %xmm12
	vdivss	%xmm10, %xmm9, %xmm11
	vaddss	%xmm12, %xmm11, %xmm13
	vmulss	%xmm1, %xmm13, %xmm14
	vmulss	%xmm14, %xmm14, %xmm6
	vaddss	%xmm14, %xmm14, %xmm15
	vdivss	%xmm6, %xmm9, %xmm0
	vaddss	%xmm15, %xmm0, %xmm7
	vmulss	%xmm1, %xmm7, %xmm5
	vmulss	%xmm5, %xmm5, %xmm3
	vaddss	%xmm5, %xmm5, %xmm2
	vdivss	%xmm3, %xmm9, %xmm8
	vaddss	%xmm2, %xmm8, %xmm4
	vmulss	%xmm1, %xmm4, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm13
	vdivss	%xmm11, %xmm9, %xmm12
	vaddss	%xmm13, %xmm12, %xmm14
	vmulss	%xmm1, %xmm14, %xmm6
	vmulss	%xmm6, %xmm6, %xmm0
	vaddss	%xmm6, %xmm6, %xmm7
	vdivss	%xmm0, %xmm9, %xmm15
	vaddss	%xmm7, %xmm15, %xmm5
	vmulss	%xmm1, %xmm5, %xmm8
	vmulss	%xmm8, %xmm8, %xmm3
	vaddss	%xmm8, %xmm8, %xmm2
	vdivss	%xmm3, %xmm9, %xmm4
	vaddss	%xmm2, %xmm4, %xmm10
	vmulss	%xmm1, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm9, %xmm13
	vaddss	%xmm14, %xmm13, %xmm6
	vmulss	%xmm1, %xmm6, %xmm0
	vmulss	%xmm0, %xmm0, %xmm15
	vaddss	%xmm0, %xmm0, %xmm5
	vdivss	%xmm15, %xmm9, %xmm7
	vaddss	%xmm5, %xmm7, %xmm8
	vmulss	%xmm1, %xmm8, %xmm3
	vmulss	%xmm3, %xmm3, %xmm4
	vaddss	%xmm3, %xmm3, %xmm10
	vdivss	%xmm4, %xmm9, %xmm2
	vaddss	%xmm10, %xmm2, %xmm11
	vmulss	%xmm1, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm14
	vdivss	%xmm13, %xmm9, %xmm9
	vaddss	%xmm14, %xmm9, %xmm6
	vmulss	%xmm1, %xmm6, %xmm0
	vmovss	%xmm0, (%rsi,%rax)
	addq	$4, %rax
	cmpq	%r11, %rax
	jne	.L101
	ret
.L109:
	vzeroupper
.L111:
	ret
.L112:
	cmpl	$6, %edx
	jbe	.L103
	vbroadcastss	.LC0(%rip), %ymm3
	shrl	$3, %r8d
	xorl	%r10d, %r10d
	salq	$5, %r8
	.p2align 4,,10
	.p2align 3
.L95:
	vmovups	(%rcx,%r10), %ymm0
	vmulps	%ymm3, %ymm0, %ymm2
	vmulps	%ymm2, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm10
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm0, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm6
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm4
	vmulps	%ymm15, %ymm14, %ymm2
	vsubps	%ymm2, %ymm4, %ymm1
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm2
	vrcpps	%ymm9, %ymm10
	vmulps	%ymm9, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm13
	vmulps	%ymm11, %ymm10, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm0, %ymm15
	vaddps	%ymm2, %ymm15, %ymm4
	vmulps	%ymm3, %ymm4, %ymm5
	vmulps	%ymm5, %ymm5, %ymm6
	vaddps	%ymm5, %ymm5, %ymm12
	vrcpps	%ymm6, %ymm1
	vmulps	%ymm6, %ymm1, %ymm7
	vaddps	%ymm1, %ymm1, %ymm9
	vmulps	%ymm7, %ymm1, %ymm8
	vsubps	%ymm8, %ymm9, %ymm10
	vmulps	%ymm10, %ymm0, %ymm11
	vaddps	%ymm12, %ymm11, %ymm13
	vmulps	%ymm3, %ymm13, %ymm14
	vmulps	%ymm14, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm8
	vrcpps	%ymm15, %ymm2
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm6
	vmulps	%ymm4, %ymm2, %ymm5
	vsubps	%ymm5, %ymm6, %ymm1
	vmulps	%ymm1, %ymm0, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm4
	vrcpps	%ymm11, %ymm12
	vmulps	%ymm11, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm15
	vmulps	%ymm13, %ymm12, %ymm14
	vsubps	%ymm14, %ymm15, %ymm2
	vmulps	%ymm2, %ymm0, %ymm5
	vaddps	%ymm4, %ymm5, %ymm6
	vmulps	%ymm3, %ymm6, %ymm7
	vmulps	%ymm7, %ymm7, %ymm8
	vaddps	%ymm7, %ymm7, %ymm14
	vrcpps	%ymm8, %ymm1
	vmulps	%ymm8, %ymm1, %ymm9
	vaddps	%ymm1, %ymm1, %ymm11
	vmulps	%ymm9, %ymm1, %ymm10
	vsubps	%ymm10, %ymm11, %ymm12
	vmulps	%ymm12, %ymm0, %ymm13
	vaddps	%ymm14, %ymm13, %ymm15
	vmulps	%ymm3, %ymm15, %ymm5
	vmulps	%ymm5, %ymm5, %ymm2
	vaddps	%ymm5, %ymm5, %ymm10
	vrcpps	%ymm2, %ymm4
	vmulps	%ymm2, %ymm4, %ymm6
	vaddps	%ymm4, %ymm4, %ymm8
	vmulps	%ymm6, %ymm4, %ymm7
	vsubps	%ymm7, %ymm8, %ymm1
	vmulps	%ymm1, %ymm0, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm7
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm2
	vmulps	%ymm15, %ymm14, %ymm5
	vsubps	%ymm5, %ymm2, %ymm4
	vmulps	%ymm4, %ymm0, %ymm6
	vaddps	%ymm7, %ymm6, %ymm8
	vmulps	%ymm3, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm5
	vrcpps	%ymm10, %ymm1
	vmulps	%ymm10, %ymm1, %ymm11
	vaddps	%ymm1, %ymm1, %ymm13
	vmulps	%ymm11, %ymm1, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm0, %ymm15
	vaddps	%ymm5, %ymm15, %ymm2
	vmulps	%ymm3, %ymm2, %ymm6
	vmulps	%ymm6, %ymm6, %ymm4
	vaddps	%ymm6, %ymm6, %ymm12
	vrcpps	%ymm4, %ymm7
	vmulps	%ymm4, %ymm7, %ymm8
	vaddps	%ymm7, %ymm7, %ymm10
	vmulps	%ymm8, %ymm7, %ymm9
	vsubps	%ymm9, %ymm10, %ymm1
	vmulps	%ymm1, %ymm0, %ymm11
	vaddps	%ymm12, %ymm11, %ymm13
	vmulps	%ymm3, %ymm13, %ymm14
	vmulps	%ymm14, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm9
	vrcpps	%ymm15, %ymm5
	vmulps	%ymm15, %ymm5, %ymm2
	vaddps	%ymm5, %ymm5, %ymm4
	vmulps	%ymm2, %ymm5, %ymm6
	vsubps	%ymm6, %ymm4, %ymm7
	vmulps	%ymm7, %ymm0, %ymm8
	vaddps	%ymm9, %ymm8, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm2
	vrcpps	%ymm12, %ymm1
	vmulps	%ymm12, %ymm1, %ymm13
	vaddps	%ymm1, %ymm1, %ymm15
	vmulps	%ymm13, %ymm1, %ymm14
	vsubps	%ymm14, %ymm15, %ymm5
	vmulps	%ymm5, %ymm0, %ymm6
	vaddps	%ymm2, %ymm6, %ymm4
	vmulps	%ymm3, %ymm4, %ymm7
	vmulps	%ymm7, %ymm7, %ymm8
	vaddps	%ymm7, %ymm7, %ymm14
	vrcpps	%ymm8, %ymm9
	vmulps	%ymm8, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm12
	vmulps	%ymm10, %ymm9, %ymm11
	vsubps	%ymm11, %ymm12, %ymm1
	vmulps	%ymm1, %ymm0, %ymm13
	vaddps	%ymm14, %ymm13, %ymm15
	vmulps	%ymm3, %ymm15, %ymm5
	vmulps	%ymm5, %ymm5, %ymm6
	vaddps	%ymm5, %ymm5, %ymm11
	vrcpps	%ymm6, %ymm2
	vmulps	%ymm6, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm8
	vmulps	%ymm4, %ymm2, %ymm7
	vsubps	%ymm7, %ymm8, %ymm9
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm1
	vmulps	%ymm1, %ymm1, %ymm13
	vaddps	%ymm1, %ymm1, %ymm4
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm6
	vmulps	%ymm15, %ymm14, %ymm5
	vsubps	%ymm5, %ymm6, %ymm2
	vmulps	%ymm2, %ymm0, %ymm0
	vaddps	%ymm4, %ymm0, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmovups	%ymm8, (%rsi,%r10)
	addq	$32, %r10
	cmpq	%r10, %r8
	jne	.L95
	movl	%eax, %r11d
	andl	$-8, %r11d
	movl	%r11d, %edx
	cmpl	%r11d, %eax
	je	.L109
	movl	%eax, %r8d
	subl	%r11d, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L113
	vzeroupper
.L94:
	vmovups	(%rcx,%rdx,4), %xmm9
	movl	%r8d, %edi
	vbroadcastss	.LC0(%rip), %xmm3
	andl	$-4, %edi
	vmulps	%xmm3, %xmm9, %xmm10
	addl	%edi, %r11d
	andl	$3, %r8d
	vmulps	%xmm10, %xmm10, %xmm11
	vaddps	%xmm10, %xmm10, %xmm6
	vrcpps	%xmm11, %xmm12
	vmulps	%xmm11, %xmm12, %xmm1
	vaddps	%xmm12, %xmm12, %xmm14
	vmulps	%xmm1, %xmm12, %xmm13
	vsubps	%xmm13, %xmm14, %xmm15
	vmulps	%xmm15, %xmm9, %xmm5
	vaddps	%xmm6, %xmm5, %xmm2
	vmulps	%xmm3, %xmm2, %xmm0
	vmulps	%xmm0, %xmm0, %xmm4
	vaddps	%xmm0, %xmm0, %xmm13
	vrcpps	%xmm4, %xmm7
	vmulps	%xmm4, %xmm7, %xmm8
	vaddps	%xmm7, %xmm7, %xmm11
	vmulps	%xmm8, %xmm7, %xmm10
	vsubps	%xmm10, %xmm11, %xmm12
	vmulps	%xmm12, %xmm9, %xmm1
	vaddps	%xmm13, %xmm1, %xmm14
	vmulps	%xmm3, %xmm14, %xmm15
	vmulps	%xmm15, %xmm15, %xmm5
	vaddps	%xmm15, %xmm15, %xmm10
	vrcpps	%xmm5, %xmm6
	vmulps	%xmm5, %xmm6, %xmm2
	vaddps	%xmm6, %xmm6, %xmm4
	vmulps	%xmm2, %xmm6, %xmm0
	vsubps	%xmm0, %xmm4, %xmm7
	vmulps	%xmm7, %xmm9, %xmm8
	vaddps	%xmm10, %xmm8, %xmm11
	vmulps	%xmm3, %xmm11, %xmm12
	vmulps	%xmm12, %xmm12, %xmm13
	vaddps	%xmm12, %xmm12, %xmm0
	vrcpps	%xmm13, %xmm1
	vmulps	%xmm13, %xmm1, %xmm14
	vaddps	%xmm1, %xmm1, %xmm5
	vmulps	%xmm14, %xmm1, %xmm15
	vsubps	%xmm15, %xmm5, %xmm6
	vmulps	%xmm6, %xmm9, %xmm2
	vaddps	%xmm0, %xmm2, %xmm4
	vmulps	%xmm3, %xmm4, %xmm7
	vmulps	%xmm7, %xmm7, %xmm8
	vaddps	%xmm7, %xmm7, %xmm15
	vrcpps	%xmm8, %xmm10
	vmulps	%xmm8, %xmm10, %xmm11
	vaddps	%xmm10, %xmm10, %xmm13
	vmulps	%xmm11, %xmm10, %xmm12
	vsubps	%xmm12, %xmm13, %xmm1
	vmulps	%xmm1, %xmm9, %xmm14
	vaddps	%xmm15, %xmm14, %xmm5
	vmulps	%xmm3, %xmm5, %xmm6
	vmulps	%xmm6, %xmm6, %xmm2
	vaddps	%xmm6, %xmm6, %xmm12
	vrcpps	%xmm2, %xmm0
	vmulps	%xmm2, %xmm0, %xmm4
	vaddps	%xmm0, %xmm0, %xmm8
	vmulps	%xmm4, %xmm0, %xmm7
	vsubps	%xmm7, %xmm8, %xmm10
	vmulps	%xmm10, %xmm9, %xmm11
	vaddps	%xmm12, %xmm11, %xmm13
	vmulps	%xmm3, %xmm13, %xmm14
	vmulps	%xmm14, %xmm14, %xmm15
	vaddps	%xmm14, %xmm14, %xmm7
	vrcpps	%xmm15, %xmm1
	vmulps	%xmm15, %xmm1, %xmm5
	vaddps	%xmm1, %xmm1, %xmm2
	vmulps	%xmm5, %xmm1, %xmm6
	vsubps	%xmm6, %xmm2, %xmm0
	vmulps	%xmm0, %xmm9, %xmm4
	vaddps	%xmm7, %xmm4, %xmm8
	vmulps	%xmm3, %xmm8, %xmm10
	vmulps	%xmm10, %xmm10, %xmm11
	vaddps	%xmm10, %xmm10, %xmm6
	vrcpps	%xmm11, %xmm12
	vmulps	%xmm11, %xmm12, %xmm13
	vaddps	%xmm12, %xmm12, %xmm15
	vmulps	%xmm13, %xmm12, %xmm14
	vsubps	%xmm14, %xmm15, %xmm1
	vmulps	%xmm1, %xmm9, %xmm5
	vaddps	%xmm6, %xmm5, %xmm2
	vmulps	%xmm3, %xmm2, %xmm0
	vmulps	%xmm0, %xmm0, %xmm4
	vaddps	%xmm0, %xmm0, %xmm14
	vrcpps	%xmm4, %xmm7
	vmulps	%xmm4, %xmm7, %xmm8
	vaddps	%xmm7, %xmm7, %xmm11
	vmulps	%xmm8, %xmm7, %xmm10
	vsubps	%xmm10, %xmm11, %xmm12
	vmulps	%xmm12, %xmm9, %xmm13
	vaddps	%xmm14, %xmm13, %xmm15
	vmulps	%xmm3, %xmm15, %xmm5
	vmulps	%xmm5, %xmm5, %xmm6
	vaddps	%xmm5, %xmm5, %xmm10
	vrcpps	%xmm6, %xmm1
	vmulps	%xmm6, %xmm1, %xmm2
	vaddps	%xmm1, %xmm1, %xmm4
	vmulps	%xmm2, %xmm1, %xmm0
	vsubps	%xmm0, %xmm4, %xmm7
	vmulps	%xmm7, %xmm9, %xmm8
	vaddps	%xmm10, %xmm8, %xmm11
	vmulps	%xmm3, %xmm11, %xmm12
	vmulps	%xmm12, %xmm12, %xmm13
	vaddps	%xmm12, %xmm12, %xmm2
	vrcpps	%xmm13, %xmm14
	vmulps	%xmm13, %xmm14, %xmm15
	vaddps	%xmm14, %xmm14, %xmm6
	vmulps	%xmm15, %xmm14, %xmm5
	vsubps	%xmm5, %xmm6, %xmm1
	vmulps	%xmm1, %xmm9, %xmm0
	vaddps	%xmm2, %xmm0, %xmm4
	vmulps	%xmm3, %xmm4, %xmm7
	vmulps	%xmm7, %xmm7, %xmm8
	vaddps	%xmm7, %xmm7, %xmm5
	vrcpps	%xmm8, %xmm10
	vmulps	%xmm8, %xmm10, %xmm11
	vaddps	%xmm10, %xmm10, %xmm13
	vmulps	%xmm11, %xmm10, %xmm12
	vsubps	%xmm12, %xmm13, %xmm14
	vmulps	%xmm14, %xmm9, %xmm15
	vaddps	%xmm5, %xmm15, %xmm6
	vmulps	%xmm3, %xmm6, %xmm0
	vmulps	%xmm0, %xmm0, %xmm2
	vaddps	%xmm0, %xmm0, %xmm12
	vrcpps	%xmm2, %xmm1
	vmulps	%xmm2, %xmm1, %xmm4
	vaddps	%xmm1, %xmm1, %xmm8
	vmulps	%xmm4, %xmm1, %xmm7
	vsubps	%xmm7, %xmm8, %xmm10
	vmulps	%xmm10, %xmm9, %xmm11
	vaddps	%xmm12, %xmm11, %xmm13
	vmulps	%xmm3, %xmm13, %xmm14
	vmulps	%xmm14, %xmm14, %xmm15
	vaddps	%xmm14, %xmm14, %xmm4
	vrcpps	%xmm15, %xmm5
	vmulps	%xmm15, %xmm5, %xmm6
	vaddps	%xmm5, %xmm5, %xmm2
	vmulps	%xmm6, %xmm5, %xmm0
	vsubps	%xmm0, %xmm2, %xmm1
	vmulps	%xmm1, %xmm9, %xmm7
	vaddps	%xmm4, %xmm7, %xmm8
	vmulps	%xmm3, %xmm8, %xmm10
	vmulps	%xmm10, %xmm10, %xmm11
	vaddps	%xmm10, %xmm10, %xmm0
	vrcpps	%xmm11, %xmm12
	vmulps	%xmm11, %xmm12, %xmm13
	vaddps	%xmm12, %xmm12, %xmm15
	vmulps	%xmm13, %xmm12, %xmm14
	vsubps	%xmm14, %xmm15, %xmm5
	vmulps	%xmm5, %xmm9, %xmm6
	vaddps	%xmm0, %xmm6, %xmm2
	vmulps	%xmm3, %xmm2, %xmm1
	vmulps	%xmm1, %xmm1, %xmm7
	vaddps	%xmm1, %xmm1, %xmm13
	vrcpps	%xmm7, %xmm8
	vmulps	%xmm7, %xmm8, %xmm4
	vaddps	%xmm8, %xmm8, %xmm11
	vmulps	%xmm4, %xmm8, %xmm10
	vsubps	%xmm10, %xmm11, %xmm12
	vmulps	%xmm12, %xmm9, %xmm9
	vaddps	%xmm13, %xmm9, %xmm14
	vmulps	%xmm3, %xmm14, %xmm3
	vmovups	%xmm3, (%rsi,%rdx,4)
	je	.L111
.L99:
	movslq	%r11d, %r10
	vmovss	.LC0(%rip), %xmm5
	leal	1(%r11), %r8d
	vmovss	(%rcx,%r10,4), %xmm15
	leaq	0(,%r10,4), %rdx
	vmulss	%xmm5, %xmm15, %xmm6
	vmulss	%xmm6, %xmm6, %xmm0
	vaddss	%xmm6, %xmm6, %xmm1
	vdivss	%xmm0, %xmm15, %xmm2
	vaddss	%xmm1, %xmm2, %xmm7
	vmulss	%xmm5, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm4
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm4, %xmm15, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm5, %xmm12, %xmm9
	vmulss	%xmm9, %xmm9, %xmm13
	vaddss	%xmm9, %xmm9, %xmm3
	vdivss	%xmm13, %xmm15, %xmm14
	vaddss	%xmm3, %xmm14, %xmm6
	vmulss	%xmm5, %xmm6, %xmm0
	vmulss	%xmm0, %xmm0, %xmm2
	vaddss	%xmm0, %xmm0, %xmm7
	vdivss	%xmm2, %xmm15, %xmm1
	vaddss	%xmm7, %xmm1, %xmm8
	vmulss	%xmm5, %xmm8, %xmm4
	vmulss	%xmm4, %xmm4, %xmm10
	vaddss	%xmm4, %xmm4, %xmm12
	vdivss	%xmm10, %xmm15, %xmm11
	vaddss	%xmm12, %xmm11, %xmm9
	vmulss	%xmm5, %xmm9, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm6
	vdivss	%xmm14, %xmm15, %xmm3
	vaddss	%xmm6, %xmm3, %xmm0
	vmulss	%xmm5, %xmm0, %xmm1
	vmulss	%xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm1, %xmm8
	vdivss	%xmm2, %xmm15, %xmm7
	vaddss	%xmm8, %xmm7, %xmm4
	vmulss	%xmm5, %xmm4, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm9
	vdivss	%xmm11, %xmm15, %xmm12
	vaddss	%xmm9, %xmm12, %xmm13
	vmulss	%xmm5, %xmm13, %xmm14
	vmulss	%xmm14, %xmm14, %xmm3
	vaddss	%xmm14, %xmm14, %xmm0
	vdivss	%xmm3, %xmm15, %xmm6
	vaddss	%xmm0, %xmm6, %xmm1
	vmulss	%xmm5, %xmm1, %xmm2
	vmulss	%xmm2, %xmm2, %xmm7
	vaddss	%xmm2, %xmm2, %xmm4
	vdivss	%xmm7, %xmm15, %xmm8
	vaddss	%xmm4, %xmm8, %xmm10
	vmulss	%xmm5, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm13
	vdivss	%xmm12, %xmm15, %xmm9
	vaddss	%xmm13, %xmm9, %xmm14
	vmulss	%xmm5, %xmm14, %xmm6
	vmulss	%xmm6, %xmm6, %xmm3
	vaddss	%xmm6, %xmm6, %xmm1
	vdivss	%xmm3, %xmm15, %xmm0
	vaddss	%xmm1, %xmm0, %xmm2
	vmulss	%xmm5, %xmm2, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm15, %xmm4
	vaddss	%xmm10, %xmm4, %xmm11
	vmulss	%xmm5, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm9
	vaddss	%xmm12, %xmm12, %xmm14
	vdivss	%xmm9, %xmm15, %xmm13
	vaddss	%xmm14, %xmm13, %xmm6
	vmulss	%xmm5, %xmm6, %xmm3
	vmulss	%xmm3, %xmm3, %xmm0
	vaddss	%xmm3, %xmm3, %xmm2
	vdivss	%xmm0, %xmm15, %xmm1
	vaddss	%xmm2, %xmm1, %xmm7
	vmulss	%xmm5, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm4
	vaddss	%xmm8, %xmm8, %xmm10
	vdivss	%xmm4, %xmm15, %xmm15
	vaddss	%xmm10, %xmm15, %xmm11
	vmulss	%xmm5, %xmm11, %xmm12
	vmovss	%xmm12, (%rsi,%r10,4)
	cmpl	%r8d, %eax
	jle	.L111
	vmovss	4(%rcx,%rdx), %xmm9
	addl	$2, %r11d
	vmulss	%xmm5, %xmm9, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm3
	vdivss	%xmm14, %xmm9, %xmm6
	vaddss	%xmm3, %xmm6, %xmm0
	vmulss	%xmm5, %xmm0, %xmm1
	vmulss	%xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm1, %xmm8
	vdivss	%xmm2, %xmm9, %xmm7
	vaddss	%xmm8, %xmm7, %xmm4
	vmulss	%xmm5, %xmm4, %xmm15
	vmulss	%xmm15, %xmm15, %xmm10
	vaddss	%xmm15, %xmm15, %xmm12
	vdivss	%xmm10, %xmm9, %xmm11
	vaddss	%xmm12, %xmm11, %xmm13
	vmulss	%xmm5, %xmm13, %xmm14
	vmulss	%xmm14, %xmm14, %xmm6
	vaddss	%xmm14, %xmm14, %xmm0
	vdivss	%xmm6, %xmm9, %xmm3
	vaddss	%xmm0, %xmm3, %xmm1
	vmulss	%xmm5, %xmm1, %xmm7
	vmulss	%xmm7, %xmm7, %xmm2
	vaddss	%xmm7, %xmm7, %xmm4
	vdivss	%xmm2, %xmm9, %xmm8
	vaddss	%xmm4, %xmm8, %xmm15
	vmulss	%xmm5, %xmm15, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm13
	vdivss	%xmm11, %xmm9, %xmm12
	vaddss	%xmm13, %xmm12, %xmm14
	vmulss	%xmm5, %xmm14, %xmm6
	vmulss	%xmm6, %xmm6, %xmm3
	vaddss	%xmm6, %xmm6, %xmm1
	vdivss	%xmm3, %xmm9, %xmm0
	vaddss	%xmm1, %xmm0, %xmm7
	vmulss	%xmm5, %xmm7, %xmm2
	vmulss	%xmm2, %xmm2, %xmm8
	vaddss	%xmm2, %xmm2, %xmm15
	vdivss	%xmm8, %xmm9, %xmm4
	vaddss	%xmm15, %xmm4, %xmm10
	vmulss	%xmm5, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm9, %xmm13
	vaddss	%xmm14, %xmm13, %xmm6
	vmulss	%xmm5, %xmm6, %xmm0
	vmulss	%xmm0, %xmm0, %xmm3
	vaddss	%xmm0, %xmm0, %xmm7
	vdivss	%xmm3, %xmm9, %xmm1
	vaddss	%xmm7, %xmm1, %xmm2
	vmulss	%xmm5, %xmm2, %xmm8
	vmulss	%xmm8, %xmm8, %xmm4
	vaddss	%xmm8, %xmm8, %xmm10
	vdivss	%xmm4, %xmm9, %xmm15
	vaddss	%xmm10, %xmm15, %xmm11
	vmulss	%xmm5, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm6
	vdivss	%xmm13, %xmm9, %xmm14
	vaddss	%xmm6, %xmm14, %xmm0
	vmulss	%xmm5, %xmm0, %xmm3
	vmulss	%xmm3, %xmm3, %xmm1
	vaddss	%xmm3, %xmm3, %xmm2
	vdivss	%xmm1, %xmm9, %xmm7
	vaddss	%xmm2, %xmm7, %xmm8
	vmulss	%xmm5, %xmm8, %xmm4
	vmulss	%xmm4, %xmm4, %xmm15
	vaddss	%xmm4, %xmm4, %xmm11
	vdivss	%xmm15, %xmm9, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm5, %xmm12, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm14, %xmm9, %xmm6
	vaddss	%xmm0, %xmm6, %xmm3
	vmulss	%xmm5, %xmm3, %xmm7
	vmulss	%xmm7, %xmm7, %xmm1
	vaddss	%xmm7, %xmm7, %xmm2
	vdivss	%xmm1, %xmm9, %xmm9
	vaddss	%xmm2, %xmm9, %xmm8
	vmulss	%xmm5, %xmm8, %xmm4
	vmovss	%xmm4, 4(%rsi,%rdx)
	cmpl	%r11d, %eax
	jle	.L111
	vmovss	8(%rcx,%rdx), %xmm15
	vmulss	%xmm5, %xmm15, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm13
	vdivss	%xmm11, %xmm15, %xmm12
	vaddss	%xmm13, %xmm12, %xmm14
	vmulss	%xmm5, %xmm14, %xmm6
	vmulss	%xmm6, %xmm6, %xmm0
	vaddss	%xmm6, %xmm6, %xmm7
	vdivss	%xmm0, %xmm15, %xmm3
	vaddss	%xmm7, %xmm3, %xmm1
	vmulss	%xmm5, %xmm1, %xmm9
	vmulss	%xmm9, %xmm9, %xmm2
	vaddss	%xmm9, %xmm9, %xmm4
	vdivss	%xmm2, %xmm15, %xmm8
	vaddss	%xmm4, %xmm8, %xmm10
	vmulss	%xmm5, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm15, %xmm13
	vaddss	%xmm14, %xmm13, %xmm6
	vmulss	%xmm5, %xmm6, %xmm0
	vmulss	%xmm0, %xmm0, %xmm3
	vaddss	%xmm0, %xmm0, %xmm1
	vdivss	%xmm3, %xmm15, %xmm7
	vaddss	%xmm1, %xmm7, %xmm9
	vmulss	%xmm5, %xmm9, %xmm2
	vmulss	%xmm2, %xmm2, %xmm8
	vaddss	%xmm2, %xmm2, %xmm10
	vdivss	%xmm8, %xmm15, %xmm4
	vaddss	%xmm10, %xmm4, %xmm11
	vmulss	%xmm5, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm6
	vdivss	%xmm13, %xmm15, %xmm14
	vaddss	%xmm6, %xmm14, %xmm0
	vmulss	%xmm5, %xmm0, %xmm7
	vmulss	%xmm7, %xmm7, %xmm3
	vaddss	%xmm7, %xmm7, %xmm9
	vdivss	%xmm3, %xmm15, %xmm1
	vaddss	%xmm9, %xmm1, %xmm2
	vmulss	%xmm5, %xmm2, %xmm8
	vmulss	%xmm8, %xmm8, %xmm4
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm4, %xmm15, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm5, %xmm12, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm14, %xmm15, %xmm6
	vaddss	%xmm0, %xmm6, %xmm7
	vmulss	%xmm5, %xmm7, %xmm3
	vmulss	%xmm3, %xmm3, %xmm1
	vaddss	%xmm3, %xmm3, %xmm2
	vdivss	%xmm1, %xmm15, %xmm9
	vaddss	%xmm2, %xmm9, %xmm8
	vmulss	%xmm5, %xmm8, %xmm4
	vmulss	%xmm4, %xmm4, %xmm10
	vaddss	%xmm4, %xmm4, %xmm12
	vdivss	%xmm10, %xmm15, %xmm11
	vaddss	%xmm12, %xmm11, %xmm13
	vmulss	%xmm5, %xmm13, %xmm14
	vmulss	%xmm14, %xmm14, %xmm6
	vaddss	%xmm14, %xmm14, %xmm7
	vdivss	%xmm6, %xmm15, %xmm0
	vaddss	%xmm7, %xmm0, %xmm3
	vmulss	%xmm5, %xmm3, %xmm1
	vmulss	%xmm1, %xmm1, %xmm9
	vaddss	%xmm1, %xmm1, %xmm2
	vdivss	%xmm9, %xmm15, %xmm8
	vaddss	%xmm2, %xmm8, %xmm4
	vmulss	%xmm5, %xmm4, %xmm10
	vmulss	%xmm10, %xmm10, %xmm11
	vaddss	%xmm10, %xmm10, %xmm13
	vdivss	%xmm11, %xmm15, %xmm12
	vaddss	%xmm13, %xmm12, %xmm14
	vmulss	%xmm5, %xmm14, %xmm6
	vmulss	%xmm6, %xmm6, %xmm0
	vaddss	%xmm6, %xmm6, %xmm7
	vdivss	%xmm0, %xmm15, %xmm15
	vaddss	%xmm7, %xmm15, %xmm3
	vmulss	%xmm5, %xmm3, %xmm5
	vmovss	%xmm5, 8(%rsi,%rdx)
	ret
.L103:
	xorl	%edx, %edx
	xorl	%r11d, %r11d
	jmp	.L94
.L113:
	vzeroupper
	jmp	.L99
	.cfi_endproc
.LFE12772:
	.size	_Z22opt_cube_root_templateILi16EEvPKfPfi, .-_Z22opt_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z23avx2_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z23avx2_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z23avx2_cube_root_templateILi16EEvPKfPfi
	.type	_Z23avx2_cube_root_templateILi16EEvPKfPfi, @function
_Z23avx2_cube_root_templateILi16EEvPKfPfi:
.LFB12773:
	.cfi_startproc
	endbr64
	movl	%edx, %eax
	leal	14(%rdx), %edx
	addl	$7, %eax
	cmovns	%eax, %edx
	andl	$-8, %edx
	jle	.L118
	vbroadcastss	.LC0(%rip), %ymm3
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L116:
	vmovups	(%rdi,%rcx,4), %ymm0
	vmulps	%ymm3, %ymm0, %ymm2
	vmulps	%ymm2, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm10
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm0, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm6
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm2
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm2, %ymm1
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm4
	vrcpps	%ymm9, %ymm10
	vmulps	%ymm9, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm13
	vmulps	%ymm11, %ymm10, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm0, %ymm15
	vaddps	%ymm4, %ymm15, %ymm2
	vmulps	%ymm3, %ymm2, %ymm6
	vmulps	%ymm6, %ymm6, %ymm5
	vaddps	%ymm6, %ymm6, %ymm12
	vrcpps	%ymm5, %ymm1
	vmulps	%ymm5, %ymm1, %ymm7
	vaddps	%ymm1, %ymm1, %ymm9
	vmulps	%ymm7, %ymm1, %ymm8
	vsubps	%ymm8, %ymm9, %ymm10
	vmulps	%ymm10, %ymm0, %ymm11
	vaddps	%ymm12, %ymm11, %ymm13
	vmulps	%ymm3, %ymm13, %ymm14
	vmulps	%ymm14, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm8
	vrcpps	%ymm15, %ymm2
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm5
	vmulps	%ymm4, %ymm2, %ymm6
	vsubps	%ymm6, %ymm5, %ymm1
	vmulps	%ymm1, %ymm0, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm4
	vrcpps	%ymm11, %ymm12
	vmulps	%ymm11, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm15
	vmulps	%ymm13, %ymm12, %ymm14
	vsubps	%ymm14, %ymm15, %ymm2
	vmulps	%ymm2, %ymm0, %ymm6
	vaddps	%ymm4, %ymm6, %ymm5
	vmulps	%ymm3, %ymm5, %ymm7
	vmulps	%ymm7, %ymm7, %ymm8
	vaddps	%ymm7, %ymm7, %ymm14
	vrcpps	%ymm8, %ymm1
	vmulps	%ymm8, %ymm1, %ymm9
	vaddps	%ymm1, %ymm1, %ymm11
	vmulps	%ymm9, %ymm1, %ymm10
	vsubps	%ymm10, %ymm11, %ymm12
	vmulps	%ymm12, %ymm0, %ymm13
	vaddps	%ymm14, %ymm13, %ymm15
	vmulps	%ymm3, %ymm15, %ymm6
	vmulps	%ymm6, %ymm6, %ymm2
	vaddps	%ymm6, %ymm6, %ymm10
	vrcpps	%ymm2, %ymm4
	vmulps	%ymm2, %ymm4, %ymm5
	vaddps	%ymm4, %ymm4, %ymm8
	vmulps	%ymm5, %ymm4, %ymm7
	vsubps	%ymm7, %ymm8, %ymm1
	vmulps	%ymm1, %ymm0, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm7
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm2
	vmulps	%ymm15, %ymm14, %ymm6
	vsubps	%ymm6, %ymm2, %ymm4
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm7, %ymm5, %ymm8
	vmulps	%ymm3, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm6
	vrcpps	%ymm10, %ymm1
	vmulps	%ymm10, %ymm1, %ymm11
	vaddps	%ymm1, %ymm1, %ymm13
	vmulps	%ymm11, %ymm1, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm0, %ymm15
	vaddps	%ymm6, %ymm15, %ymm2
	vmulps	%ymm3, %ymm2, %ymm7
	vmulps	%ymm7, %ymm7, %ymm4
	vaddps	%ymm7, %ymm7, %ymm12
	vrcpps	%ymm4, %ymm8
	vmulps	%ymm4, %ymm8, %ymm5
	vaddps	%ymm8, %ymm8, %ymm10
	vmulps	%ymm5, %ymm8, %ymm9
	vsubps	%ymm9, %ymm10, %ymm1
	vmulps	%ymm1, %ymm0, %ymm11
	vaddps	%ymm12, %ymm11, %ymm13
	vmulps	%ymm3, %ymm13, %ymm14
	vmulps	%ymm14, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm9
	vrcpps	%ymm15, %ymm6
	vmulps	%ymm15, %ymm6, %ymm2
	vaddps	%ymm6, %ymm6, %ymm4
	vmulps	%ymm2, %ymm6, %ymm7
	vsubps	%ymm7, %ymm4, %ymm8
	vmulps	%ymm8, %ymm0, %ymm5
	vaddps	%ymm9, %ymm5, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm2
	vrcpps	%ymm12, %ymm1
	vmulps	%ymm12, %ymm1, %ymm13
	vaddps	%ymm1, %ymm1, %ymm15
	vmulps	%ymm13, %ymm1, %ymm14
	vsubps	%ymm14, %ymm15, %ymm6
	vmulps	%ymm6, %ymm0, %ymm7
	vaddps	%ymm2, %ymm7, %ymm4
	vmulps	%ymm3, %ymm4, %ymm8
	vmulps	%ymm8, %ymm8, %ymm5
	vaddps	%ymm8, %ymm8, %ymm14
	vrcpps	%ymm5, %ymm9
	vmulps	%ymm5, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm12
	vmulps	%ymm10, %ymm9, %ymm11
	vsubps	%ymm11, %ymm12, %ymm1
	vmulps	%ymm1, %ymm0, %ymm13
	vaddps	%ymm14, %ymm13, %ymm15
	vmulps	%ymm3, %ymm15, %ymm6
	vmulps	%ymm6, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm11
	vrcpps	%ymm7, %ymm2
	vmulps	%ymm7, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm5
	vmulps	%ymm4, %ymm2, %ymm8
	vsubps	%ymm8, %ymm5, %ymm9
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm1
	vmulps	%ymm1, %ymm1, %ymm13
	vaddps	%ymm1, %ymm1, %ymm4
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm7
	vmulps	%ymm15, %ymm14, %ymm6
	vsubps	%ymm6, %ymm7, %ymm2
	vmulps	%ymm2, %ymm0, %ymm0
	vaddps	%ymm4, %ymm0, %ymm8
	vmulps	%ymm3, %ymm8, %ymm5
	vmovups	%ymm5, (%rsi,%rcx,4)
	addq	$8, %rcx
	cmpl	%ecx, %edx
	jg	.L116
	vzeroupper
.L118:
	ret
	.cfi_endproc
.LFE12773:
	.size	_Z23avx2_cube_root_templateILi16EEvPKfPfi, .-_Z23avx2_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z25native_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z25native_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z25native_cube_root_templateILi18EEvPKfPfi
	.type	_Z25native_cube_root_templateILi18EEvPKfPfi, @function
_Z25native_cube_root_templateILi18EEvPKfPfi:
.LFB12778:
	.cfi_startproc
	endbr64
	testl	%edx, %edx
	jle	.L126
	movslq	%edx, %rdx
	vmovss	.LC0(%rip), %xmm3
	xorl	%r8d, %r8d
	leaq	0(,%rdx,4), %rcx
	.p2align 4,,10
	.p2align 3
.L122:
	vmovss	(%rdi,%r8), %xmm2
	movl	$18, %eax
	vmulss	%xmm3, %xmm2, %xmm6
	.p2align 4,,10
	.p2align 3
.L121:
	vmulss	%xmm6, %xmm6, %xmm1
	vaddss	%xmm6, %xmm6, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	subl	$1, %eax
	jne	.L121
	vmovss	%xmm6, (%rsi,%r8)
	addq	$4, %r8
	cmpq	%r8, %rcx
	jne	.L122
.L126:
	ret
	.cfi_endproc
.LFE12778:
	.size	_Z25native_cube_root_templateILi18EEvPKfPfi, .-_Z25native_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z22opt_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z22opt_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z22opt_cube_root_templateILi18EEvPKfPfi
	.type	_Z22opt_cube_root_templateILi18EEvPKfPfi, @function
_Z22opt_cube_root_templateILi18EEvPKfPfi:
.LFB12779:
	.cfi_startproc
	endbr64
	testl	%edx, %edx
	jle	.L142
	movslq	%edx, %rdx
	vmovss	.LC0(%rip), %xmm3
	xorl	%r8d, %r8d
	leaq	0(,%rdx,4), %rcx
	.p2align 4,,10
	.p2align 3
.L130:
	vmovss	(%rdi,%r8), %xmm2
	movl	$18, %eax
	vmulss	%xmm3, %xmm2, %xmm5
.L129:
	vmulss	%xmm5, %xmm5, %xmm1
	vaddss	%xmm5, %xmm5, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	vmulss	%xmm6, %xmm6, %xmm7
	vaddss	%xmm6, %xmm6, %xmm9
	vdivss	%xmm7, %xmm2, %xmm8
	vaddss	%xmm9, %xmm8, %xmm10
	vmulss	%xmm3, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm2, %xmm13
	vaddss	%xmm14, %xmm13, %xmm15
	vmulss	%xmm3, %xmm15, %xmm4
	vmulss	%xmm4, %xmm4, %xmm1
	vaddss	%xmm4, %xmm4, %xmm0
	vdivss	%xmm1, %xmm2, %xmm5
	vaddss	%xmm0, %xmm5, %xmm6
	vmulss	%xmm3, %xmm6, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm2, %xmm9
	vaddss	%xmm10, %xmm9, %xmm11
	vmulss	%xmm3, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm15
	vdivss	%xmm13, %xmm2, %xmm14
	vaddss	%xmm15, %xmm14, %xmm4
	vmulss	%xmm3, %xmm4, %xmm1
	vmulss	%xmm1, %xmm1, %xmm5
	vaddss	%xmm1, %xmm1, %xmm6
	vdivss	%xmm5, %xmm2, %xmm0
	vaddss	%xmm6, %xmm0, %xmm7
	vmulss	%xmm3, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm9, %xmm2, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm3, %xmm12, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm4
	vdivss	%xmm14, %xmm2, %xmm15
	vaddss	%xmm4, %xmm15, %xmm1
	vmulss	%xmm3, %xmm1, %xmm5
	subl	$9, %eax
	jne	.L129
	vmovss	%xmm5, (%rsi,%r8)
	addq	$4, %r8
	cmpq	%r8, %rcx
	jne	.L130
.L142:
	ret
	.cfi_endproc
.LFE12779:
	.size	_Z22opt_cube_root_templateILi18EEvPKfPfi, .-_Z22opt_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z23avx2_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z23avx2_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z23avx2_cube_root_templateILi18EEvPKfPfi
	.type	_Z23avx2_cube_root_templateILi18EEvPKfPfi, @function
_Z23avx2_cube_root_templateILi18EEvPKfPfi:
.LFB12780:
	.cfi_startproc
	endbr64
	leal	14(%rdx), %ecx
	addl	$7, %edx
	cmovns	%edx, %ecx
	andl	$-8, %ecx
	jle	.L154
	vbroadcastss	.LC0(%rip), %ymm3
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L146:
	vmovups	(%rdi,%rdx,4), %ymm2
	movl	$18, %eax
	vmulps	%ymm3, %ymm2, %ymm10
.L145:
	vmulps	%ymm10, %ymm10, %ymm4
	vaddps	%ymm10, %ymm10, %ymm0
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm0, %ymm9, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm6
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm4
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm4, %ymm1
	vmulps	%ymm1, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm15
	vrcpps	%ymm9, %ymm0
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm0, %ymm0, %ymm12
	vmulps	%ymm10, %ymm0, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm1
	vmulps	%ymm1, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm11
	vrcpps	%ymm5, %ymm6
	vmulps	%ymm5, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm9
	vmulps	%ymm7, %ymm6, %ymm8
	vsubps	%ymm8, %ymm9, %ymm0
	vmulps	%ymm0, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm8
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm5
	vsubps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm6, %ymm2, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm5
	vrcpps	%ymm11, %ymm0
	vmulps	%ymm11, %ymm0, %ymm12
	vaddps	%ymm0, %ymm0, %ymm14
	vmulps	%ymm12, %ymm0, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm5, %ymm4, %ymm1
	vmulps	%ymm3, %ymm1, %ymm10
	subl	$6, %eax
	jne	.L145
	vmovups	%ymm10, (%rsi,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L146
	vzeroupper
.L154:
	ret
	.cfi_endproc
.LFE12780:
	.size	_Z23avx2_cube_root_templateILi18EEvPKfPfi, .-_Z23avx2_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z25native_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z25native_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z25native_cube_root_templateILi100EEvPKfPfi
	.type	_Z25native_cube_root_templateILi100EEvPKfPfi, @function
_Z25native_cube_root_templateILi100EEvPKfPfi:
.LFB12785:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	testl	%eax, %eax
	jle	.L162
	vmovss	.LC0(%rip), %xmm3
	leaq	0(,%rax,4), %rcx
	xorl	%edx, %edx
.L158:
	vmovss	(%rdi,%rdx), %xmm2
	movl	$100, %r8d
	vmulss	%xmm3, %xmm2, %xmm6
	.p2align 4,,10
	.p2align 3
.L157:
	vmulss	%xmm6, %xmm6, %xmm1
	vaddss	%xmm6, %xmm6, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	subl	$1, %r8d
	jne	.L157
	vmovss	%xmm6, (%rsi,%rdx)
	addq	$4, %rdx
	cmpq	%rdx, %rcx
	jne	.L158
.L162:
	ret
	.cfi_endproc
.LFE12785:
	.size	_Z25native_cube_root_templateILi100EEvPKfPfi, .-_Z25native_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z22opt_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z22opt_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z22opt_cube_root_templateILi100EEvPKfPfi
	.type	_Z22opt_cube_root_templateILi100EEvPKfPfi, @function
_Z22opt_cube_root_templateILi100EEvPKfPfi:
.LFB12786:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	testl	%eax, %eax
	jle	.L179
	vmovss	.LC0(%rip), %xmm3
	leaq	0(,%rax,4), %rcx
	xorl	%edx, %edx
.L166:
	vmovss	(%rdi,%rdx), %xmm2
	movl	$100, %r8d
	vmulss	%xmm3, %xmm2, %xmm9
	.p2align 4,,10
	.p2align 3
.L165:
	vmulss	%xmm9, %xmm9, %xmm1
	vaddss	%xmm9, %xmm9, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	vmulss	%xmm6, %xmm6, %xmm7
	vaddss	%xmm6, %xmm6, %xmm9
	vdivss	%xmm7, %xmm2, %xmm8
	vaddss	%xmm9, %xmm8, %xmm10
	vmulss	%xmm3, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm2, %xmm13
	vaddss	%xmm14, %xmm13, %xmm15
	vmulss	%xmm3, %xmm15, %xmm1
	vmulss	%xmm1, %xmm1, %xmm4
	vaddss	%xmm1, %xmm1, %xmm5
	vdivss	%xmm4, %xmm2, %xmm0
	vaddss	%xmm5, %xmm0, %xmm6
	vmulss	%xmm3, %xmm6, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm2, %xmm9
	vaddss	%xmm10, %xmm9, %xmm11
	vmulss	%xmm3, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm15
	vdivss	%xmm13, %xmm2, %xmm14
	vaddss	%xmm15, %xmm14, %xmm1
	vmulss	%xmm3, %xmm1, %xmm4
	vmulss	%xmm4, %xmm4, %xmm0
	vaddss	%xmm4, %xmm4, %xmm6
	vdivss	%xmm0, %xmm2, %xmm5
	vaddss	%xmm6, %xmm5, %xmm7
	vmulss	%xmm3, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm9, %xmm2, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm3, %xmm12, %xmm13
	vmulss	%xmm13, %xmm13, %xmm14
	vaddss	%xmm13, %xmm13, %xmm1
	vdivss	%xmm14, %xmm2, %xmm15
	vaddss	%xmm1, %xmm15, %xmm4
	vmulss	%xmm3, %xmm4, %xmm5
	vmulss	%xmm5, %xmm5, %xmm0
	vaddss	%xmm5, %xmm5, %xmm7
	vdivss	%xmm0, %xmm2, %xmm6
	vaddss	%xmm7, %xmm6, %xmm8
	vmulss	%xmm3, %xmm8, %xmm9
	subl	$10, %r8d
	jne	.L165
	vmovss	%xmm9, (%rsi,%rdx)
	addq	$4, %rdx
	cmpq	%rdx, %rcx
	jne	.L166
.L179:
	ret
	.cfi_endproc
.LFE12786:
	.size	_Z22opt_cube_root_templateILi100EEvPKfPfi, .-_Z22opt_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z23avx2_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z23avx2_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z23avx2_cube_root_templateILi100EEvPKfPfi
	.type	_Z23avx2_cube_root_templateILi100EEvPKfPfi, @function
_Z23avx2_cube_root_templateILi100EEvPKfPfi:
.LFB12787:
	.cfi_startproc
	endbr64
	movl	%edx, %eax
	leal	14(%rdx), %ecx
	addl	$7, %eax
	cmovns	%eax, %ecx
	andl	$-8, %ecx
	jle	.L194
	vbroadcastss	.LC0(%rip), %ymm3
	xorl	%edx, %edx
.L183:
	vmovups	(%rdi,%rdx,4), %ymm2
	movl	$98, %r8d
	vmulps	%ymm3, %ymm2, %ymm4
	vmulps	%ymm4, %ymm4, %ymm1
	vaddps	%ymm4, %ymm4, %ymm10
	vrcpps	%ymm1, %ymm0
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm7
	vmulps	%ymm5, %ymm0, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm6
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm1
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm1, %ymm0
	vmulps	%ymm0, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm0
	.p2align 4,,10
	.p2align 3
.L182:
	vmulps	%ymm0, %ymm0, %ymm8
	vaddps	%ymm0, %ymm0, %ymm15
	vrcpps	%ymm8, %ymm9
	vmulps	%ymm8, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm12
	vmulps	%ymm10, %ymm9, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm6
	vmulps	%ymm6, %ymm6, %ymm1
	vaddps	%ymm6, %ymm6, %ymm11
	vrcpps	%ymm1, %ymm0
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm8
	vmulps	%ymm5, %ymm0, %ymm7
	vsubps	%ymm7, %ymm8, %ymm9
	vmulps	%ymm9, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm7
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm6
	vsubps	%ymm6, %ymm1, %ymm0
	vmulps	%ymm0, %ymm2, %ymm5
	vaddps	%ymm7, %ymm5, %ymm8
	vmulps	%ymm3, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm4
	vrcpps	%ymm10, %ymm11
	vmulps	%ymm10, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm14
	vmulps	%ymm12, %ymm11, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm6
	vaddps	%ymm4, %ymm6, %ymm1
	vmulps	%ymm3, %ymm1, %ymm7
	vmulps	%ymm7, %ymm7, %ymm5
	vaddps	%ymm7, %ymm7, %ymm13
	vrcpps	%ymm5, %ymm0
	vmulps	%ymm5, %ymm0, %ymm8
	vaddps	%ymm0, %ymm0, %ymm10
	vmulps	%ymm8, %ymm0, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm2, %ymm12
	vaddps	%ymm13, %ymm12, %ymm14
	vmulps	%ymm3, %ymm14, %ymm15
	vmulps	%ymm15, %ymm15, %ymm6
	vaddps	%ymm15, %ymm15, %ymm9
	vrcpps	%ymm6, %ymm4
	vmulps	%ymm6, %ymm4, %ymm1
	vaddps	%ymm4, %ymm4, %ymm5
	vmulps	%ymm1, %ymm4, %ymm7
	vsubps	%ymm7, %ymm5, %ymm0
	vmulps	%ymm0, %ymm2, %ymm8
	vaddps	%ymm9, %ymm8, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm1
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm6
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm6, %ymm4
	vmulps	%ymm4, %ymm2, %ymm7
	vaddps	%ymm1, %ymm7, %ymm5
	vmulps	%ymm3, %ymm5, %ymm0
	subl	$7, %r8d
	jne	.L182
	vmovups	%ymm0, (%rsi,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L183
	vzeroupper
.L194:
	ret
	.cfi_endproc
.LFE12787:
	.size	_Z23avx2_cube_root_templateILi100EEvPKfPfi, .-_Z23avx2_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z25native_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z25native_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z25native_cube_root_templateILi1000EEvPKfPfi
	.type	_Z25native_cube_root_templateILi1000EEvPKfPfi, @function
_Z25native_cube_root_templateILi1000EEvPKfPfi:
.LFB12792:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	testl	%eax, %eax
	jle	.L202
	vmovss	.LC0(%rip), %xmm3
	leaq	0(,%rax,4), %rcx
	xorl	%edx, %edx
.L198:
	vmovss	(%rdi,%rdx), %xmm2
	movl	$1000, %r8d
	vmulss	%xmm3, %xmm2, %xmm6
	.p2align 4,,10
	.p2align 3
.L197:
	vmulss	%xmm6, %xmm6, %xmm1
	vaddss	%xmm6, %xmm6, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	subl	$1, %r8d
	jne	.L197
	vmovss	%xmm6, (%rsi,%rdx)
	addq	$4, %rdx
	cmpq	%rdx, %rcx
	jne	.L198
.L202:
	ret
	.cfi_endproc
.LFE12792:
	.size	_Z25native_cube_root_templateILi1000EEvPKfPfi, .-_Z25native_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z22opt_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z22opt_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z22opt_cube_root_templateILi1000EEvPKfPfi
	.type	_Z22opt_cube_root_templateILi1000EEvPKfPfi, @function
_Z22opt_cube_root_templateILi1000EEvPKfPfi:
.LFB12793:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	testl	%eax, %eax
	jle	.L217
	vmovss	.LC0(%rip), %xmm3
	leaq	0(,%rax,4), %rcx
	xorl	%edx, %edx
.L206:
	vmovss	(%rdi,%rdx), %xmm2
	movl	$1000, %r8d
	vmulss	%xmm3, %xmm2, %xmm13
	.p2align 4,,10
	.p2align 3
.L205:
	vmulss	%xmm13, %xmm13, %xmm1
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	vmulss	%xmm6, %xmm6, %xmm7
	vaddss	%xmm6, %xmm6, %xmm9
	vdivss	%xmm7, %xmm2, %xmm8
	vaddss	%xmm9, %xmm8, %xmm10
	vmulss	%xmm3, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm2, %xmm13
	vaddss	%xmm14, %xmm13, %xmm15
	vmulss	%xmm3, %xmm15, %xmm1
	vmulss	%xmm1, %xmm1, %xmm4
	vaddss	%xmm1, %xmm1, %xmm5
	vdivss	%xmm4, %xmm2, %xmm0
	vaddss	%xmm5, %xmm0, %xmm6
	vmulss	%xmm3, %xmm6, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm2, %xmm9
	vaddss	%xmm10, %xmm9, %xmm11
	vmulss	%xmm3, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm15
	vdivss	%xmm13, %xmm2, %xmm14
	vaddss	%xmm15, %xmm14, %xmm1
	vmulss	%xmm3, %xmm1, %xmm4
	vmulss	%xmm4, %xmm4, %xmm0
	vaddss	%xmm4, %xmm4, %xmm6
	vdivss	%xmm0, %xmm2, %xmm5
	vaddss	%xmm6, %xmm5, %xmm7
	vmulss	%xmm3, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm9, %xmm2, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm3, %xmm12, %xmm13
	subl	$8, %r8d
	jne	.L205
	vmovss	%xmm13, (%rsi,%rdx)
	addq	$4, %rdx
	cmpq	%rdx, %rcx
	jne	.L206
.L217:
	ret
	.cfi_endproc
.LFE12793:
	.size	_Z22opt_cube_root_templateILi1000EEvPKfPfi, .-_Z22opt_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z23avx2_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z23avx2_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z23avx2_cube_root_templateILi1000EEvPKfPfi
	.type	_Z23avx2_cube_root_templateILi1000EEvPKfPfi, @function
_Z23avx2_cube_root_templateILi1000EEvPKfPfi:
.LFB12794:
	.cfi_startproc
	endbr64
	movl	%edx, %eax
	leal	14(%rdx), %ecx
	addl	$7, %eax
	cmovns	%eax, %ecx
	andl	$-8, %ecx
	jle	.L231
	vbroadcastss	.LC0(%rip), %ymm3
	xorl	%edx, %edx
.L221:
	vmovups	(%rdi,%rdx,4), %ymm2
	movl	$1000, %r8d
	vmulps	%ymm3, %ymm2, %ymm0
	.p2align 4,,10
	.p2align 3
.L220:
	vmulps	%ymm0, %ymm0, %ymm4
	vaddps	%ymm0, %ymm0, %ymm0
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm0, %ymm9, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm6
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm4
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm4, %ymm1
	vmulps	%ymm1, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm15
	vrcpps	%ymm9, %ymm0
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm0, %ymm0, %ymm12
	vmulps	%ymm10, %ymm0, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm1
	vmulps	%ymm1, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm11
	vrcpps	%ymm5, %ymm6
	vmulps	%ymm5, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm9
	vmulps	%ymm7, %ymm6, %ymm8
	vsubps	%ymm8, %ymm9, %ymm0
	vmulps	%ymm0, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm8
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm5
	vsubps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm6, %ymm2, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm5
	vrcpps	%ymm11, %ymm0
	vmulps	%ymm11, %ymm0, %ymm12
	vaddps	%ymm0, %ymm0, %ymm14
	vmulps	%ymm12, %ymm0, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm5, %ymm4, %ymm1
	vmulps	%ymm3, %ymm1, %ymm6
	vmulps	%ymm6, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm13
	vrcpps	%ymm7, %ymm8
	vmulps	%ymm7, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm11
	vmulps	%ymm9, %ymm8, %ymm10
	vsubps	%ymm10, %ymm11, %ymm0
	vmulps	%ymm0, %ymm2, %ymm12
	vaddps	%ymm13, %ymm12, %ymm14
	vmulps	%ymm3, %ymm14, %ymm15
	vmulps	%ymm15, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm10
	vrcpps	%ymm4, %ymm5
	vmulps	%ymm4, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm7
	vmulps	%ymm1, %ymm5, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm0
	subl	$8, %r8d
	jne	.L220
	vmovups	%ymm0, (%rsi,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L221
	vzeroupper
.L231:
	ret
	.cfi_endproc
.LFE12794:
	.size	_Z23avx2_cube_root_templateILi1000EEvPKfPfi, .-_Z23avx2_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z25native_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z25native_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z25native_cube_root_templateILi10000EEvPKfPfi
	.type	_Z25native_cube_root_templateILi10000EEvPKfPfi, @function
_Z25native_cube_root_templateILi10000EEvPKfPfi:
.LFB12799:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	testl	%eax, %eax
	jle	.L239
	vmovss	.LC0(%rip), %xmm3
	leaq	0(,%rax,4), %rcx
	xorl	%edx, %edx
.L235:
	vmovss	(%rdi,%rdx), %xmm2
	movl	$10000, %r8d
	vmulss	%xmm3, %xmm2, %xmm6
	.p2align 4,,10
	.p2align 3
.L234:
	vmulss	%xmm6, %xmm6, %xmm1
	vaddss	%xmm6, %xmm6, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	subl	$1, %r8d
	jne	.L234
	vmovss	%xmm6, (%rsi,%rdx)
	addq	$4, %rdx
	cmpq	%rdx, %rcx
	jne	.L235
.L239:
	ret
	.cfi_endproc
.LFE12799:
	.size	_Z25native_cube_root_templateILi10000EEvPKfPfi, .-_Z25native_cube_root_templateILi10000EEvPKfPfi
	.section	.text._Z22opt_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z22opt_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z22opt_cube_root_templateILi10000EEvPKfPfi
	.type	_Z22opt_cube_root_templateILi10000EEvPKfPfi, @function
_Z22opt_cube_root_templateILi10000EEvPKfPfi:
.LFB12800:
	.cfi_startproc
	endbr64
	movslq	%edx, %rax
	testl	%eax, %eax
	jle	.L254
	vmovss	.LC0(%rip), %xmm3
	leaq	0(,%rax,4), %rcx
	xorl	%edx, %edx
.L243:
	vmovss	(%rdi,%rdx), %xmm2
	movl	$10000, %r8d
	vmulss	%xmm3, %xmm2, %xmm13
	.p2align 4,,10
	.p2align 3
.L242:
	vmulss	%xmm13, %xmm13, %xmm1
	vaddss	%xmm13, %xmm13, %xmm0
	vdivss	%xmm1, %xmm2, %xmm4
	vaddss	%xmm0, %xmm4, %xmm5
	vmulss	%xmm3, %xmm5, %xmm6
	vmulss	%xmm6, %xmm6, %xmm7
	vaddss	%xmm6, %xmm6, %xmm9
	vdivss	%xmm7, %xmm2, %xmm8
	vaddss	%xmm9, %xmm8, %xmm10
	vmulss	%xmm3, %xmm10, %xmm11
	vmulss	%xmm11, %xmm11, %xmm12
	vaddss	%xmm11, %xmm11, %xmm14
	vdivss	%xmm12, %xmm2, %xmm13
	vaddss	%xmm14, %xmm13, %xmm15
	vmulss	%xmm3, %xmm15, %xmm1
	vmulss	%xmm1, %xmm1, %xmm4
	vaddss	%xmm1, %xmm1, %xmm5
	vdivss	%xmm4, %xmm2, %xmm0
	vaddss	%xmm5, %xmm0, %xmm6
	vmulss	%xmm3, %xmm6, %xmm7
	vmulss	%xmm7, %xmm7, %xmm8
	vaddss	%xmm7, %xmm7, %xmm10
	vdivss	%xmm8, %xmm2, %xmm9
	vaddss	%xmm10, %xmm9, %xmm11
	vmulss	%xmm3, %xmm11, %xmm12
	vmulss	%xmm12, %xmm12, %xmm13
	vaddss	%xmm12, %xmm12, %xmm15
	vdivss	%xmm13, %xmm2, %xmm14
	vaddss	%xmm15, %xmm14, %xmm1
	vmulss	%xmm3, %xmm1, %xmm4
	vmulss	%xmm4, %xmm4, %xmm0
	vaddss	%xmm4, %xmm4, %xmm6
	vdivss	%xmm0, %xmm2, %xmm5
	vaddss	%xmm6, %xmm5, %xmm7
	vmulss	%xmm3, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vaddss	%xmm8, %xmm8, %xmm11
	vdivss	%xmm9, %xmm2, %xmm10
	vaddss	%xmm11, %xmm10, %xmm12
	vmulss	%xmm3, %xmm12, %xmm13
	subl	$8, %r8d
	jne	.L242
	vmovss	%xmm13, (%rsi,%rdx)
	addq	$4, %rdx
	cmpq	%rdx, %rcx
	jne	.L243
.L254:
	ret
	.cfi_endproc
.LFE12800:
	.size	_Z22opt_cube_root_templateILi10000EEvPKfPfi, .-_Z22opt_cube_root_templateILi10000EEvPKfPfi
	.section	.text._Z23avx2_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z23avx2_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z23avx2_cube_root_templateILi10000EEvPKfPfi
	.type	_Z23avx2_cube_root_templateILi10000EEvPKfPfi, @function
_Z23avx2_cube_root_templateILi10000EEvPKfPfi:
.LFB12801:
	.cfi_startproc
	endbr64
	movl	%edx, %eax
	leal	14(%rdx), %ecx
	addl	$7, %eax
	cmovns	%eax, %ecx
	andl	$-8, %ecx
	jle	.L268
	vbroadcastss	.LC0(%rip), %ymm3
	xorl	%edx, %edx
.L258:
	vmovups	(%rdi,%rdx,4), %ymm2
	movl	$10000, %r8d
	vmulps	%ymm3, %ymm2, %ymm0
	.p2align 4,,10
	.p2align 3
.L257:
	vmulps	%ymm0, %ymm0, %ymm4
	vaddps	%ymm0, %ymm0, %ymm0
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm0, %ymm9, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm6
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm4
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm4, %ymm1
	vmulps	%ymm1, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm15
	vrcpps	%ymm9, %ymm0
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm0, %ymm0, %ymm12
	vmulps	%ymm10, %ymm0, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm1
	vmulps	%ymm1, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm11
	vrcpps	%ymm5, %ymm6
	vmulps	%ymm5, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm9
	vmulps	%ymm7, %ymm6, %ymm8
	vsubps	%ymm8, %ymm9, %ymm0
	vmulps	%ymm0, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm8
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm5
	vsubps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm6, %ymm2, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm5
	vrcpps	%ymm11, %ymm0
	vmulps	%ymm11, %ymm0, %ymm12
	vaddps	%ymm0, %ymm0, %ymm14
	vmulps	%ymm12, %ymm0, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm5, %ymm4, %ymm1
	vmulps	%ymm3, %ymm1, %ymm6
	vmulps	%ymm6, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm13
	vrcpps	%ymm7, %ymm8
	vmulps	%ymm7, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm11
	vmulps	%ymm9, %ymm8, %ymm10
	vsubps	%ymm10, %ymm11, %ymm0
	vmulps	%ymm0, %ymm2, %ymm12
	vaddps	%ymm13, %ymm12, %ymm14
	vmulps	%ymm3, %ymm14, %ymm15
	vmulps	%ymm15, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm10
	vrcpps	%ymm4, %ymm5
	vmulps	%ymm4, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm7
	vmulps	%ymm1, %ymm5, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm0
	subl	$8, %r8d
	jne	.L257
	vmovups	%ymm0, (%rsi,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L258
	vzeroupper
.L268:
	ret
	.cfi_endproc
.LFE12801:
	.size	_Z23avx2_cube_root_templateILi10000EEvPKfPfi, .-_Z23avx2_cube_root_templateILi10000EEvPKfPfi
	.section	.text._ZNSt12format_errorD2Ev,"axG",@progbits,_ZNSt12format_errorD5Ev,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt12format_errorD2Ev
	.type	_ZNSt12format_errorD2Ev, @function
_ZNSt12format_errorD2Ev:
.LFB11378:
	.cfi_startproc
	endbr64
	leaq	16+_ZTVSt12format_error(%rip), %rax
	movq	%rax, (%rdi)
	jmp	_ZNSt13runtime_errorD2Ev@PLT
	.cfi_endproc
.LFE11378:
	.size	_ZNSt12format_errorD2Ev, .-_ZNSt12format_errorD2Ev
	.weak	_ZNSt12format_errorD1Ev
	.set	_ZNSt12format_errorD1Ev,_ZNSt12format_errorD2Ev
	.section	.text._ZNSt12format_errorD0Ev,"axG",@progbits,_ZNSt12format_errorD5Ev,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt12format_errorD0Ev
	.type	_ZNSt12format_errorD0Ev, @function
_ZNSt12format_errorD0Ev:
.LFB11380:
	.cfi_startproc
	endbr64
	leaq	16+_ZTVSt12format_error(%rip), %rax
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	%rax, (%rdi)
	call	_ZNSt13runtime_errorD2Ev@PLT
	movq	%rbx, %rdi
	movl	$16, %esi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT
	.cfi_endproc
.LFE11380:
	.size	_ZNSt12format_errorD0Ev, .-_ZNSt12format_errorD0Ev
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0:
.LFB13936:
	.cfi_startproc
	endbr64
	jmp	sched_getcpu@PLT
	.cfi_endproc
.LFE13936:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0, .-_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0
	.section	.text._Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi
	.type	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi, @function
_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi:
.LFB12744:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12744:
	.size	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi, .-_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi:
.LFB12743:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12743:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi, .-_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z27parallel_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z27parallel_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z27parallel_cube_root_templateILi2EEvPKfPfi
	.type	_Z27parallel_cube_root_templateILi2EEvPKfPfi, @function
_Z27parallel_cube_root_templateILi2EEvPKfPfi:
.LFB12742:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z27parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12742:
	.size	_Z27parallel_cube_root_templateILi2EEvPKfPfi, .-_Z27parallel_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi
	.type	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi, @function
_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi:
.LFB12776:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12776:
	.size	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi, .-_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi:
.LFB12775:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12775:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi, .-_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z27parallel_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z27parallel_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z27parallel_cube_root_templateILi16EEvPKfPfi
	.type	_Z27parallel_cube_root_templateILi16EEvPKfPfi, @function
_Z27parallel_cube_root_templateILi16EEvPKfPfi:
.LFB12774:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z27parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12774:
	.size	_Z27parallel_cube_root_templateILi16EEvPKfPfi, .-_Z27parallel_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi
	.type	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi, @function
_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi:
.LFB12783:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12783:
	.size	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi, .-_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi:
.LFB12782:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12782:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi, .-_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z27parallel_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z27parallel_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z27parallel_cube_root_templateILi18EEvPKfPfi
	.type	_Z27parallel_cube_root_templateILi18EEvPKfPfi, @function
_Z27parallel_cube_root_templateILi18EEvPKfPfi:
.LFB12781:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z27parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12781:
	.size	_Z27parallel_cube_root_templateILi18EEvPKfPfi, .-_Z27parallel_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi
	.type	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi, @function
_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi:
.LFB12790:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12790:
	.size	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi, .-_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi:
.LFB12789:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12789:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi, .-_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z27parallel_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z27parallel_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z27parallel_cube_root_templateILi100EEvPKfPfi
	.type	_Z27parallel_cube_root_templateILi100EEvPKfPfi, @function
_Z27parallel_cube_root_templateILi100EEvPKfPfi:
.LFB12788:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z27parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12788:
	.size	_Z27parallel_cube_root_templateILi100EEvPKfPfi, .-_Z27parallel_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi
	.type	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi, @function
_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi:
.LFB12797:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12797:
	.size	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi, .-_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi:
.LFB12796:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12796:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi, .-_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z27parallel_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z27parallel_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z27parallel_cube_root_templateILi1000EEvPKfPfi
	.type	_Z27parallel_cube_root_templateILi1000EEvPKfPfi, @function
_Z27parallel_cube_root_templateILi1000EEvPKfPfi:
.LFB12795:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z27parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12795:
	.size	_Z27parallel_cube_root_templateILi1000EEvPKfPfi, .-_Z27parallel_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi
	.type	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi, @function
_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi:
.LFB12804:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12804:
	.size	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi, .-_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi:
.LFB12803:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12803:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi, .-_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi
	.section	.text._Z27parallel_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z27parallel_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z27parallel_cube_root_templateILi10000EEvPKfPfi
	.type	_Z27parallel_cube_root_templateILi10000EEvPKfPfi, @function
_Z27parallel_cube_root_templateILi10000EEvPKfPfi:
.LFB12802:
	.cfi_startproc
	endbr64
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_Z27parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0(%rip), %rdi
	xorl	%esi, %esi
	jmp	GOMP_parallel@PLT
	.cfi_endproc
.LFE12802:
	.size	_Z27parallel_cube_root_templateILi10000EEvPKfPfi, .-_Z27parallel_cube_root_templateILi10000EEvPKfPfi
	.section	.rodata._Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi.str1.1,"aMS",@progbits,1
.LC4:
	.string	"too large residual %d\n"
	.section	.rodata._Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi.str1.8,"aMS",@progbits,1
	.align 8
.LC6:
	.string	"workspace_offset[32] %d != size %d\n"
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi
	.type	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi, @function
_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi:
.LFB12745:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edx, %xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	movl	%edx, %r13d
	pushq	%r12
	.cfi_offset 12, -40
	movl	%edx, %r12d
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
	.cfi_offset 3, -48
	vmovsd	performance_cpu_weight(%rip), %xmm2
	vmovsd	efficient_cpu_weight(%rip), %xmm1
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	vmulsd	.LC3(%rip), %xmm3, %xmm4
	vaddsd	%xmm1, %xmm2, %xmm5
	vdivsd	%xmm5, %xmm4, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm8
	vcvttsd2sil	%xmm7, %ebx
	vcvttsd2sil	%xmm8, %r14d
	movl	%ebx, %eax
	sall	$4, %eax
	movl	%r14d, %edx
	subl	%eax, %r13d
	sall	$4, %edx
	subl	%edx, %r13d
	cmpl	$32, %r13d
	jg	.L334
	vmovd	%ebx, %xmm9
	vmovd	%r14d, %xmm11
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm9, %ymm10
	vpbroadcastd	%xmm11, %ymm12
	vmovdqa	%ymm10, 32(%rsp)
	vmovdqa	%ymm10, 64(%rsp)
	vmovdqa	%ymm12, 96(%rsp)
	vmovdqa	%ymm12, 128(%rsp)
	testl	%r13d, %r13d
	jle	.L333
	leal	-1(%r13), %ecx
	cmpl	$6, %ecx
	jbe	.L303
.L293:
	addl	$1, %ebx
	movl	%r13d, %esi
	vmovd	%ebx, %xmm1
	shrl	$3, %esi
	vpbroadcastd	%xmm1, %ymm0
	vmovdqa	%ymm0, 32(%rsp)
	cmpl	$1, %esi
	je	.L296
	vmovdqa	%ymm0, 64(%rsp)
	cmpl	$2, %esi
	je	.L296
	addl	$1, %r14d
	vmovd	%r14d, %xmm3
	vpbroadcastd	%xmm3, %ymm4
	vmovdqa	%ymm4, 96(%rsp)
	cmpl	$3, %esi
	je	.L296
	vmovdqa	%ymm4, 128(%rsp)
.L296:
	movl	%r13d, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	testb	$7, %r13b
	je	.L333
.L295:
	movl	%r13d, %r8d
	leaq	32(%rsp), %rbx
	subl	%edi, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L298
	movl	$1, %r11d
	leaq	(%rbx,%rdi,4), %r10
	movl	%r8d, %r14d
	vmovd	%r11d, %xmm5
	andl	$-4, %r14d
	vpshufd	$0, %xmm5, %xmm6
	vpaddd	(%r10), %xmm6, %xmm7
	addl	%r14d, %eax
	andl	$3, %r8d
	vmovdqa	%xmm7, (%r10)
	je	.L294
.L298:
	movslq	%eax, %rdx
	leal	1(%rax), %ecx
	addl	$1, 32(%rsp,%rdx,4)
	cmpl	%ecx, %r13d
	jle	.L294
	movslq	%ecx, %rdi
	addl	$2, %eax
	addl	$1, 32(%rsp,%rdi,4)
	cmpl	%r13d, %eax
	jge	.L294
	cltq
	addl	$1, 32(%rsp,%rax,4)
.L294:
	movl	$0, 160(%rsp)
	movl	32(%rsp), %r13d
	leaq	36(%rsp), %rax
	leaq	164(%rsp), %rsi
.L300:
	addl	(%rax), %r13d
	addq	$32, %rax
	movl	%r13d, -32(%rax)
	addl	-28(%rax), %r13d
	movl	%r13d, -28(%rax)
	addl	-24(%rax), %r13d
	movl	%r13d, -24(%rax)
	addl	-20(%rax), %r13d
	movl	%r13d, -20(%rax)
	addl	-16(%rax), %r13d
	movl	%r13d, -16(%rax)
	addl	-12(%rax), %r13d
	movl	%r13d, -12(%rax)
	addl	-8(%rax), %r13d
	movl	%r13d, -8(%rax)
	addl	-4(%rax), %r13d
	movl	%r13d, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L300
	movl	160(%rsp), %edx
	cmpl	%r12d, %edx
	jne	.L335
	vzeroupper
.L301:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	24(%rsp), %rsi
	movq	%rbx, 24(%rsp)
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L336
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L335:
	.cfi_restore_state
	movl	%r12d, %ecx
	leaq	.LC6(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	vzeroupper
	call	__printf_chk@PLT
	jmp	.L301
.L334:
	movl	%r13d, %edx
	leaq	.LC4(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vmovd	%ebx, %xmm13
	vmovd	%r14d, %xmm15
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm15, %ymm2
	vpbroadcastd	%xmm13, %ymm14
	vmovdqa	%ymm14, 64(%rsp)
	vmovdqa	%ymm2, 96(%rsp)
	vmovdqa	%ymm2, 128(%rsp)
	jmp	.L293
.L333:
	leaq	32(%rsp), %rbx
	jmp	.L294
.L303:
	xorl	%edi, %edi
	xorl	%eax, %eax
	jmp	.L295
.L336:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12745:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi, .-_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi
	.type	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi, @function
_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi:
.LFB12777:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edx, %xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	movl	%edx, %r13d
	pushq	%r12
	.cfi_offset 12, -40
	movl	%edx, %r12d
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
	.cfi_offset 3, -48
	vmovsd	performance_cpu_weight(%rip), %xmm2
	vmovsd	efficient_cpu_weight(%rip), %xmm1
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	vmulsd	.LC3(%rip), %xmm3, %xmm4
	vaddsd	%xmm1, %xmm2, %xmm5
	vdivsd	%xmm5, %xmm4, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm8
	vcvttsd2sil	%xmm7, %ebx
	vcvttsd2sil	%xmm8, %r14d
	movl	%ebx, %eax
	sall	$4, %eax
	movl	%r14d, %edx
	subl	%eax, %r13d
	sall	$4, %edx
	subl	%edx, %r13d
	cmpl	$32, %r13d
	jg	.L380
	vmovd	%ebx, %xmm9
	vmovd	%r14d, %xmm11
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm9, %ymm10
	vpbroadcastd	%xmm11, %ymm12
	vmovdqa	%ymm10, 32(%rsp)
	vmovdqa	%ymm10, 64(%rsp)
	vmovdqa	%ymm12, 96(%rsp)
	vmovdqa	%ymm12, 128(%rsp)
	testl	%r13d, %r13d
	jle	.L379
	leal	-1(%r13), %ecx
	cmpl	$6, %ecx
	jbe	.L349
.L339:
	addl	$1, %ebx
	movl	%r13d, %esi
	vmovd	%ebx, %xmm1
	shrl	$3, %esi
	vpbroadcastd	%xmm1, %ymm0
	vmovdqa	%ymm0, 32(%rsp)
	cmpl	$1, %esi
	je	.L342
	vmovdqa	%ymm0, 64(%rsp)
	cmpl	$2, %esi
	je	.L342
	addl	$1, %r14d
	vmovd	%r14d, %xmm3
	vpbroadcastd	%xmm3, %ymm4
	vmovdqa	%ymm4, 96(%rsp)
	cmpl	$3, %esi
	je	.L342
	vmovdqa	%ymm4, 128(%rsp)
.L342:
	movl	%r13d, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	testb	$7, %r13b
	je	.L379
.L341:
	movl	%r13d, %r8d
	leaq	32(%rsp), %rbx
	subl	%edi, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L344
	movl	$1, %r11d
	leaq	(%rbx,%rdi,4), %r10
	movl	%r8d, %r14d
	vmovd	%r11d, %xmm5
	andl	$-4, %r14d
	vpshufd	$0, %xmm5, %xmm6
	vpaddd	(%r10), %xmm6, %xmm7
	addl	%r14d, %eax
	andl	$3, %r8d
	vmovdqa	%xmm7, (%r10)
	je	.L340
.L344:
	movslq	%eax, %rdx
	leal	1(%rax), %ecx
	addl	$1, 32(%rsp,%rdx,4)
	cmpl	%ecx, %r13d
	jle	.L340
	movslq	%ecx, %rdi
	addl	$2, %eax
	addl	$1, 32(%rsp,%rdi,4)
	cmpl	%r13d, %eax
	jge	.L340
	cltq
	addl	$1, 32(%rsp,%rax,4)
.L340:
	movl	$0, 160(%rsp)
	movl	32(%rsp), %r13d
	leaq	36(%rsp), %rax
	leaq	164(%rsp), %rsi
.L346:
	addl	(%rax), %r13d
	addq	$32, %rax
	movl	%r13d, -32(%rax)
	addl	-28(%rax), %r13d
	movl	%r13d, -28(%rax)
	addl	-24(%rax), %r13d
	movl	%r13d, -24(%rax)
	addl	-20(%rax), %r13d
	movl	%r13d, -20(%rax)
	addl	-16(%rax), %r13d
	movl	%r13d, -16(%rax)
	addl	-12(%rax), %r13d
	movl	%r13d, -12(%rax)
	addl	-8(%rax), %r13d
	movl	%r13d, -8(%rax)
	addl	-4(%rax), %r13d
	movl	%r13d, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L346
	movl	160(%rsp), %edx
	cmpl	%r12d, %edx
	jne	.L381
	vzeroupper
.L347:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	24(%rsp), %rsi
	movq	%rbx, 24(%rsp)
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L382
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L381:
	.cfi_restore_state
	movl	%r12d, %ecx
	leaq	.LC6(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	vzeroupper
	call	__printf_chk@PLT
	jmp	.L347
.L380:
	movl	%r13d, %edx
	leaq	.LC4(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vmovd	%ebx, %xmm13
	vmovd	%r14d, %xmm15
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm15, %ymm2
	vpbroadcastd	%xmm13, %ymm14
	vmovdqa	%ymm14, 64(%rsp)
	vmovdqa	%ymm2, 96(%rsp)
	vmovdqa	%ymm2, 128(%rsp)
	jmp	.L339
.L379:
	leaq	32(%rsp), %rbx
	jmp	.L340
.L349:
	xorl	%edi, %edi
	xorl	%eax, %eax
	jmp	.L341
.L382:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12777:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi, .-_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi
	.type	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi, @function
_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi:
.LFB12784:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edx, %xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	movl	%edx, %r13d
	pushq	%r12
	.cfi_offset 12, -40
	movl	%edx, %r12d
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
	.cfi_offset 3, -48
	vmovsd	performance_cpu_weight(%rip), %xmm2
	vmovsd	efficient_cpu_weight(%rip), %xmm1
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	vmulsd	.LC3(%rip), %xmm3, %xmm4
	vaddsd	%xmm1, %xmm2, %xmm5
	vdivsd	%xmm5, %xmm4, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm8
	vcvttsd2sil	%xmm7, %ebx
	vcvttsd2sil	%xmm8, %r14d
	movl	%ebx, %eax
	sall	$4, %eax
	movl	%r14d, %edx
	subl	%eax, %r13d
	sall	$4, %edx
	subl	%edx, %r13d
	cmpl	$32, %r13d
	jg	.L426
	vmovd	%ebx, %xmm9
	vmovd	%r14d, %xmm11
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm9, %ymm10
	vpbroadcastd	%xmm11, %ymm12
	vmovdqa	%ymm10, 32(%rsp)
	vmovdqa	%ymm10, 64(%rsp)
	vmovdqa	%ymm12, 96(%rsp)
	vmovdqa	%ymm12, 128(%rsp)
	testl	%r13d, %r13d
	jle	.L425
	leal	-1(%r13), %ecx
	cmpl	$6, %ecx
	jbe	.L395
.L385:
	addl	$1, %ebx
	movl	%r13d, %esi
	vmovd	%ebx, %xmm1
	shrl	$3, %esi
	vpbroadcastd	%xmm1, %ymm0
	vmovdqa	%ymm0, 32(%rsp)
	cmpl	$1, %esi
	je	.L388
	vmovdqa	%ymm0, 64(%rsp)
	cmpl	$2, %esi
	je	.L388
	addl	$1, %r14d
	vmovd	%r14d, %xmm3
	vpbroadcastd	%xmm3, %ymm4
	vmovdqa	%ymm4, 96(%rsp)
	cmpl	$3, %esi
	je	.L388
	vmovdqa	%ymm4, 128(%rsp)
.L388:
	movl	%r13d, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	testb	$7, %r13b
	je	.L425
.L387:
	movl	%r13d, %r8d
	leaq	32(%rsp), %rbx
	subl	%edi, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L390
	movl	$1, %r11d
	leaq	(%rbx,%rdi,4), %r10
	movl	%r8d, %r14d
	vmovd	%r11d, %xmm5
	andl	$-4, %r14d
	vpshufd	$0, %xmm5, %xmm6
	vpaddd	(%r10), %xmm6, %xmm7
	addl	%r14d, %eax
	andl	$3, %r8d
	vmovdqa	%xmm7, (%r10)
	je	.L386
.L390:
	movslq	%eax, %rdx
	leal	1(%rax), %ecx
	addl	$1, 32(%rsp,%rdx,4)
	cmpl	%ecx, %r13d
	jle	.L386
	movslq	%ecx, %rdi
	addl	$2, %eax
	addl	$1, 32(%rsp,%rdi,4)
	cmpl	%r13d, %eax
	jge	.L386
	cltq
	addl	$1, 32(%rsp,%rax,4)
.L386:
	movl	$0, 160(%rsp)
	movl	32(%rsp), %r13d
	leaq	36(%rsp), %rax
	leaq	164(%rsp), %rsi
.L392:
	addl	(%rax), %r13d
	addq	$32, %rax
	movl	%r13d, -32(%rax)
	addl	-28(%rax), %r13d
	movl	%r13d, -28(%rax)
	addl	-24(%rax), %r13d
	movl	%r13d, -24(%rax)
	addl	-20(%rax), %r13d
	movl	%r13d, -20(%rax)
	addl	-16(%rax), %r13d
	movl	%r13d, -16(%rax)
	addl	-12(%rax), %r13d
	movl	%r13d, -12(%rax)
	addl	-8(%rax), %r13d
	movl	%r13d, -8(%rax)
	addl	-4(%rax), %r13d
	movl	%r13d, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L392
	movl	160(%rsp), %edx
	cmpl	%r12d, %edx
	jne	.L427
	vzeroupper
.L393:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	24(%rsp), %rsi
	movq	%rbx, 24(%rsp)
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L428
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L427:
	.cfi_restore_state
	movl	%r12d, %ecx
	leaq	.LC6(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	vzeroupper
	call	__printf_chk@PLT
	jmp	.L393
.L426:
	movl	%r13d, %edx
	leaq	.LC4(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vmovd	%ebx, %xmm13
	vmovd	%r14d, %xmm15
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm15, %ymm2
	vpbroadcastd	%xmm13, %ymm14
	vmovdqa	%ymm14, 64(%rsp)
	vmovdqa	%ymm2, 96(%rsp)
	vmovdqa	%ymm2, 128(%rsp)
	jmp	.L385
.L425:
	leaq	32(%rsp), %rbx
	jmp	.L386
.L395:
	xorl	%edi, %edi
	xorl	%eax, %eax
	jmp	.L387
.L428:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12784:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi, .-_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi
	.type	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi, @function
_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi:
.LFB12791:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edx, %xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	movl	%edx, %r13d
	pushq	%r12
	.cfi_offset 12, -40
	movl	%edx, %r12d
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
	.cfi_offset 3, -48
	vmovsd	performance_cpu_weight(%rip), %xmm2
	vmovsd	efficient_cpu_weight(%rip), %xmm1
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	vmulsd	.LC3(%rip), %xmm3, %xmm4
	vaddsd	%xmm1, %xmm2, %xmm5
	vdivsd	%xmm5, %xmm4, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm8
	vcvttsd2sil	%xmm7, %ebx
	vcvttsd2sil	%xmm8, %r14d
	movl	%ebx, %eax
	sall	$4, %eax
	movl	%r14d, %edx
	subl	%eax, %r13d
	sall	$4, %edx
	subl	%edx, %r13d
	cmpl	$32, %r13d
	jg	.L472
	vmovd	%ebx, %xmm9
	vmovd	%r14d, %xmm11
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm9, %ymm10
	vpbroadcastd	%xmm11, %ymm12
	vmovdqa	%ymm10, 32(%rsp)
	vmovdqa	%ymm10, 64(%rsp)
	vmovdqa	%ymm12, 96(%rsp)
	vmovdqa	%ymm12, 128(%rsp)
	testl	%r13d, %r13d
	jle	.L471
	leal	-1(%r13), %ecx
	cmpl	$6, %ecx
	jbe	.L441
.L431:
	addl	$1, %ebx
	movl	%r13d, %esi
	vmovd	%ebx, %xmm1
	shrl	$3, %esi
	vpbroadcastd	%xmm1, %ymm0
	vmovdqa	%ymm0, 32(%rsp)
	cmpl	$1, %esi
	je	.L434
	vmovdqa	%ymm0, 64(%rsp)
	cmpl	$2, %esi
	je	.L434
	addl	$1, %r14d
	vmovd	%r14d, %xmm3
	vpbroadcastd	%xmm3, %ymm4
	vmovdqa	%ymm4, 96(%rsp)
	cmpl	$3, %esi
	je	.L434
	vmovdqa	%ymm4, 128(%rsp)
.L434:
	movl	%r13d, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	testb	$7, %r13b
	je	.L471
.L433:
	movl	%r13d, %r8d
	leaq	32(%rsp), %rbx
	subl	%edi, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L436
	movl	$1, %r11d
	leaq	(%rbx,%rdi,4), %r10
	movl	%r8d, %r14d
	vmovd	%r11d, %xmm5
	andl	$-4, %r14d
	vpshufd	$0, %xmm5, %xmm6
	vpaddd	(%r10), %xmm6, %xmm7
	addl	%r14d, %eax
	andl	$3, %r8d
	vmovdqa	%xmm7, (%r10)
	je	.L432
.L436:
	movslq	%eax, %rdx
	leal	1(%rax), %ecx
	addl	$1, 32(%rsp,%rdx,4)
	cmpl	%ecx, %r13d
	jle	.L432
	movslq	%ecx, %rdi
	addl	$2, %eax
	addl	$1, 32(%rsp,%rdi,4)
	cmpl	%r13d, %eax
	jge	.L432
	cltq
	addl	$1, 32(%rsp,%rax,4)
.L432:
	movl	$0, 160(%rsp)
	movl	32(%rsp), %r13d
	leaq	36(%rsp), %rax
	leaq	164(%rsp), %rsi
.L438:
	addl	(%rax), %r13d
	addq	$32, %rax
	movl	%r13d, -32(%rax)
	addl	-28(%rax), %r13d
	movl	%r13d, -28(%rax)
	addl	-24(%rax), %r13d
	movl	%r13d, -24(%rax)
	addl	-20(%rax), %r13d
	movl	%r13d, -20(%rax)
	addl	-16(%rax), %r13d
	movl	%r13d, -16(%rax)
	addl	-12(%rax), %r13d
	movl	%r13d, -12(%rax)
	addl	-8(%rax), %r13d
	movl	%r13d, -8(%rax)
	addl	-4(%rax), %r13d
	movl	%r13d, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L438
	movl	160(%rsp), %edx
	cmpl	%r12d, %edx
	jne	.L473
	vzeroupper
.L439:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	24(%rsp), %rsi
	movq	%rbx, 24(%rsp)
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L474
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L473:
	.cfi_restore_state
	movl	%r12d, %ecx
	leaq	.LC6(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	vzeroupper
	call	__printf_chk@PLT
	jmp	.L439
.L472:
	movl	%r13d, %edx
	leaq	.LC4(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vmovd	%ebx, %xmm13
	vmovd	%r14d, %xmm15
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm15, %ymm2
	vpbroadcastd	%xmm13, %ymm14
	vmovdqa	%ymm14, 64(%rsp)
	vmovdqa	%ymm2, 96(%rsp)
	vmovdqa	%ymm2, 128(%rsp)
	jmp	.L431
.L471:
	leaq	32(%rsp), %rbx
	jmp	.L432
.L441:
	xorl	%edi, %edi
	xorl	%eax, %eax
	jmp	.L433
.L474:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12791:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi, .-_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi
	.type	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi, @function
_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi:
.LFB12798:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edx, %xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	movl	%edx, %r13d
	pushq	%r12
	.cfi_offset 12, -40
	movl	%edx, %r12d
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
	.cfi_offset 3, -48
	vmovsd	performance_cpu_weight(%rip), %xmm2
	vmovsd	efficient_cpu_weight(%rip), %xmm1
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	vmulsd	.LC3(%rip), %xmm3, %xmm4
	vaddsd	%xmm1, %xmm2, %xmm5
	vdivsd	%xmm5, %xmm4, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm8
	vcvttsd2sil	%xmm7, %ebx
	vcvttsd2sil	%xmm8, %r14d
	movl	%ebx, %eax
	sall	$4, %eax
	movl	%r14d, %edx
	subl	%eax, %r13d
	sall	$4, %edx
	subl	%edx, %r13d
	cmpl	$32, %r13d
	jg	.L518
	vmovd	%ebx, %xmm9
	vmovd	%r14d, %xmm11
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm9, %ymm10
	vpbroadcastd	%xmm11, %ymm12
	vmovdqa	%ymm10, 32(%rsp)
	vmovdqa	%ymm10, 64(%rsp)
	vmovdqa	%ymm12, 96(%rsp)
	vmovdqa	%ymm12, 128(%rsp)
	testl	%r13d, %r13d
	jle	.L517
	leal	-1(%r13), %ecx
	cmpl	$6, %ecx
	jbe	.L487
.L477:
	addl	$1, %ebx
	movl	%r13d, %esi
	vmovd	%ebx, %xmm1
	shrl	$3, %esi
	vpbroadcastd	%xmm1, %ymm0
	vmovdqa	%ymm0, 32(%rsp)
	cmpl	$1, %esi
	je	.L480
	vmovdqa	%ymm0, 64(%rsp)
	cmpl	$2, %esi
	je	.L480
	addl	$1, %r14d
	vmovd	%r14d, %xmm3
	vpbroadcastd	%xmm3, %ymm4
	vmovdqa	%ymm4, 96(%rsp)
	cmpl	$3, %esi
	je	.L480
	vmovdqa	%ymm4, 128(%rsp)
.L480:
	movl	%r13d, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	testb	$7, %r13b
	je	.L517
.L479:
	movl	%r13d, %r8d
	leaq	32(%rsp), %rbx
	subl	%edi, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L482
	movl	$1, %r11d
	leaq	(%rbx,%rdi,4), %r10
	movl	%r8d, %r14d
	vmovd	%r11d, %xmm5
	andl	$-4, %r14d
	vpshufd	$0, %xmm5, %xmm6
	vpaddd	(%r10), %xmm6, %xmm7
	addl	%r14d, %eax
	andl	$3, %r8d
	vmovdqa	%xmm7, (%r10)
	je	.L478
.L482:
	movslq	%eax, %rdx
	leal	1(%rax), %ecx
	addl	$1, 32(%rsp,%rdx,4)
	cmpl	%ecx, %r13d
	jle	.L478
	movslq	%ecx, %rdi
	addl	$2, %eax
	addl	$1, 32(%rsp,%rdi,4)
	cmpl	%r13d, %eax
	jge	.L478
	cltq
	addl	$1, 32(%rsp,%rax,4)
.L478:
	movl	$0, 160(%rsp)
	movl	32(%rsp), %r13d
	leaq	36(%rsp), %rax
	leaq	164(%rsp), %rsi
.L484:
	addl	(%rax), %r13d
	addq	$32, %rax
	movl	%r13d, -32(%rax)
	addl	-28(%rax), %r13d
	movl	%r13d, -28(%rax)
	addl	-24(%rax), %r13d
	movl	%r13d, -24(%rax)
	addl	-20(%rax), %r13d
	movl	%r13d, -20(%rax)
	addl	-16(%rax), %r13d
	movl	%r13d, -16(%rax)
	addl	-12(%rax), %r13d
	movl	%r13d, -12(%rax)
	addl	-8(%rax), %r13d
	movl	%r13d, -8(%rax)
	addl	-4(%rax), %r13d
	movl	%r13d, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L484
	movl	160(%rsp), %edx
	cmpl	%r12d, %edx
	jne	.L519
	vzeroupper
.L485:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	24(%rsp), %rsi
	movq	%rbx, 24(%rsp)
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L520
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L519:
	.cfi_restore_state
	movl	%r12d, %ecx
	leaq	.LC6(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	vzeroupper
	call	__printf_chk@PLT
	jmp	.L485
.L518:
	movl	%r13d, %edx
	leaq	.LC4(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vmovd	%ebx, %xmm13
	vmovd	%r14d, %xmm15
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm15, %ymm2
	vpbroadcastd	%xmm13, %ymm14
	vmovdqa	%ymm14, 64(%rsp)
	vmovdqa	%ymm2, 96(%rsp)
	vmovdqa	%ymm2, 128(%rsp)
	jmp	.L477
.L517:
	leaq	32(%rsp), %rbx
	jmp	.L478
.L487:
	xorl	%edi, %edi
	xorl	%eax, %eax
	jmp	.L479
.L520:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12798:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi, .-_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.weak	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi
	.type	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi, @function
_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi:
.LFB12805:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edx, %xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	movl	%edx, %r13d
	pushq	%r12
	.cfi_offset 12, -40
	movl	%edx, %r12d
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
	.cfi_offset 3, -48
	vmovsd	performance_cpu_weight(%rip), %xmm2
	vmovsd	efficient_cpu_weight(%rip), %xmm1
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	vmulsd	.LC3(%rip), %xmm3, %xmm4
	vaddsd	%xmm1, %xmm2, %xmm5
	vdivsd	%xmm5, %xmm4, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm8
	vcvttsd2sil	%xmm7, %ebx
	vcvttsd2sil	%xmm8, %r14d
	movl	%ebx, %eax
	sall	$4, %eax
	movl	%r14d, %edx
	subl	%eax, %r13d
	sall	$4, %edx
	subl	%edx, %r13d
	cmpl	$32, %r13d
	jg	.L564
	vmovd	%ebx, %xmm9
	vmovd	%r14d, %xmm11
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm9, %ymm10
	vpbroadcastd	%xmm11, %ymm12
	vmovdqa	%ymm10, 32(%rsp)
	vmovdqa	%ymm10, 64(%rsp)
	vmovdqa	%ymm12, 96(%rsp)
	vmovdqa	%ymm12, 128(%rsp)
	testl	%r13d, %r13d
	jle	.L563
	leal	-1(%r13), %ecx
	cmpl	$6, %ecx
	jbe	.L533
.L523:
	addl	$1, %ebx
	movl	%r13d, %esi
	vmovd	%ebx, %xmm1
	shrl	$3, %esi
	vpbroadcastd	%xmm1, %ymm0
	vmovdqa	%ymm0, 32(%rsp)
	cmpl	$1, %esi
	je	.L526
	vmovdqa	%ymm0, 64(%rsp)
	cmpl	$2, %esi
	je	.L526
	addl	$1, %r14d
	vmovd	%r14d, %xmm3
	vpbroadcastd	%xmm3, %ymm4
	vmovdqa	%ymm4, 96(%rsp)
	cmpl	$3, %esi
	je	.L526
	vmovdqa	%ymm4, 128(%rsp)
.L526:
	movl	%r13d, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	testb	$7, %r13b
	je	.L563
.L525:
	movl	%r13d, %r8d
	leaq	32(%rsp), %rbx
	subl	%edi, %r8d
	leal	-1(%r8), %r9d
	cmpl	$2, %r9d
	jbe	.L528
	movl	$1, %r11d
	leaq	(%rbx,%rdi,4), %r10
	movl	%r8d, %r14d
	vmovd	%r11d, %xmm5
	andl	$-4, %r14d
	vpshufd	$0, %xmm5, %xmm6
	vpaddd	(%r10), %xmm6, %xmm7
	addl	%r14d, %eax
	andl	$3, %r8d
	vmovdqa	%xmm7, (%r10)
	je	.L524
.L528:
	movslq	%eax, %rdx
	leal	1(%rax), %ecx
	addl	$1, 32(%rsp,%rdx,4)
	cmpl	%ecx, %r13d
	jle	.L524
	movslq	%ecx, %rdi
	addl	$2, %eax
	addl	$1, 32(%rsp,%rdi,4)
	cmpl	%r13d, %eax
	jge	.L524
	cltq
	addl	$1, 32(%rsp,%rax,4)
.L524:
	movl	$0, 160(%rsp)
	movl	32(%rsp), %r13d
	leaq	36(%rsp), %rax
	leaq	164(%rsp), %rsi
.L530:
	addl	(%rax), %r13d
	addq	$32, %rax
	movl	%r13d, -32(%rax)
	addl	-28(%rax), %r13d
	movl	%r13d, -28(%rax)
	addl	-24(%rax), %r13d
	movl	%r13d, -24(%rax)
	addl	-20(%rax), %r13d
	movl	%r13d, -20(%rax)
	addl	-16(%rax), %r13d
	movl	%r13d, -16(%rax)
	addl	-12(%rax), %r13d
	movl	%r13d, -12(%rax)
	addl	-8(%rax), %r13d
	movl	%r13d, -8(%rax)
	addl	-4(%rax), %r13d
	movl	%r13d, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L530
	movl	160(%rsp), %edx
	cmpl	%r12d, %edx
	jne	.L565
	vzeroupper
.L531:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	24(%rsp), %rsi
	movq	%rbx, 24(%rsp)
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L566
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L565:
	.cfi_restore_state
	movl	%r12d, %ecx
	leaq	.LC6(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	vzeroupper
	call	__printf_chk@PLT
	jmp	.L531
.L564:
	movl	%r13d, %edx
	leaq	.LC4(%rip), %rsi
	movl	$2, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vmovd	%ebx, %xmm13
	vmovd	%r14d, %xmm15
	movl	$0, 160(%rsp)
	vpbroadcastd	%xmm15, %ymm2
	vpbroadcastd	%xmm13, %ymm14
	vmovdqa	%ymm14, 64(%rsp)
	vmovdqa	%ymm2, 96(%rsp)
	vmovdqa	%ymm2, 128(%rsp)
	jmp	.L523
.L563:
	leaq	32(%rsp), %rbx
	jmp	.L524
.L533:
	xorl	%edi, %edi
	xorl	%eax, %eax
	jmp	.L525
.L566:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12805:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi, .-_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi
	.section	.text._Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0, @function
_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0:
.LFB13937:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE13937:
	.size	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0, .-_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0:
.LFB13938:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%esi, %esi
	xorl	%edi, %edi
	movl	$1000, %ecx
	movl	$8, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %r12
	leaq	8(%rsp), %rbx
	movq	%r12, %r9
	movq	%rbx, %r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L569
	vbroadcastss	.LC0(%rip), %ymm2
	.p2align 4,,10
	.p2align 3
.L571:
	movslq	8(%rsp), %rsi
	xorl	%edx, %edx
	movl	16(%rsp), %edi
	vmovups	(%rdx,%rsi,4), %ymm1
	movl	%esi, %ecx
	leaq	8(%rsi), %rax
	notl	%ecx
	vmulps	%ymm2, %ymm1, %ymm3
	addl	%edi, %ecx
	shrl	$3, %ecx
	andl	$1, %ecx
	vmulps	%ymm3, %ymm3, %ymm4
	vaddps	%ymm3, %ymm3, %ymm10
	vrcpps	%ymm4, %ymm0
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm7
	vmulps	%ymm5, %ymm0, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm1, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm5
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm3
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm3, %ymm0
	vmulps	%ymm0, %ymm1, %ymm1
	vaddps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm2, %ymm6, %ymm7
	vmovups	%ymm7, 0(,%rsi,4)
	cmpl	%eax, %edi
	jle	.L584
	testl	%ecx, %ecx
	je	.L570
	xorl	%r8d, %r8d
	vmovups	(%r8,%rax,4), %ymm8
	vmulps	%ymm2, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm3
	vrcpps	%ymm10, %ymm11
	vmulps	%ymm10, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm14
	vmulps	%ymm12, %ymm11, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm8, %ymm4
	vaddps	%ymm3, %ymm4, %ymm0
	vmulps	%ymm2, %ymm0, %ymm5
	vmulps	%ymm5, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm12
	vrcpps	%ymm1, %ymm6
	vmulps	%ymm1, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm10
	vmulps	%ymm7, %ymm6, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm8, %ymm8
	vaddps	%ymm12, %ymm8, %ymm13
	vmulps	%ymm2, %ymm13, %ymm14
	vmovups	%ymm14, 0(,%rax,4)
	leaq	16(%rsi), %rax
	cmpl	%eax, %edi
	jle	.L584
	.p2align 4,,10
	.p2align 3
.L570:
	xorl	%r9d, %r9d
	leaq	8(%rax), %r11
	xorl	%r10d, %r10d
	vmovups	(%r9,%rax,4), %ymm15
	vmulps	%ymm2, %ymm15, %ymm3
	vmulps	%ymm3, %ymm3, %ymm4
	vaddps	%ymm3, %ymm3, %ymm10
	vrcpps	%ymm4, %ymm0
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm6
	vmulps	%ymm5, %ymm0, %ymm1
	vsubps	%ymm1, %ymm6, %ymm7
	vmulps	%ymm7, %ymm15, %ymm9
	vmovups	(%r9,%r11,4), %ymm7
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm2, %ymm7, %ymm9
	vmulps	%ymm2, %ymm11, %ymm8
	vmulps	%ymm9, %ymm9, %ymm10
	vmulps	%ymm8, %ymm8, %ymm12
	vaddps	%ymm8, %ymm8, %ymm5
	vrcpps	%ymm10, %ymm11
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm10, %ymm11, %ymm8
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm3
	vmulps	%ymm8, %ymm11, %ymm12
	vmulps	%ymm14, %ymm13, %ymm4
	vaddps	%ymm11, %ymm11, %ymm13
	vsubps	%ymm12, %ymm13, %ymm14
	vsubps	%ymm4, %ymm3, %ymm0
	vaddps	%ymm9, %ymm9, %ymm3
	vmulps	%ymm14, %ymm7, %ymm4
	vmulps	%ymm0, %ymm15, %ymm15
	vaddps	%ymm3, %ymm4, %ymm0
	vaddps	%ymm5, %ymm15, %ymm1
	vmulps	%ymm2, %ymm0, %ymm15
	vmulps	%ymm2, %ymm1, %ymm6
	vmulps	%ymm15, %ymm15, %ymm5
	vaddps	%ymm15, %ymm15, %ymm8
	vmovups	%ymm6, (%r10,%rax,4)
	addq	$16, %rax
	vrcpps	%ymm5, %ymm1
	vmulps	%ymm5, %ymm1, %ymm6
	vaddps	%ymm1, %ymm1, %ymm10
	vmulps	%ymm6, %ymm1, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm7, %ymm7
	vaddps	%ymm8, %ymm7, %ymm12
	vmulps	%ymm2, %ymm12, %ymm13
	vmovups	%ymm13, (%r10,%r11,4)
	cmpl	%eax, %edi
	jg	.L570
.L584:
	movq	%r12, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	.LC1(%rip), %ymm2
	testb	%al, %al
	jne	.L571
	vzeroupper
.L569:
	call	GOMP_loop_end_nowait@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L587
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L587:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13938:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0, .-_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0:
.LFB13942:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%esi, %esi
	xorl	%edi, %edi
	movl	$1000, %ecx
	movl	$8, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %r12
	leaq	8(%rsp), %rbx
	movq	%r12, %r9
	movq	%rbx, %r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L589
	vbroadcastss	.LC0(%rip), %ymm3
.L591:
	movl	16(%rsp), %edx
	movslq	8(%rsp), %rax
	.p2align 4,,10
	.p2align 3
.L590:
	vmovups	0(,%rax,4), %ymm0
	vmulps	%ymm3, %ymm0, %ymm2
	vmulps	%ymm2, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm10
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm0, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm6
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm2
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm2, %ymm1
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm4
	vrcpps	%ymm9, %ymm10
	vmulps	%ymm9, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm13
	vmulps	%ymm11, %ymm10, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm0, %ymm15
	vaddps	%ymm4, %ymm15, %ymm2
	vmulps	%ymm3, %ymm2, %ymm6
	vmulps	%ymm6, %ymm6, %ymm5
	vaddps	%ymm6, %ymm6, %ymm12
	vrcpps	%ymm5, %ymm1
	vmulps	%ymm5, %ymm1, %ymm7
	vaddps	%ymm1, %ymm1, %ymm9
	vmulps	%ymm7, %ymm1, %ymm8
	vsubps	%ymm8, %ymm9, %ymm10
	vmulps	%ymm10, %ymm0, %ymm11
	vaddps	%ymm12, %ymm11, %ymm13
	vmulps	%ymm3, %ymm13, %ymm14
	vmulps	%ymm14, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm8
	vrcpps	%ymm15, %ymm2
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm5
	vmulps	%ymm4, %ymm2, %ymm6
	vsubps	%ymm6, %ymm5, %ymm1
	vmulps	%ymm1, %ymm0, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm4
	vrcpps	%ymm11, %ymm12
	vmulps	%ymm11, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm15
	vmulps	%ymm13, %ymm12, %ymm14
	vsubps	%ymm14, %ymm15, %ymm2
	vmulps	%ymm2, %ymm0, %ymm6
	vaddps	%ymm4, %ymm6, %ymm5
	vmulps	%ymm3, %ymm5, %ymm7
	vmulps	%ymm7, %ymm7, %ymm8
	vaddps	%ymm7, %ymm7, %ymm14
	vrcpps	%ymm8, %ymm1
	vmulps	%ymm8, %ymm1, %ymm9
	vaddps	%ymm1, %ymm1, %ymm11
	vmulps	%ymm9, %ymm1, %ymm10
	vsubps	%ymm10, %ymm11, %ymm12
	vmulps	%ymm12, %ymm0, %ymm13
	vaddps	%ymm14, %ymm13, %ymm15
	vmulps	%ymm3, %ymm15, %ymm6
	vmulps	%ymm6, %ymm6, %ymm2
	vaddps	%ymm6, %ymm6, %ymm10
	vrcpps	%ymm2, %ymm4
	vmulps	%ymm2, %ymm4, %ymm5
	vaddps	%ymm4, %ymm4, %ymm8
	vmulps	%ymm5, %ymm4, %ymm7
	vsubps	%ymm7, %ymm8, %ymm1
	vmulps	%ymm1, %ymm0, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm7
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm2
	vmulps	%ymm15, %ymm14, %ymm6
	vsubps	%ymm6, %ymm2, %ymm4
	vmulps	%ymm4, %ymm0, %ymm5
	vaddps	%ymm7, %ymm5, %ymm8
	vmulps	%ymm3, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm6
	vrcpps	%ymm10, %ymm1
	vmulps	%ymm10, %ymm1, %ymm11
	vaddps	%ymm1, %ymm1, %ymm13
	vmulps	%ymm11, %ymm1, %ymm12
	vsubps	%ymm12, %ymm13, %ymm14
	vmulps	%ymm14, %ymm0, %ymm15
	vaddps	%ymm6, %ymm15, %ymm2
	vmulps	%ymm3, %ymm2, %ymm7
	vmulps	%ymm7, %ymm7, %ymm4
	vaddps	%ymm7, %ymm7, %ymm12
	vrcpps	%ymm4, %ymm8
	vmulps	%ymm4, %ymm8, %ymm5
	vaddps	%ymm8, %ymm8, %ymm10
	vmulps	%ymm5, %ymm8, %ymm9
	vsubps	%ymm9, %ymm10, %ymm1
	vmulps	%ymm1, %ymm0, %ymm11
	vaddps	%ymm12, %ymm11, %ymm13
	vmulps	%ymm3, %ymm13, %ymm14
	vmulps	%ymm14, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm9
	vrcpps	%ymm15, %ymm6
	vmulps	%ymm15, %ymm6, %ymm2
	vaddps	%ymm6, %ymm6, %ymm4
	vmulps	%ymm2, %ymm6, %ymm7
	vsubps	%ymm7, %ymm4, %ymm8
	vmulps	%ymm8, %ymm0, %ymm5
	vaddps	%ymm9, %ymm5, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm2
	vrcpps	%ymm12, %ymm1
	vmulps	%ymm12, %ymm1, %ymm13
	vaddps	%ymm1, %ymm1, %ymm15
	vmulps	%ymm13, %ymm1, %ymm14
	vsubps	%ymm14, %ymm15, %ymm6
	vmulps	%ymm6, %ymm0, %ymm7
	vaddps	%ymm2, %ymm7, %ymm4
	vmulps	%ymm3, %ymm4, %ymm8
	vmulps	%ymm8, %ymm8, %ymm5
	vaddps	%ymm8, %ymm8, %ymm14
	vrcpps	%ymm5, %ymm9
	vmulps	%ymm5, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm12
	vmulps	%ymm10, %ymm9, %ymm11
	vsubps	%ymm11, %ymm12, %ymm1
	vmulps	%ymm1, %ymm0, %ymm13
	vaddps	%ymm14, %ymm13, %ymm15
	vmulps	%ymm3, %ymm15, %ymm6
	vmulps	%ymm6, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm11
	vrcpps	%ymm7, %ymm2
	vmulps	%ymm7, %ymm2, %ymm4
	vaddps	%ymm2, %ymm2, %ymm5
	vmulps	%ymm4, %ymm2, %ymm8
	vsubps	%ymm8, %ymm5, %ymm9
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm1
	vmulps	%ymm1, %ymm1, %ymm13
	vaddps	%ymm1, %ymm1, %ymm4
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm7
	vmulps	%ymm15, %ymm14, %ymm6
	vsubps	%ymm6, %ymm7, %ymm2
	vmulps	%ymm2, %ymm0, %ymm0
	vaddps	%ymm4, %ymm0, %ymm8
	vmulps	%ymm3, %ymm8, %ymm5
	vmovups	%ymm5, 0(,%rax,4)
	addq	$8, %rax
	cmpl	%eax, %edx
	jg	.L590
	movq	%r12, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	.LC1(%rip), %ymm3
	testb	%al, %al
	jne	.L591
	vzeroupper
.L589:
	call	GOMP_loop_end_nowait@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L600
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L600:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13942:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0, .-_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0:
.LFB13946:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%esi, %esi
	xorl	%edi, %edi
	movl	$1000, %ecx
	movl	$8, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %r12
	leaq	8(%rsp), %rbx
	movq	%r12, %r9
	movq	%rbx, %r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L602
	vbroadcastss	.LC0(%rip), %ymm3
.L605:
	movl	16(%rsp), %ecx
	movslq	8(%rsp), %rdx
	.p2align 4,,10
	.p2align 3
.L604:
	vmovups	0(,%rdx,4), %ymm2
	movl	$18, %eax
	vmulps	%ymm3, %ymm2, %ymm10
.L603:
	vmulps	%ymm10, %ymm10, %ymm4
	vaddps	%ymm10, %ymm10, %ymm0
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm0, %ymm9, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm6
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm4
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm4, %ymm1
	vmulps	%ymm1, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm15
	vrcpps	%ymm9, %ymm0
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm0, %ymm0, %ymm12
	vmulps	%ymm10, %ymm0, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm1
	vmulps	%ymm1, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm11
	vrcpps	%ymm5, %ymm6
	vmulps	%ymm5, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm9
	vmulps	%ymm7, %ymm6, %ymm8
	vsubps	%ymm8, %ymm9, %ymm0
	vmulps	%ymm0, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm8
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm5
	vsubps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm6, %ymm2, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm5
	vrcpps	%ymm11, %ymm0
	vmulps	%ymm11, %ymm0, %ymm12
	vaddps	%ymm0, %ymm0, %ymm14
	vmulps	%ymm12, %ymm0, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm5, %ymm4, %ymm1
	vmulps	%ymm3, %ymm1, %ymm10
	subl	$6, %eax
	jne	.L603
	vmovups	%ymm10, 0(,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L604
	movq	%r12, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	.LC1(%rip), %ymm3
	testb	%al, %al
	jne	.L605
	vzeroupper
.L602:
	call	GOMP_loop_end_nowait@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L620
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L620:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13946:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0, .-_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0:
.LFB13950:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%esi, %esi
	xorl	%edi, %edi
	movl	$1000, %ecx
	movl	$8, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %r12
	leaq	8(%rsp), %rbx
	movq	%r12, %r9
	movq	%rbx, %r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L622
	vbroadcastss	.LC0(%rip), %ymm3
.L625:
	movl	16(%rsp), %ecx
	movslq	8(%rsp), %rdx
.L624:
	vmovups	0(,%rdx,4), %ymm2
	movl	$98, %eax
	vmulps	%ymm3, %ymm2, %ymm4
	vmulps	%ymm4, %ymm4, %ymm1
	vaddps	%ymm4, %ymm4, %ymm10
	vrcpps	%ymm1, %ymm0
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm7
	vmulps	%ymm5, %ymm0, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm12
	vmulps	%ymm12, %ymm12, %ymm13
	vaddps	%ymm12, %ymm12, %ymm6
	vrcpps	%ymm13, %ymm14
	vmulps	%ymm13, %ymm14, %ymm15
	vaddps	%ymm14, %ymm14, %ymm1
	vmulps	%ymm15, %ymm14, %ymm4
	vsubps	%ymm4, %ymm1, %ymm0
	vmulps	%ymm0, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm0
	.p2align 4,,10
	.p2align 3
.L623:
	vmulps	%ymm0, %ymm0, %ymm8
	vaddps	%ymm0, %ymm0, %ymm15
	vrcpps	%ymm8, %ymm9
	vmulps	%ymm8, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm12
	vmulps	%ymm10, %ymm9, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm6
	vmulps	%ymm6, %ymm6, %ymm1
	vaddps	%ymm6, %ymm6, %ymm11
	vrcpps	%ymm1, %ymm0
	vmulps	%ymm1, %ymm0, %ymm5
	vaddps	%ymm0, %ymm0, %ymm8
	vmulps	%ymm5, %ymm0, %ymm7
	vsubps	%ymm7, %ymm8, %ymm9
	vmulps	%ymm9, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm7
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm6
	vsubps	%ymm6, %ymm1, %ymm0
	vmulps	%ymm0, %ymm2, %ymm5
	vaddps	%ymm7, %ymm5, %ymm8
	vmulps	%ymm3, %ymm8, %ymm9
	vmulps	%ymm9, %ymm9, %ymm10
	vaddps	%ymm9, %ymm9, %ymm4
	vrcpps	%ymm10, %ymm11
	vmulps	%ymm10, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm14
	vmulps	%ymm12, %ymm11, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm6
	vaddps	%ymm4, %ymm6, %ymm1
	vmulps	%ymm3, %ymm1, %ymm7
	vmulps	%ymm7, %ymm7, %ymm5
	vaddps	%ymm7, %ymm7, %ymm13
	vrcpps	%ymm5, %ymm0
	vmulps	%ymm5, %ymm0, %ymm8
	vaddps	%ymm0, %ymm0, %ymm10
	vmulps	%ymm8, %ymm0, %ymm9
	vsubps	%ymm9, %ymm10, %ymm11
	vmulps	%ymm11, %ymm2, %ymm12
	vaddps	%ymm13, %ymm12, %ymm14
	vmulps	%ymm3, %ymm14, %ymm15
	vmulps	%ymm15, %ymm15, %ymm6
	vaddps	%ymm15, %ymm15, %ymm9
	vrcpps	%ymm6, %ymm4
	vmulps	%ymm6, %ymm4, %ymm1
	vaddps	%ymm4, %ymm4, %ymm5
	vmulps	%ymm1, %ymm4, %ymm7
	vsubps	%ymm7, %ymm5, %ymm0
	vmulps	%ymm0, %ymm2, %ymm8
	vaddps	%ymm9, %ymm8, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm1
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm6
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm6, %ymm4
	vmulps	%ymm4, %ymm2, %ymm7
	vaddps	%ymm1, %ymm7, %ymm5
	vmulps	%ymm3, %ymm5, %ymm0
	subl	$7, %eax
	jne	.L623
	vmovups	%ymm0, 0(,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L624
	movq	%r12, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	.LC1(%rip), %ymm3
	testb	%al, %al
	jne	.L625
	vzeroupper
.L622:
	call	GOMP_loop_end_nowait@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L643
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L643:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13950:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0, .-_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0:
.LFB13954:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%esi, %esi
	xorl	%edi, %edi
	movl	$1000, %ecx
	movl	$8, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %r12
	leaq	8(%rsp), %rbx
	movq	%r12, %r9
	movq	%rbx, %r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L645
	vbroadcastss	.LC0(%rip), %ymm3
.L648:
	movl	16(%rsp), %ecx
	movslq	8(%rsp), %rdx
.L647:
	vmovups	0(,%rdx,4), %ymm2
	movl	$1000, %eax
	vmulps	%ymm3, %ymm2, %ymm0
	.p2align 4,,10
	.p2align 3
.L646:
	vmulps	%ymm0, %ymm0, %ymm4
	vaddps	%ymm0, %ymm0, %ymm0
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm0, %ymm9, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm6
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm4
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm4, %ymm1
	vmulps	%ymm1, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm15
	vrcpps	%ymm9, %ymm0
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm0, %ymm0, %ymm12
	vmulps	%ymm10, %ymm0, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm1
	vmulps	%ymm1, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm11
	vrcpps	%ymm5, %ymm6
	vmulps	%ymm5, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm9
	vmulps	%ymm7, %ymm6, %ymm8
	vsubps	%ymm8, %ymm9, %ymm0
	vmulps	%ymm0, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm8
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm5
	vsubps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm6, %ymm2, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm5
	vrcpps	%ymm11, %ymm0
	vmulps	%ymm11, %ymm0, %ymm12
	vaddps	%ymm0, %ymm0, %ymm14
	vmulps	%ymm12, %ymm0, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm5, %ymm4, %ymm1
	vmulps	%ymm3, %ymm1, %ymm6
	vmulps	%ymm6, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm13
	vrcpps	%ymm7, %ymm8
	vmulps	%ymm7, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm11
	vmulps	%ymm9, %ymm8, %ymm10
	vsubps	%ymm10, %ymm11, %ymm0
	vmulps	%ymm0, %ymm2, %ymm12
	vaddps	%ymm13, %ymm12, %ymm14
	vmulps	%ymm3, %ymm14, %ymm15
	vmulps	%ymm15, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm10
	vrcpps	%ymm4, %ymm5
	vmulps	%ymm4, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm7
	vmulps	%ymm1, %ymm5, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm0
	subl	$8, %eax
	jne	.L646
	vmovups	%ymm0, 0(,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L647
	movq	%r12, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	.LC1(%rip), %ymm3
	testb	%al, %al
	jne	.L648
	vzeroupper
.L645:
	call	GOMP_loop_end_nowait@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L665
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L665:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13954:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, .-_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0
	.section	.text._Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, @function
_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0:
.LFB13958:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%esi, %esi
	xorl	%edi, %edi
	movl	$1000, %ecx
	movl	$8, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %r12
	leaq	8(%rsp), %rbx
	movq	%r12, %r9
	movq	%rbx, %r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L667
	vbroadcastss	.LC0(%rip), %ymm3
.L670:
	movl	16(%rsp), %ecx
	movslq	8(%rsp), %rdx
.L669:
	vmovups	0(,%rdx,4), %ymm2
	movl	$10000, %eax
	vmulps	%ymm3, %ymm2, %ymm0
	.p2align 4,,10
	.p2align 3
.L668:
	vmulps	%ymm0, %ymm0, %ymm4
	vaddps	%ymm0, %ymm0, %ymm0
	vrcpps	%ymm4, %ymm1
	vmulps	%ymm4, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm7
	vmulps	%ymm5, %ymm1, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm0, %ymm9, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vmulps	%ymm11, %ymm11, %ymm12
	vaddps	%ymm11, %ymm11, %ymm6
	vrcpps	%ymm12, %ymm13
	vmulps	%ymm12, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm4
	vmulps	%ymm14, %ymm13, %ymm15
	vsubps	%ymm15, %ymm4, %ymm1
	vmulps	%ymm1, %ymm2, %ymm5
	vaddps	%ymm6, %ymm5, %ymm7
	vmulps	%ymm3, %ymm7, %ymm8
	vmulps	%ymm8, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm15
	vrcpps	%ymm9, %ymm0
	vmulps	%ymm9, %ymm0, %ymm10
	vaddps	%ymm0, %ymm0, %ymm12
	vmulps	%ymm10, %ymm0, %ymm11
	vsubps	%ymm11, %ymm12, %ymm13
	vmulps	%ymm13, %ymm2, %ymm14
	vaddps	%ymm15, %ymm14, %ymm4
	vmulps	%ymm3, %ymm4, %ymm1
	vmulps	%ymm1, %ymm1, %ymm5
	vaddps	%ymm1, %ymm1, %ymm11
	vrcpps	%ymm5, %ymm6
	vmulps	%ymm5, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm9
	vmulps	%ymm7, %ymm6, %ymm8
	vsubps	%ymm8, %ymm9, %ymm0
	vmulps	%ymm0, %ymm2, %ymm10
	vaddps	%ymm11, %ymm10, %ymm12
	vmulps	%ymm3, %ymm12, %ymm13
	vmulps	%ymm13, %ymm13, %ymm14
	vaddps	%ymm13, %ymm13, %ymm8
	vrcpps	%ymm14, %ymm15
	vmulps	%ymm14, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm1
	vmulps	%ymm4, %ymm15, %ymm5
	vsubps	%ymm5, %ymm1, %ymm6
	vmulps	%ymm6, %ymm2, %ymm7
	vaddps	%ymm8, %ymm7, %ymm9
	vmulps	%ymm3, %ymm9, %ymm10
	vmulps	%ymm10, %ymm10, %ymm11
	vaddps	%ymm10, %ymm10, %ymm5
	vrcpps	%ymm11, %ymm0
	vmulps	%ymm11, %ymm0, %ymm12
	vaddps	%ymm0, %ymm0, %ymm14
	vmulps	%ymm12, %ymm0, %ymm13
	vsubps	%ymm13, %ymm14, %ymm15
	vmulps	%ymm15, %ymm2, %ymm4
	vaddps	%ymm5, %ymm4, %ymm1
	vmulps	%ymm3, %ymm1, %ymm6
	vmulps	%ymm6, %ymm6, %ymm7
	vaddps	%ymm6, %ymm6, %ymm13
	vrcpps	%ymm7, %ymm8
	vmulps	%ymm7, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm11
	vmulps	%ymm9, %ymm8, %ymm10
	vsubps	%ymm10, %ymm11, %ymm0
	vmulps	%ymm0, %ymm2, %ymm12
	vaddps	%ymm13, %ymm12, %ymm14
	vmulps	%ymm3, %ymm14, %ymm15
	vmulps	%ymm15, %ymm15, %ymm4
	vaddps	%ymm15, %ymm15, %ymm10
	vrcpps	%ymm4, %ymm5
	vmulps	%ymm4, %ymm5, %ymm1
	vaddps	%ymm5, %ymm5, %ymm7
	vmulps	%ymm1, %ymm5, %ymm6
	vsubps	%ymm6, %ymm7, %ymm8
	vmulps	%ymm8, %ymm2, %ymm9
	vaddps	%ymm10, %ymm9, %ymm11
	vmulps	%ymm3, %ymm11, %ymm0
	subl	$8, %eax
	jne	.L668
	vmovups	%ymm0, 0(,%rdx,4)
	addq	$8, %rdx
	cmpl	%edx, %ecx
	jg	.L669
	movq	%r12, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	.LC1(%rip), %ymm3
	testb	%al, %al
	jne	.L670
	vzeroupper
.L667:
	call	GOMP_loop_end_nowait@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L687
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L687:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13958:
	.size	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, .-_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0
	.text
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0, @function
_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0:
.LFB14027:
	.cfi_startproc
	ret
	.cfi_endproc
.LFE14027:
	.size	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0, .-_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.section	.text._Z27parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z27parallel_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z27parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, @function
_Z27parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0:
.LFB14024:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14024:
	.size	_Z27parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, .-_Z27parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0
	.section	.text._Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, @function
_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi._omp_fn.0:
.LFB14022:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14022:
	.size	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, .-_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi._omp_fn.0
	.section	.text._Z27parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z27parallel_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z27parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, @function
_Z27parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0:
.LFB14020:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14020:
	.size	_Z27parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, .-_Z27parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0
	.section	.text._Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, @function
_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi._omp_fn.0:
.LFB14018:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14018:
	.size	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, .-_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi._omp_fn.0
	.section	.text._Z27parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z27parallel_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.type	_Z27parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0, @function
_Z27parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0:
.LFB14016:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14016:
	.size	_Z27parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0, .-_Z27parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0
	.section	.text._Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi._omp_fn.0, @function
_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi._omp_fn.0:
.LFB14014:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14014:
	.size	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi._omp_fn.0, .-_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi._omp_fn.0
	.section	.text._Z27parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z27parallel_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.type	_Z27parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0, @function
_Z27parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0:
.LFB14012:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14012:
	.size	_Z27parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0, .-_Z27parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0
	.section	.text._Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi._omp_fn.0, @function
_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi._omp_fn.0:
.LFB14010:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14010:
	.size	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi._omp_fn.0, .-_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi._omp_fn.0
	.section	.text._Z27parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z27parallel_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.type	_Z27parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0, @function
_Z27parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0:
.LFB14008:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14008:
	.size	_Z27parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0, .-_Z27parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0
	.section	.text._Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.type	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi._omp_fn.0, @function
_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi._omp_fn.0:
.LFB14006:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14006:
	.size	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi._omp_fn.0, .-_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi._omp_fn.0
	.section	.text._Z27parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z27parallel_cube_root_templateILi2EEvPKfPfi,comdat
	.p2align 4
	.type	_Z27parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0, @function
_Z27parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0:
.LFB14004:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE14004:
	.size	_Z27parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0, .-_Z27parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0
	.text
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0:
.LFB14026:
	.cfi_startproc
	jmp	sched_getcpu@PLT
	.cfi_endproc
.LFE14026:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0, .-_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0:
.LFB14002:
	.cfi_startproc
	endbr64
	jmp	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.cfi_endproc
.LFE14002:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0, .-_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi._omp_fn.0
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi,comdat
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0:
.LFB14000:
	.cfi_startproc
	endbr64
	jmp	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.cfi_endproc
.LFE14000:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0, .-_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi._omp_fn.0
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi,comdat
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0:
.LFB13998:
	.cfi_startproc
	endbr64
	jmp	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.cfi_endproc
.LFE13998:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0, .-_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi._omp_fn.0
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi,comdat
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0:
.LFB13996:
	.cfi_startproc
	endbr64
	jmp	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.cfi_endproc
.LFE13996:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0, .-_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi._omp_fn.0
	.section	.text._Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0,"axG",@progbits,_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi,comdat
	.p2align 4
	.type	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0, @function
_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0:
.LFB13994:
	.cfi_startproc
	endbr64
	jmp	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi._omp_fn.0.constprop.0
	.cfi_endproc
.LFE13994:
	.size	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0, .-_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi._omp_fn.0
	.text
	.p2align 4
	.type	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0, @function
_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0:
.LFB14028:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movslq	%r8d, %r15
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movl	%r15d, %eax
	movq	%r15, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	sarl	$31, %eax
	movl	%ecx, %r13d
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdx, %rbx
	imulq	$1717986919, %r15, %rdx
	sarq	$34, %rdx
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	subl	%eax, %edx
	movl	%edx, (%rsp)
	cmpl	$9, %r15d
	jle	.L707
	leal	-1(%rdx), %ecx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	movl	%r13d, %edx
	andl	$7, %ecx
	movl	$1, %r15d
	movl	%ecx, 12(%rsp)
	call	*%rbp
	movl	(%rsp), %esi
	movl	12(%rsp), %edi
	cmpl	%esi, %r15d
	jge	.L785
	testl	%edi, %edi
	je	.L708
	cmpl	$1, %edi
	je	.L765
	cmpl	$2, %edi
	je	.L766
	cmpl	$3, %edi
	je	.L767
	cmpl	$4, %edi
	je	.L768
	cmpl	$5, %edi
	je	.L769
	cmpl	$6, %edi
	je	.L770
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	movl	$2, %r15d
	call	*%rbp
.L770:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L769:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L768:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L767:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L766:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L765:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	movl	(%rsp), %r8d
	cmpl	%r8d, %r15d
	jge	.L785
.L708:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$8, %r15d
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	(%rsp), %r9d
	cmpl	%r9d, %r15d
	jl	.L708
.L785:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movl	$0x00000000, 28(%rsp)
	movq	%rax, (%rsp)
.L712:
	movl	%r14d, %r10d
	xorl	%r15d, %r15d
	andl	$7, %r10d
	je	.L711
	cmpl	$1, %r10d
	je	.L759
	cmpl	$2, %r10d
	je	.L760
	cmpl	$3, %r10d
	je	.L761
	cmpl	$4, %r10d
	je	.L762
	cmpl	$5, %r10d
	je	.L763
	cmpl	$6, %r10d
	jne	.L789
.L764:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm2
	vaddss	(%rbx), %xmm2, %xmm3
	vmovss	%xmm3, 28(%rsp)
.L763:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm4
	vaddss	(%rbx), %xmm4, %xmm5
	vmovss	%xmm5, 28(%rsp)
.L762:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm6
	vaddss	(%rbx), %xmm6, %xmm7
	vmovss	%xmm7, 28(%rsp)
.L761:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm8
	vaddss	(%rbx), %xmm8, %xmm9
	vmovss	%xmm9, 28(%rsp)
.L760:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm10
	vaddss	(%rbx), %xmm10, %xmm11
	vmovss	%xmm11, 28(%rsp)
.L759:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm12
	vaddss	(%rbx), %xmm12, %xmm13
	vmovss	%xmm13, 28(%rsp)
	cmpl	%r15d, %r14d
	je	.L710
.L711:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$8, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm14
	vaddss	(%rbx), %xmm14, %xmm15
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm15, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm0
	vaddss	(%rbx), %xmm0, %xmm1
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm1, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm2
	vaddss	(%rbx), %xmm2, %xmm3
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm3, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm4
	vaddss	(%rbx), %xmm4, %xmm5
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm5, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm6
	vaddss	(%rbx), %xmm6, %xmm7
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm7, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm8
	vaddss	(%rbx), %xmm8, %xmm9
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm9, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm10
	vaddss	(%rbx), %xmm10, %xmm11
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm11, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm12
	vaddss	(%rbx), %xmm12, %xmm13
	vmovss	%xmm13, 28(%rsp)
	cmpl	%r15d, %r14d
	jne	.L711
.L710:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	(%rsp), %r11
	vxorpd	%xmm14, %xmm14, %xmm14
	subq	%r11, %rax
	vcvtsi2sdq	%rax, %xmm14, %xmm15
	vmulsd	.LC8(%rip), %xmm15, %xmm0
	vmovsd	%xmm0, (%rsp)
	call	omp_get_max_threads@PLT
	movl	%eax, %edi
	call	omp_set_num_threads@PLT
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	(%rsp), %xmm3
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	vcvtsi2sdl	%r14d, %xmm1, %xmm2
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	vdivsd	%xmm2, %xmm3, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L789:
	.cfi_restore_state
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	movl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm0
	vaddss	(%rbx), %xmm0, %xmm1
	vmovss	%xmm1, 28(%rsp)
	jmp	.L764
	.p2align 4,,10
	.p2align 3
.L707:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movl	$0x00000000, 28(%rsp)
	movq	%rax, (%rsp)
	testl	%r15d, %r15d
	jle	.L710
	jmp	.L712
	.cfi_endproc
.LFE14028:
	.size	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0, .-_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	.align 2
	.p2align 4
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0:
.LFB14029:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	leaq	16(%rbp), %rax
	movq	%rsi, %rbx
	leaq	16(%rbx), %r8
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movq	(%rdi), %rdi
	movq	(%rsi), %rsi
	movq	8(%rbx), %rdx
	cmpq	%rax, %rdi
	je	.L806
	cmpq	%rsi, %r8
	je	.L792
	movq	%rsi, 0(%rbp)
	movq	16(%rbp), %rcx
	movq	%rdx, 8(%rbp)
	movq	16(%rbx), %rdx
	movq	%rdx, 16(%rbp)
	testq	%rdi, %rdi
	je	.L800
	movq	%rdi, (%rbx)
	movq	%rcx, 16(%rbx)
.L795:
	movq	$0, 8(%rbx)
	movb	$0, (%rdi)
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L806:
	.cfi_restore_state
	cmpq	%rsi, %r8
	je	.L792
	movq	%rsi, 0(%rbp)
	movq	%r8, %rdi
	movq	%rdx, 8(%rbp)
	movq	16(%rbx), %rsi
	movq	%rsi, 16(%rbp)
.L798:
	movq	%rdi, (%rbx)
	jmp	.L795
	.p2align 4,,10
	.p2align 3
.L792:
	cmpq	%rbx, %rbp
	je	.L799
	testq	%rdx, %rdx
	je	.L796
	cmpq	$1, %rdx
	je	.L807
	call	memcpy@PLT
	movq	8(%rbx), %rdx
	movq	0(%rbp), %rdi
.L796:
	movq	%rdx, 8(%rbp)
	movb	$0, (%rdi,%rdx)
	movq	(%rbx), %rdi
	jmp	.L795
	.p2align 4,,10
	.p2align 3
.L800:
	movq	%r8, %rdi
	jmp	.L798
	.p2align 4,,10
	.p2align 3
.L807:
	movzbl	(%rsi), %r9d
	movb	%r9b, (%rdi)
	movq	8(%rbx), %rdx
	movq	0(%rbp), %rdi
	jmp	.L796
	.p2align 4,,10
	.p2align 3
.L799:
	movq	%rsi, %rdi
	jmp	.L795
	.cfi_endproc
.LFE14029:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC9:
	.string	"basic_string_view::substr"
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC10:
	.string	"%s: __pos (which is %zu) > __size (which is %zu)"
	.text
	.align 2
	.p2align 4
	.type	_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0, @function
_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0:
.LFB14031:
	.cfi_startproc
	cmpq	%rdx, %rdi
	jb	.L813
	subq	%rdx, %rdi
	movq	%rcx, %rax
	cmpq	%rcx, %rdi
	cmovbe	%rdi, %rax
	addq	%rsi, %rdx
	ret
.L813:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rdi, %rcx
	leaq	.LC9(%rip), %rsi
	xorl	%eax, %eax
	leaq	.LC10(%rip), %rdi
	call	_ZSt24__throw_out_of_range_fmtPKcz@PLT
	.cfi_endproc
.LFE14031:
	.size	_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0, .-_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0
	.p2align 4
	.type	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, @function
_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0:
.LFB14033:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movq	(%rdi), %rax
	movq	-24(%rax), %rdx
	movq	240(%rdi,%rdx), %rbp
	testq	%rbp, %rbp
	je	.L820
	cmpb	$0, 56(%rbp)
	movq	%rdi, %rbx
	je	.L816
	movsbl	67(%rbp), %esi
.L817:
	movq	%rbx, %rdi
	call	_ZNSo3putEc@PLT
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	movq	%rax, %rdi
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	_ZNSo5flushEv@PLT
	.p2align 4,,10
	.p2align 3
.L816:
	.cfi_restore_state
	movq	%rbp, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	0(%rbp), %rcx
	movl	$10, %esi
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rdi
	movq	48(%rcx), %rax
	cmpq	%rdi, %rax
	je	.L817
	movq	%rbp, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L817
.L820:
	call	_ZSt16__throw_bad_castv@PLT
	.cfi_endproc
.LFE14033:
	.size	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, .-_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	.section	.text._ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv,"axG",@progbits,_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv
	.type	_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv, @function
_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv:
.LFB13500:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$8, %rsp
	.cfi_def_cfa_offset 64
	movq	24(%rdi), %r13
	movq	8(%rdi), %rbp
	movq	296(%rdi), %rax
	movq	%r13, %r14
	subq	%rbp, %r14
	testq	%rax, %rax
	js	.L879
	movq	304(%rdi), %r15
	cmpq	%rax, %r15
	jnb	.L826
	subq	%r15, %rax
	movq	288(%rdi), %rbx
	cmpq	%r14, %rax
	cmova	%r14, %rax
	testq	%rax, %rax
	jle	.L827
	leaq	0(%rbp,%rax), %r15
	andl	$3, %eax
	je	.L829
	cmpq	$1, %rax
	je	.L864
	cmpq	$2, %rax
	je	.L865
	movq	24(%rbx), %rsi
	movzbl	0(%rbp), %edx
	leaq	1(%rsi), %rcx
	movq	%rcx, 24(%rbx)
	movb	%dl, (%rsi)
	movq	24(%rbx), %rdi
	subq	8(%rbx), %rdi
	cmpq	16(%rbx), %rdi
	je	.L880
.L847:
	addq	$1, %rbp
.L865:
	movq	24(%rbx), %r10
	movzbl	0(%rbp), %r9d
	leaq	1(%r10), %r11
	movq	%r11, 24(%rbx)
	movb	%r9b, (%r10)
	movq	24(%rbx), %r13
	subq	8(%rbx), %r13
	cmpq	16(%rbx), %r13
	je	.L881
.L850:
	addq	$1, %rbp
.L864:
	movq	24(%rbx), %rsi
	movzbl	0(%rbp), %edx
	leaq	1(%rsi), %rcx
	movq	%rcx, 24(%rbx)
	movb	%dl, (%rsi)
	movq	24(%rbx), %rdi
	subq	8(%rbx), %rdi
	cmpq	16(%rbx), %rdi
	je	.L882
.L853:
	addq	$1, %rbp
	cmpq	%r15, %rbp
	je	.L877
.L829:
	movq	24(%rbx), %r10
	movzbl	0(%rbp), %r9d
	leaq	1(%r10), %r11
	movq	%r11, 24(%rbx)
	movb	%r9b, (%r10)
	movq	24(%rbx), %r13
	subq	8(%rbx), %r13
	cmpq	16(%rbx), %r13
	je	.L883
.L828:
	movq	24(%rbx), %rdx
	leaq	1(%rbp), %r13
	movzbl	1(%rbp), %ebp
	leaq	1(%rdx), %rsi
	movq	%rsi, 24(%rbx)
	movb	%bpl, (%rdx)
	movq	24(%rbx), %rcx
	subq	8(%rbx), %rcx
	cmpq	16(%rbx), %rcx
	je	.L884
.L856:
	movq	24(%rbx), %r9
	movzbl	1(%r13), %edi
	leaq	1(%r9), %r10
	movq	%r10, 24(%rbx)
	movb	%dil, (%r9)
	movq	24(%rbx), %r11
	subq	8(%rbx), %r11
	cmpq	16(%rbx), %r11
	je	.L885
.L858:
	movq	24(%rbx), %rdx
	movzbl	2(%r13), %ebp
	leaq	1(%rdx), %rsi
	movq	%rsi, 24(%rbx)
	movb	%bpl, (%rdx)
	movq	24(%rbx), %rcx
	subq	8(%rbx), %rcx
	cmpq	16(%rbx), %rcx
	je	.L886
.L860:
	leaq	3(%r13), %rbp
	cmpq	%r15, %rbp
	jne	.L829
.L877:
	movq	8(%r12), %rbp
	movq	304(%r12), %r15
.L827:
	movq	%rbx, 288(%r12)
.L826:
	addq	%r14, %r15
	movq	%rbp, 24(%r12)
	movq	%r15, 304(%r12)
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L879:
	.cfi_restore_state
	movq	288(%rdi), %rbx
	testq	%r14, %r14
	jle	.L823
	movq	%r14, %rdi
	andl	$3, %edi
	je	.L825
	cmpq	$1, %rdi
	je	.L862
	cmpq	$2, %rdi
	je	.L863
	movq	24(%rbx), %r10
	movzbl	0(%rbp), %r9d
	leaq	1(%r10), %r11
	movq	%r11, 24(%rbx)
	movb	%r9b, (%r10)
	movq	24(%rbx), %rax
	subq	8(%rbx), %rax
	cmpq	16(%rbx), %rax
	je	.L887
.L832:
	addq	$1, %rbp
.L863:
	movq	24(%rbx), %r8
	movzbl	0(%rbp), %esi
	leaq	1(%r8), %rcx
	movq	%rcx, 24(%rbx)
	movb	%sil, (%r8)
	movq	24(%rbx), %r15
	subq	8(%rbx), %r15
	cmpq	16(%rbx), %r15
	je	.L888
.L835:
	addq	$1, %rbp
.L862:
	movq	24(%rbx), %r10
	movzbl	0(%rbp), %edi
	leaq	1(%r10), %r11
	movq	%r11, 24(%rbx)
	movb	%dil, (%r10)
	movq	24(%rbx), %rax
	subq	8(%rbx), %rax
	cmpq	16(%rbx), %rax
	je	.L889
.L838:
	addq	$1, %rbp
	cmpq	%rbp, %r13
	je	.L876
.L825:
	movq	24(%rbx), %r8
	movzbl	0(%rbp), %esi
	leaq	1(%r8), %rcx
	movq	%rcx, 24(%rbx)
	movb	%sil, (%r8)
	movq	24(%rbx), %r15
	subq	8(%rbx), %r15
	cmpq	16(%rbx), %r15
	je	.L890
.L824:
	movq	24(%rbx), %rdi
	leaq	1(%rbp), %r15
	movzbl	1(%rbp), %ebp
	leaq	1(%rdi), %r10
	movq	%r10, 24(%rbx)
	movb	%bpl, (%rdi)
	movq	24(%rbx), %r11
	subq	8(%rbx), %r11
	cmpq	16(%rbx), %r11
	je	.L891
.L841:
	movq	24(%rbx), %rsi
	movzbl	1(%r15), %edx
	leaq	1(%rsi), %r8
	movq	%r8, 24(%rbx)
	movb	%dl, (%rsi)
	movq	24(%rbx), %rcx
	subq	8(%rbx), %rcx
	cmpq	16(%rbx), %rcx
	je	.L892
.L843:
	movq	24(%rbx), %rdi
	movzbl	2(%r15), %ebp
	leaq	1(%rdi), %r10
	movq	%r10, 24(%rbx)
	movb	%bpl, (%rdi)
	movq	24(%rbx), %r11
	subq	8(%rbx), %r11
	cmpq	16(%rbx), %r11
	je	.L893
.L845:
	leaq	3(%r15), %rbp
	cmpq	%rbp, %r13
	jne	.L825
.L876:
	movq	8(%r12), %rbp
.L823:
	movq	%rbx, 288(%r12)
	movq	304(%r12), %r15
	jmp	.L826
	.p2align 4,,10
	.p2align 3
.L890:
	movq	(%rbx), %r9
	movq	%rbx, %rdi
	call	*(%r9)
	jmp	.L824
	.p2align 4,,10
	.p2align 3
.L893:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	call	*(%rax)
	jmp	.L845
	.p2align 4,,10
	.p2align 3
.L892:
	movq	(%rbx), %r9
	movq	%rbx, %rdi
	call	*(%r9)
	jmp	.L843
	.p2align 4,,10
	.p2align 3
.L891:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	call	*(%rax)
	jmp	.L841
	.p2align 4,,10
	.p2align 3
.L886:
	movq	(%rbx), %r8
	movq	%rbx, %rdi
	call	*(%r8)
	jmp	.L860
	.p2align 4,,10
	.p2align 3
.L885:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	call	*(%rax)
	jmp	.L858
	.p2align 4,,10
	.p2align 3
.L884:
	movq	(%rbx), %r8
	movq	%rbx, %rdi
	call	*(%r8)
	jmp	.L856
	.p2align 4,,10
	.p2align 3
.L883:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	call	*(%rax)
	jmp	.L828
.L889:
	movq	(%rbx), %rdx
	movq	%rbx, %rdi
	call	*(%rdx)
	jmp	.L838
.L882:
	movq	(%rbx), %r8
	movq	%rbx, %rdi
	call	*(%r8)
	jmp	.L853
.L888:
	movq	(%rbx), %r9
	movq	%rbx, %rdi
	call	*(%r9)
	jmp	.L835
.L881:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	call	*(%rax)
	jmp	.L850
.L887:
	movq	(%rbx), %rdx
	movq	%rbx, %rdi
	call	*(%rdx)
	jmp	.L832
.L880:
	movq	(%rbx), %r8
	movq	%rbx, %rdi
	call	*(%r8)
	jmp	.L847
	.cfi_endproc
.LFE13500:
	.size	_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv, .-_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv
	.section	.rodata.str1.1
.LC11:
	.string	"basic_string::_M_replace_aux"
.LC12:
	.string	"basic_string::_M_create"
	.text
	.align 2
	.p2align 4
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0:
.LFB14035:
	.cfi_startproc
	movabsq	$9223372036854775807, %rax
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	addq	%rdx, %rax
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	movq	8(%rdi), %r13
	movl	%r8d, 4(%rsp)
	subq	%r13, %rax
	cmpq	%rcx, %rax
	jb	.L933
	movq	%rcx, %r8
	movq	(%rdi), %rax
	leaq	16(%rdi), %r10
	movq	%rdi, %rbx
	subq	%rdx, %r8
	movq	%rsi, %r9
	movq	%rdx, %rbp
	movq	%rcx, %r12
	leaq	(%r8,%r13), %r14
	cmpq	%r10, %rax
	je	.L934
	movq	16(%rdi), %r15
	cmpq	%r14, %r15
	jb	.L935
.L897:
	leaq	0(%rbp,%r9), %rdi
	subq	%rdi, %r13
	je	.L901
	cmpq	%r12, %rbp
	je	.L901
	addq	%r9, %rax
	leaq	(%rax,%rbp), %rsi
	leaq	(%rax,%r12), %rdi
	cmpq	$1, %r13
	je	.L936
	movq	%r13, %rdx
	movq	%r9, 8(%rsp)
	call	memmove@PLT
	movq	(%rbx), %rax
	movq	8(%rsp), %r9
.L901:
	testq	%r12, %r12
	je	.L914
.L941:
	leaq	(%rax,%r9), %rdi
	cmpq	$1, %r12
	je	.L937
	movsbl	4(%rsp), %esi
	movq	%r12, %rdx
	call	memset@PLT
	movq	(%rbx), %rax
.L914:
	movq	%r14, 8(%rbx)
	movb	$0, (%rax,%r14)
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L935:
	.cfi_restore_state
	testq	%r14, %r14
	js	.L898
	addq	%rsi, %rbp
	addq	%r15, %r15
	subq	%rbp, %r13
	cmpq	%r15, %r14
	jb	.L938
.L904:
	movq	%r14, %rdi
	movq	%r14, %r15
	addq	$1, %rdi
	js	.L905
.L906:
	movq	%r9, 16(%rsp)
	movq	%r10, 8(%rsp)
	call	_Znwm@PLT
	movq	16(%rsp), %r9
	movq	(%rbx), %r11
	movq	8(%rsp), %r8
	testq	%r9, %r9
	je	.L908
	cmpq	$1, %r9
	je	.L939
	movq	%r9, %rdx
	movq	%r11, %rsi
	movq	%rax, %rdi
	movq	%r8, 24(%rsp)
	movq	%r9, 16(%rsp)
	movq	%r11, 8(%rsp)
	call	memcpy@PLT
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movq	8(%rsp), %r11
.L908:
	testq	%r13, %r13
	jne	.L940
.L911:
	cmpq	%r11, %r8
	je	.L913
	movq	16(%rbx), %rcx
	movq	%r11, %rdi
	movq	%r9, 16(%rsp)
	movq	%rax, 8(%rsp)
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
	movq	16(%rsp), %r9
	movq	8(%rsp), %rax
.L913:
	movq	%rax, (%rbx)
	movq	%r15, 16(%rbx)
	testq	%r12, %r12
	je	.L914
	jmp	.L941
	.p2align 4,,10
	.p2align 3
.L940:
	leaq	(%r12,%r9), %rdi
	leaq	(%r11,%rbp), %rsi
	addq	%rax, %rdi
	cmpq	$1, %r13
	je	.L942
	movq	%r13, %rdx
	movq	%r9, 24(%rsp)
	movq	%r8, 8(%rsp)
	movq	%rax, 16(%rsp)
	call	memcpy@PLT
	movq	(%rbx), %r11
	movq	24(%rsp), %r9
	movq	16(%rsp), %rax
	movq	8(%rsp), %r8
	jmp	.L911
	.p2align 4,,10
	.p2align 3
.L938:
	leaq	1(%r15), %rdi
	testq	%r15, %r15
	jns	.L906
.L905:
	call	_ZSt17__throw_bad_allocv@PLT
	.p2align 4,,10
	.p2align 3
.L934:
	cmpq	$15, %r14
	jbe	.L897
	testq	%r14, %r14
	js	.L898
	addq	%rsi, %rbp
	subq	%rbp, %r13
	cmpq	$29, %r14
	ja	.L904
	movl	$31, %edi
	movl	$30, %r15d
	jmp	.L906
	.p2align 4,,10
	.p2align 3
.L937:
	movzbl	4(%rsp), %ebp
	movb	%bpl, (%rdi)
	movq	(%rbx), %rax
	jmp	.L914
	.p2align 4,,10
	.p2align 3
.L936:
	movzbl	(%rsi), %r13d
	movb	%r13b, (%rdi)
	movq	(%rbx), %rax
	testq	%r12, %r12
	je	.L914
	jmp	.L941
	.p2align 4,,10
	.p2align 3
.L942:
	movzbl	(%rsi), %esi
	movb	%sil, (%rdi)
	movq	(%rbx), %r11
	jmp	.L911
	.p2align 4,,10
	.p2align 3
.L939:
	movzbl	(%r11), %edx
	movb	%dl, (%rax)
	jmp	.L908
.L933:
	leaq	.LC11(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L898:
	leaq	.LC12(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE14035:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	.section	.text._ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc,"axG",@progbits,_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc
	.type	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc, @function
_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc:
.LFB13439:
	.cfi_startproc
	endbr64
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	48(%rdi), %r14
	movq	8(%rdi), %r13
	movq	16(%r14), %rbx
	subq	%r13, %rsi
	jne	.L958
.L944:
	movq	%rbx, 16(%r14)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L958:
	.cfi_restore_state
	movq	24(%rbx), %rdi
	movq	16(%rbx), %rbp
	movq	%rsi, %r12
	movq	%rdi, %rax
	subq	8(%rbx), %rax
	subq	%rax, %rbp
	cmpq	%rbp, %rsi
	jnb	.L949
	jmp	.L945
	.p2align 4,,10
	.p2align 3
.L960:
	movq	%r13, %rsi
	movq	%rbp, %rdx
	addq	%rbp, %r13
	subq	%rbp, %r12
	call	memcpy@PLT
	addq	%rbp, 24(%rbx)
.L957:
	movq	(%rbx), %rdx
	movq	%rbx, %rdi
	call	*(%rdx)
	movq	24(%rbx), %rdi
	movq	16(%rbx), %rbp
	movq	%rdi, %rcx
	subq	8(%rbx), %rcx
	subq	%rcx, %rbp
	cmpq	%rbp, %r12
	jb	.L959
.L949:
	testq	%rbp, %rbp
	jne	.L960
	movq	%rdi, 24(%rbx)
	jmp	.L957
	.p2align 4,,10
	.p2align 3
.L959:
	testq	%r12, %r12
	je	.L944
.L945:
	movq	%r12, %rdx
	movq	%r13, %rsi
	call	memcpy@PLT
	addq	%r12, 24(%rbx)
	movq	%rbx, 16(%r14)
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE13439:
	.size	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc, .-_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc
	.section	.rodata.str1.1
.LC13:
	.string	"basic_string::_M_replace"
	.section	.text.unlikely,"ax",@progbits
	.align 2
.LCOLDB14:
	.text
.LHOTB14:
	.align 2
	.p2align 4
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0:
.LFB14043:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdx, %rax
	movabsq	$9223372036854775807, %rdx
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	addq	%rax, %rdx
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	movq	8(%rdi), %r13
	subq	%r13, %rdx
	cmpq	%r8, %rdx
	jb	.L1010
	movq	%r8, %r15
	movq	%rsi, %r9
	movq	(%rdi), %rsi
	movq	%r8, %rbp
	subq	%rax, %r15
	leaq	16(%rdi), %r8
	movq	%rdi, %rbx
	addq	%r13, %r15
	cmpq	%r8, %rsi
	je	.L1011
	movq	16(%rdi), %r14
	cmpq	%r15, %r14
	jb	.L1012
.L964:
	leaq	(%rsi,%r9), %r14
	movq	%r13, %rdx
	addq	%rax, %r9
	subq	%r9, %rdx
	cmpq	%rsi, %rcx
	jnb	.L1013
.L968:
	testq	%rdx, %rdx
	je	.L970
	cmpq	%rbp, %rax
	je	.L970
	leaq	(%r14,%rax), %rsi
	leaq	(%r14,%rbp), %rdi
	cmpq	$1, %rdx
	je	.L1014
	movq	%rcx, (%rsp)
	call	memmove@PLT
	movq	(%rsp), %rcx
.L970:
	testq	%rbp, %rbp
	je	.L1009
.L972:
	cmpq	$1, %rbp
	je	.L1015
	movq	%rbp, %rdx
	movq	%rcx, %rsi
	movq	%r14, %rdi
	call	memcpy@PLT
	movq	(%rbx), %r12
	jmp	.L973
	.p2align 4,,10
	.p2align 3
.L1015:
	movzbl	(%rcx), %ebp
	movb	%bpl, (%r14)
.L1009:
	movq	(%rbx), %r12
.L973:
	movq	%r15, 8(%rbx)
	movb	$0, (%r12,%r15)
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L1013:
	.cfi_restore_state
	addq	%r13, %rsi
	cmpq	%rcx, %rsi
	jnb	.L969
	jmp	.L968
	.p2align 4,,10
	.p2align 3
.L1012:
	testq	%r15, %r15
	js	.L965
	addq	%r9, %rax
	addq	%r14, %r14
	movq	%rax, 24(%rsp)
	subq	%rax, %r13
	cmpq	%r14, %r15
	jb	.L1016
.L976:
	movq	%r15, %rdi
	movq	%r15, %r14
	addq	$1, %rdi
	js	.L977
.L978:
	movq	%rcx, 16(%rsp)
	movq	%r9, 8(%rsp)
	movq	%r8, (%rsp)
	call	_Znwm@PLT
	movq	8(%rsp), %r10
	movq	(%rsp), %r9
	movq	16(%rsp), %rcx
	movq	%rax, %r12
	testq	%r10, %r10
	je	.L979
	movq	(%rbx), %rsi
	cmpq	$1, %r10
	je	.L1017
	movq	%r10, %rdx
	movq	%rax, %rdi
	movq	%rcx, 16(%rsp)
	movq	%r9, 8(%rsp)
	movq	%r10, (%rsp)
	call	memcpy@PLT
	movq	16(%rsp), %rcx
	movq	8(%rsp), %r9
	movq	(%rsp), %r10
.L979:
	testq	%rcx, %rcx
	je	.L981
	testq	%rbp, %rbp
	je	.L981
	leaq	(%r12,%r10), %rdi
	cmpq	$1, %rbp
	je	.L1018
	movq	%rbp, %rdx
	movq	%rcx, %rsi
	movq	%r10, 8(%rsp)
	movq	%r9, (%rsp)
	call	memcpy@PLT
	movq	8(%rsp), %r10
	movq	(%rsp), %r9
.L981:
	movq	(%rbx), %rdx
	testq	%r13, %r13
	jne	.L1019
.L983:
	cmpq	%rdx, %r9
	je	.L985
	movq	16(%rbx), %r13
	movq	%rdx, %rdi
	leaq	1(%r13), %rsi
	call	_ZdlPvm@PLT
.L985:
	movq	%r12, (%rbx)
	movq	%r14, 16(%rbx)
	jmp	.L973
	.p2align 4,,10
	.p2align 3
.L1019:
	movq	24(%rsp), %rsi
	addq	%r10, %rbp
	leaq	(%r12,%rbp), %rdi
	addq	%rdx, %rsi
	cmpq	$1, %r13
	je	.L1020
	movq	%rdx, 8(%rsp)
	movq	%r13, %rdx
	movq	%r9, (%rsp)
	call	memcpy@PLT
	movq	8(%rsp), %rdx
	movq	(%rsp), %r9
	jmp	.L983
	.p2align 4,,10
	.p2align 3
.L1016:
	leaq	1(%r14), %rdi
	testq	%r14, %r14
	jns	.L978
.L977:
	call	_ZSt17__throw_bad_allocv@PLT
	.p2align 4,,10
	.p2align 3
.L1011:
	cmpq	$15, %r15
	jbe	.L964
	testq	%r15, %r15
	js	.L965
	addq	%r9, %rax
	movq	%rax, 24(%rsp)
	subq	%rax, %r13
	cmpq	$29, %r15
	ja	.L976
	movl	$31, %edi
	movl	$30, %r14d
	jmp	.L978
	.p2align 4,,10
	.p2align 3
.L1014:
	movzbl	(%rsi), %esi
	movb	%sil, (%rdi)
	testq	%rbp, %rbp
	jne	.L972
	jmp	.L1009
	.p2align 4,,10
	.p2align 3
.L1020:
	movzbl	(%rsi), %eax
	movb	%al, (%rdi)
	jmp	.L983
	.p2align 4,,10
	.p2align 3
.L1018:
	movzbl	(%rcx), %r11d
	movb	%r11b, (%rdi)
	jmp	.L981
	.p2align 4,,10
	.p2align 3
.L1017:
	movzbl	(%rsi), %edi
	movb	%dil, (%rax)
	jmp	.L979
.L1010:
	leaq	.LC13(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L965:
	leaq	.LC12(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0.cold, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0.cold:
.LFSB14043:
.L969:
	.cfi_def_cfa_offset 96
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
	movq	%rdx, %r9
	movq	%rbp, %r8
	movq	%rax, %rdx
	movq	%r14, %rsi
	movq	%rbx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_replace_coldEPcmPKcmm@PLT
	movq	(%rbx), %r12
	jmp	.L973
	.cfi_endproc
.LFE14043:
	.text
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	.section	.text.unlikely
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0.cold, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0.cold
.LCOLDE14:
	.text
.LHOTE14:
	.section	.text.unlikely._ZSt20__throw_format_errorPKc,"axG",@progbits,_ZSt20__throw_format_errorPKc,comdat
	.weak	_ZSt20__throw_format_errorPKc
	.type	_ZSt20__throw_format_errorPKc, @function
_ZSt20__throw_format_errorPKc:
.LFB11376:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA11376
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %rbp
	movl	$16, %edi
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	pushq	%rax
	.cfi_def_cfa_offset 32
	call	__cxa_allocate_exception@PLT
	movq	%rbp, %rsi
	movq	%rax, %rdi
	movq	%rax, %rbx
.LEHB0:
	call	_ZNSt13runtime_errorC2EPKc@PLT
.LEHE0:
	leaq	16+_ZTVSt12format_error(%rip), %rax
	leaq	_ZNSt12format_errorD1Ev(%rip), %rdx
	movq	%rbx, %rdi
	leaq	_ZTISt12format_error(%rip), %rsi
	movq	%rax, (%rbx)
.LEHB1:
	call	__cxa_throw@PLT
.L1023:
	endbr64
	movq	%rax, %rbp
.L1022:
	movq	%rbx, %rdi
	vzeroupper
	call	__cxa_free_exception@PLT
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE1:
	.cfi_endproc
.LFE11376:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table._ZSt20__throw_format_errorPKc,"aG",@progbits,_ZSt20__throw_format_errorPKc,comdat
.LLSDA11376:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11376-.LLSDACSB11376
.LLSDACSB11376:
	.uleb128 .LEHB0-.LFB11376
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L1023-.LFB11376
	.uleb128 0
	.uleb128 .LEHB1-.LFB11376
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE11376:
	.section	.text.unlikely._ZSt20__throw_format_errorPKc,"axG",@progbits,_ZSt20__throw_format_errorPKc,comdat
	.size	_ZSt20__throw_format_errorPKc, .-_ZSt20__throw_format_errorPKc
	.section	.rodata._ZNSt8__format39__unmatched_left_brace_in_format_stringEv.str1.8,"aMS",@progbits,1
	.align 8
.LC15:
	.string	"format error: unmatched '{' in format string"
	.section	.text.unlikely._ZNSt8__format39__unmatched_left_brace_in_format_stringEv,"axG",@progbits,_ZNSt8__format39__unmatched_left_brace_in_format_stringEv,comdat
	.weak	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
	.type	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv, @function
_ZNSt8__format39__unmatched_left_brace_in_format_stringEv:
.LFB11381:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC15(%rip), %rdi
	pushq	%rax
	.cfi_def_cfa_offset 16
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE11381:
	.size	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv, .-_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
	.section	.rodata._ZNSt8__format39__conflicting_indexing_in_format_stringEv.str1.8,"aMS",@progbits,1
	.align 8
.LC16:
	.string	"format error: conflicting indexing style in format string"
	.section	.text.unlikely._ZNSt8__format39__conflicting_indexing_in_format_stringEv,"axG",@progbits,_ZNSt8__format39__conflicting_indexing_in_format_stringEv,comdat
	.weak	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
	.type	_ZNSt8__format39__conflicting_indexing_in_format_stringEv, @function
_ZNSt8__format39__conflicting_indexing_in_format_stringEv:
.LFB11383:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC16(%rip), %rdi
	pushq	%rax
	.cfi_def_cfa_offset 16
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE11383:
	.size	_ZNSt8__format39__conflicting_indexing_in_format_stringEv, .-_ZNSt8__format39__conflicting_indexing_in_format_stringEv
	.section	.rodata._ZNSt8__format33__invalid_arg_id_in_format_stringEv.str1.8,"aMS",@progbits,1
	.align 8
.LC17:
	.string	"format error: invalid arg-id in format string"
	.section	.text.unlikely._ZNSt8__format33__invalid_arg_id_in_format_stringEv,"axG",@progbits,_ZNSt8__format33__invalid_arg_id_in_format_stringEv,comdat
	.weak	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.type	_ZNSt8__format33__invalid_arg_id_in_format_stringEv, @function
_ZNSt8__format33__invalid_arg_id_in_format_stringEv:
.LFB11384:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC17(%rip), %rdi
	pushq	%rax
	.cfi_def_cfa_offset 16
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE11384:
	.size	_ZNSt8__format33__invalid_arg_id_in_format_stringEv, .-_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.section	.rodata.str1.8
	.align 8
.LC18:
	.string	"format error: argument used for width or precision must be a non-negative integer"
	.section	.text.unlikely
	.align 2
.LCOLDB19:
	.text
.LHOTB19:
	.align 2
	.p2align 4
	.type	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0, @function
_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0:
.LFB14044:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movzwl	%di, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movzbl	(%rsi), %eax
	movl	%eax, %edx
	andl	$15, %eax
	andl	$15, %edx
	cmpq	%rax, %rdi
	jb	.L1046
	testb	%dl, %dl
	jne	.L1034
	movq	(%rsi), %rcx
	shrq	$4, %rcx
	cmpq	%rcx, %rdi
	jnb	.L1034
	salq	$5, %rdi
	addq	8(%rsi), %rdi
	vmovdqu	(%rdi), %xmm2
	movzbl	16(%rdi), %r8d
	vmovdqa	%xmm2, 32(%rsp)
	jmp	.L1033
	.p2align 4,,10
	.p2align 3
.L1046:
	movq	(%rsi), %r8
	leaq	(%rdi,%rdi,4), %rcx
	salq	$4, %rdi
	addq	8(%rsi), %rdi
	vmovdqa	(%rdi), %xmm1
	shrq	$4, %r8
	shrq	%cl, %r8
	vmovdqa	%xmm1, 32(%rsp)
	andl	$31, %r8d
.L1033:
	leaq	.L1037(%rip), %r9
	movzbl	%r8b, %esi
	movb	%r8b, 48(%rsp)
	vmovdqu	32(%rsp), %ymm0
	movslq	(%r9,%rsi,4), %r10
	vmovdqu	%ymm0, (%rsp)
	addq	%r9, %r10
	notrack jmp	*%r10
	.section	.rodata
	.align 4
	.align 4
.L1037:
	.long	.L1044-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1041-.L1037
	.long	.L1040-.L1037
	.long	.L1039-.L1037
	.long	.L1038-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.long	.L1036-.L1037
	.text
	.p2align 4,,10
	.p2align 3
.L1039:
	movq	(%rsp), %rax
	testq	%rax, %rax
	js	.L1042
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1040:
	.cfi_restore_state
	movl	(%rsp), %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1041:
	.cfi_restore_state
	movslq	(%rsp), %rax
	testl	%eax, %eax
	js	.L1042
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1038:
	.cfi_restore_state
	movq	(%rsp), %rax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1044:
	.cfi_restore_state
	vzeroupper
	jmp	.L1034
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold, @function
_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold:
.LFSB14044:
.L1036:
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L1042:
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L1034:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.cfi_endproc
.LFE14044:
	.text
	.size	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0, .-_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
	.section	.text.unlikely
	.size	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold, .-_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold
.LCOLDE19:
	.text
.LHOTE19:
	.section	.text.unlikely
	.align 2
.LCOLDB20:
	.text
.LHOTB20:
	.align 2
	.p2align 4
	.type	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0, @function
_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0:
.LFB14047:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movzwl	%di, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movzbl	(%rsi), %eax
	movl	%eax, %edx
	andl	$15, %eax
	andl	$15, %edx
	cmpq	%rax, %rdi
	jb	.L1062
	testb	%dl, %dl
	jne	.L1050
	movq	(%rsi), %rcx
	shrq	$4, %rcx
	cmpq	%rcx, %rdi
	jnb	.L1050
	salq	$5, %rdi
	addq	8(%rsi), %rdi
	vmovdqu	(%rdi), %xmm2
	movzbl	16(%rdi), %r8d
	vmovdqa	%xmm2, 32(%rsp)
	jmp	.L1049
	.p2align 4,,10
	.p2align 3
.L1062:
	movq	(%rsi), %r8
	leaq	(%rdi,%rdi,4), %rcx
	salq	$4, %rdi
	addq	8(%rsi), %rdi
	vmovdqa	(%rdi), %xmm1
	shrq	$4, %r8
	shrq	%cl, %r8
	vmovdqa	%xmm1, 32(%rsp)
	andl	$31, %r8d
.L1049:
	leaq	.L1053(%rip), %r9
	movzbl	%r8b, %esi
	movb	%r8b, 48(%rsp)
	vmovdqu	32(%rsp), %ymm0
	movslq	(%r9,%rsi,4), %r10
	vmovdqu	%ymm0, (%rsp)
	addq	%r9, %r10
	notrack jmp	*%r10
	.section	.rodata
	.align 4
	.align 4
.L1053:
	.long	.L1060-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1057-.L1053
	.long	.L1056-.L1053
	.long	.L1055-.L1053
	.long	.L1054-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.long	.L1052-.L1053
	.text
	.p2align 4,,10
	.p2align 3
.L1055:
	movq	(%rsp), %rax
	testq	%rax, %rax
	js	.L1058
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1056:
	.cfi_restore_state
	movl	(%rsp), %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1057:
	.cfi_restore_state
	movslq	(%rsp), %rax
	testl	%eax, %eax
	js	.L1058
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1054:
	.cfi_restore_state
	movq	(%rsp), %rax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L1060:
	.cfi_restore_state
	vzeroupper
	jmp	.L1050
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold, @function
_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold:
.LFSB14047:
.L1052:
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L1058:
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L1050:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.cfi_endproc
.LFE14047:
	.text
	.size	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0, .-_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
	.section	.text.unlikely
	.size	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold, .-_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0.cold
.LCOLDE20:
	.text
.LHOTE20:
	.section	.rodata._ZNSt8__format29__failed_to_parse_format_specEv.str1.8,"aMS",@progbits,1
	.align 8
.LC21:
	.string	"format error: failed to parse format-spec"
	.section	.text.unlikely._ZNSt8__format29__failed_to_parse_format_specEv,"axG",@progbits,_ZNSt8__format29__failed_to_parse_format_specEv,comdat
	.weak	_ZNSt8__format29__failed_to_parse_format_specEv
	.type	_ZNSt8__format29__failed_to_parse_format_specEv, @function
_ZNSt8__format29__failed_to_parse_format_specEv:
.LFB11385:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC21(%rip), %rdi
	pushq	%rax
	.cfi_def_cfa_offset 16
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE11385:
	.size	_ZNSt8__format29__failed_to_parse_format_specEv, .-_ZNSt8__format29__failed_to_parse_format_specEv
	.text
	.p2align 4
	.globl	_Z9benchmarkPFvPKfPfiES0_S1_iii
	.type	_Z9benchmarkPFvPKfPfiES0_S1_iii, @function
_Z9benchmarkPFvPKfPfiES0_S1_iii:
.LFB11757:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movl	%r8d, %r14d
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movl	%ecx, %r13d
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdx, %rbx
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	testl	%r9d, %r9d
	jle	.L1066
	movl	%r9d, %edi
	call	omp_set_num_threads@PLT
.L1066:
	movslq	%r14d, %r15
	movl	%r14d, %eax
	imulq	$1717986919, %r15, %rdx
	sarl	$31, %eax
	sarq	$34, %rdx
	subl	%eax, %edx
	movl	%edx, (%rsp)
	cmpl	$9, %r14d
	jle	.L1067
	leal	-1(%rdx), %esi
	movq	%r12, %rdi
	movl	%r13d, %edx
	movl	$1, %r15d
	andl	$7, %esi
	movl	%esi, 12(%rsp)
	movq	%rbx, %rsi
	call	*%rbp
	movl	(%rsp), %edi
	movl	12(%rsp), %r8d
	cmpl	%edi, %r15d
	jge	.L1145
	testl	%r8d, %r8d
	je	.L1068
	cmpl	$1, %r8d
	je	.L1125
	cmpl	$2, %r8d
	je	.L1126
	cmpl	$3, %r8d
	je	.L1127
	cmpl	$4, %r8d
	je	.L1128
	cmpl	$5, %r8d
	je	.L1129
	cmpl	$6, %r8d
	je	.L1130
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	movl	$2, %r15d
	call	*%rbp
.L1130:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L1129:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L1128:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L1127:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L1126:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
.L1125:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	movl	(%rsp), %r9d
	cmpl	%r9d, %r15d
	jge	.L1145
.L1068:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$8, %r15d
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	*%rbp
	movl	(%rsp), %r10d
	cmpl	%r10d, %r15d
	jl	.L1068
.L1145:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movl	$0x00000000, 28(%rsp)
	movq	%rax, (%rsp)
.L1072:
	movl	%r14d, %r11d
	xorl	%r15d, %r15d
	andl	$7, %r11d
	je	.L1071
	cmpl	$1, %r11d
	je	.L1119
	cmpl	$2, %r11d
	je	.L1120
	cmpl	$3, %r11d
	je	.L1121
	cmpl	$4, %r11d
	je	.L1122
	cmpl	$5, %r11d
	je	.L1123
	cmpl	$6, %r11d
	jne	.L1149
.L1124:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm2
	vaddss	(%rbx), %xmm2, %xmm3
	vmovss	%xmm3, 28(%rsp)
.L1123:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm4
	vaddss	(%rbx), %xmm4, %xmm5
	vmovss	%xmm5, 28(%rsp)
.L1122:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm6
	vaddss	(%rbx), %xmm6, %xmm7
	vmovss	%xmm7, 28(%rsp)
.L1121:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm8
	vaddss	(%rbx), %xmm8, %xmm9
	vmovss	%xmm9, 28(%rsp)
.L1120:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm10
	vaddss	(%rbx), %xmm10, %xmm11
	vmovss	%xmm11, 28(%rsp)
.L1119:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm12
	vaddss	(%rbx), %xmm12, %xmm13
	vmovss	%xmm13, 28(%rsp)
	cmpl	%r15d, %r14d
	je	.L1070
.L1071:
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	addl	$8, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm14
	vaddss	(%rbx), %xmm14, %xmm15
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm15, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm0
	vaddss	(%rbx), %xmm0, %xmm1
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm1, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm2
	vaddss	(%rbx), %xmm2, %xmm3
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm3, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm4
	vaddss	(%rbx), %xmm4, %xmm5
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm5, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm6
	vaddss	(%rbx), %xmm6, %xmm7
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm7, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm8
	vaddss	(%rbx), %xmm8, %xmm9
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm9, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm10
	vaddss	(%rbx), %xmm10, %xmm11
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vmovss	%xmm11, 28(%rsp)
	call	*%rbp
	vmovss	28(%rsp), %xmm12
	vaddss	(%rbx), %xmm12, %xmm13
	vmovss	%xmm13, 28(%rsp)
	cmpl	%r15d, %r14d
	jne	.L1071
.L1070:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	(%rsp), %rdx
	vxorpd	%xmm14, %xmm14, %xmm14
	subq	%rdx, %rax
	vcvtsi2sdq	%rax, %xmm14, %xmm15
	vmulsd	.LC8(%rip), %xmm15, %xmm0
	vmovsd	%xmm0, (%rsp)
	call	omp_get_max_threads@PLT
	movl	%eax, %edi
	call	omp_set_num_threads@PLT
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	(%rsp), %xmm3
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	vcvtsi2sdl	%r14d, %xmm1, %xmm2
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	vdivsd	%xmm2, %xmm3, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L1149:
	.cfi_restore_state
	movl	%r13d, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	movl	$1, %r15d
	call	*%rbp
	vmovss	28(%rsp), %xmm0
	vaddss	(%rbx), %xmm0, %xmm1
	vmovss	%xmm1, 28(%rsp)
	jmp	.L1124
	.p2align 4,,10
	.p2align 3
.L1067:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movl	$0x00000000, 28(%rsp)
	movq	%rax, (%rsp)
	testl	%r14d, %r14d
	jle	.L1070
	jmp	.L1072
	.cfi_endproc
.LFE11757:
	.size	_Z9benchmarkPFvPKfPfiES0_S1_iii, .-_Z9benchmarkPFvPKfPfiES0_S1_iii
	.section	.text._ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_,"axG",@progbits,_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_
	.type	_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_, @function
_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_:
.LFB11822:
	.cfi_startproc
	endbr64
	movzbl	(%rsi), %ecx
	movq	%rdi, %rax
	movq	%rsi, %r10
	cmpb	$123, %cl
	je	.L1150
	subq	%rsi, %rdx
	cmpq	$1, %rdx
	jle	.L1152
	movzbl	1(%rsi), %edx
	cmpb	$62, %dl
	je	.L1156
	cmpb	$94, %dl
	je	.L1157
	movl	$1, %edi
	cmpb	$60, %dl
	jne	.L1152
.L1153:
	movzbl	(%rax), %r11d
	movb	%cl, 6(%rax)
	leaq	2(%rsi), %r10
	andl	$-4, %r11d
	orl	%edi, %r11d
	movb	%r11b, (%rax)
.L1150:
	movq	%r10, %rax
	ret
	.p2align 4,,10
	.p2align 3
.L1152:
	cmpb	$62, %cl
	je	.L1158
	cmpb	$94, %cl
	je	.L1159
	movq	%rsi, %r10
	cmpb	$60, %cl
	jne	.L1150
	movl	$1, %r8d
.L1154:
	movzbl	(%rax), %r9d
	leaq	1(%rsi), %r10
	movb	$32, 6(%rax)
	andl	$-4, %r9d
	orl	%r8d, %r9d
	movb	%r9b, (%rax)
	movq	%r10, %rax
	ret
	.p2align 4,,10
	.p2align 3
.L1157:
	movl	$3, %edi
	jmp	.L1153
	.p2align 4,,10
	.p2align 3
.L1159:
	movl	$3, %r8d
	jmp	.L1154
	.p2align 4,,10
	.p2align 3
.L1156:
	movl	$2, %edi
	jmp	.L1153
	.p2align 4,,10
	.p2align 3
.L1158:
	movl	$2, %r8d
	jmp	.L1154
	.cfi_endproc
.LFE11822:
	.size	_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_, .-_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_
	.section	.rodata._ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE.str1.8,"aMS",@progbits,1
	.align 8
.LC22:
	.string	"format error: width must be non-zero in format string"
	.align 8
.LC23:
	.string	"format error: invalid width or precision in format-spec"
	.section	.text._ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE,"axG",@progbits,_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE
	.type	_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE, @function
_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE:
.LFB11824:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %r11
	movq	%rsi, %r9
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movzbl	(%rsi), %edx
	cmpb	$48, %dl
	je	.L1266
	movq	%rcx, %rsi
	movzbl	%dl, %eax
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %rcx
	movq	%rdi, %r10
	cmpb	$9, (%rcx,%rax)
	movq	%r9, %rcx
	ja	.L1164
	movq	%r11, %rbp
	xorl	%eax, %eax
	movl	$16, %esi
	subq	%r9, %rbp
	andl	$3, %ebp
	je	.L1173
	cmpq	$1, %rbp
	je	.L1230
	cmpq	$2, %rbp
	jne	.L1267
.L1231:
	movzbl	(%rcx), %edx
	leal	-48(%rdx), %r8d
	cmpb	$9, %r8b
	ja	.L1165
	subl	$4, %esi
	js	.L1268
	leal	(%rax,%rax,4), %eax
	movzbl	%r8b, %ebp
	leal	0(%rbp,%rax,2), %eax
.L1245:
	addq	$1, %rcx
.L1230:
	movzbl	(%rcx), %edx
	leal	-48(%rdx), %r8d
	cmpb	$9, %r8b
	ja	.L1165
	subl	$4, %esi
	js	.L1269
	leal	(%rax,%rax,4), %eax
	movzbl	%r8b, %ebp
	leal	0(%rbp,%rax,2), %eax
.L1247:
	addq	$1, %rcx
	cmpq	%rcx, %r11
	je	.L1175
.L1173:
	movzbl	(%rcx), %edx
	leal	-48(%rdx), %r8d
	cmpb	$9, %r8b
	ja	.L1165
	subl	$4, %esi
	js	.L1166
	leal	(%rax,%rax,4), %eax
	movzbl	%r8b, %ebp
	leal	0(%rbp,%rax,2), %eax
.L1167:
	movzbl	1(%rcx), %edx
	leaq	1(%rcx), %rdi
	movq	%rdi, %rcx
	leal	-48(%rdx), %r8d
	cmpb	$9, %r8b
	ja	.L1165
	movl	%esi, %ecx
	subl	$4, %ecx
	js	.L1270
	leal	(%rax,%rax,4), %eax
	movzbl	%r8b, %edx
	leal	(%rdx,%rax,2), %eax
.L1250:
	movzbl	1(%rdi), %r8d
	leaq	1(%rdi), %rcx
	leal	-48(%r8), %ebp
	cmpb	$9, %bpl
	ja	.L1165
	movl	%esi, %ecx
	subl	$8, %ecx
	js	.L1271
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %r8d
	leal	(%r8,%rax,2), %eax
.L1252:
	movzbl	2(%rdi), %ebp
	leaq	2(%rdi), %rcx
	leal	-48(%rbp), %ebx
	cmpb	$9, %bl
	ja	.L1165
	subl	$12, %esi
	js	.L1272
	leal	(%rax,%rax,4), %eax
	movzbl	%bl, %r8d
	leal	(%r8,%rax,2), %eax
.L1254:
	leaq	3(%rdi), %rcx
	cmpq	%rcx, %r11
	jne	.L1173
	.p2align 4,,10
	.p2align 3
.L1175:
	movw	%ax, 2(%r10)
	movl	$1, %r9d
.L1174:
	movzwl	(%r10), %r11d
	andl	$3, %r9d
	sall	$7, %r9d
	andw	$-385, %r11w
	orl	%r9d, %r11d
	movw	%r11w, (%r10)
.L1162:
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	movq	%rcx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L1164:
	.cfi_restore_state
	cmpb	$123, %dl
	jne	.L1162
	leaq	1(%r9), %rcx
	cmpq	%rcx, %r11
	je	.L1273
	movsbw	1(%r9), %ax
	cmpb	$125, %al
	je	.L1274
	cmpb	$48, %al
	je	.L1275
	leal	-49(%rax), %ebx
	cmpb	$8, %bl
	ja	.L1183
	leaq	2(%r9), %rdi
	cmpq	%rdi, %r11
	je	.L1183
	movzbl	2(%r9), %ebp
	leal	-48(%rbp), %r8d
	cmpb	$9, %r8b
	jbe	.L1197
	subl	$48, %eax
	movq	%rdi, %rcx
	.p2align 4,,10
	.p2align 3
.L1185:
	cmpb	$125, (%rcx)
	jne	.L1183
	cmpl	$2, 16(%rsi)
	je	.L1195
	movl	$1, 16(%rsi)
	movw	%ax, 2(%r10)
	jmp	.L1180
	.p2align 4,,10
	.p2align 3
.L1267:
	subl	$48, %edx
	cmpb	$9, %dl
	ja	.L1165
	movl	$12, %esi
	movzbl	%dl, %eax
	leaq	1(%r9), %rcx
	jmp	.L1231
	.p2align 4,,10
	.p2align 3
.L1165:
	cmpq	%rcx, %r9
	jne	.L1175
.L1170:
	leaq	.LC23(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.p2align 4,,10
	.p2align 3
.L1166:
	movl	$10, %edi
	mulw	%di
	jo	.L1170
	movzbl	%r8b, %ebx
	addw	%ax, %bx
	jc	.L1170
	movl	%ebx, %eax
	jmp	.L1167
	.p2align 4,,10
	.p2align 3
.L1270:
	movl	$10, %ebx
	mulw	%bx
	jo	.L1170
	movzbl	%r8b, %ebp
	addw	%ax, %bp
	jc	.L1170
	movl	%ebp, %eax
	jmp	.L1250
	.p2align 4,,10
	.p2align 3
.L1271:
	movl	$10, %ebx
	mulw	%bx
	jo	.L1170
	movzbl	%bpl, %edx
	addw	%ax, %dx
	jc	.L1170
	movl	%edx, %eax
	jmp	.L1252
	.p2align 4,,10
	.p2align 3
.L1272:
	movl	$10, %ecx
	mulw	%cx
	jo	.L1170
	movzbl	%bl, %edx
	addw	%ax, %dx
	jc	.L1170
	movl	%edx, %eax
	jmp	.L1254
	.p2align 4,,10
	.p2align 3
.L1274:
	cmpl	$1, 16(%rsi)
	je	.L1195
	movq	24(%rsi), %rdi
	movl	$2, 16(%rsi)
	leaq	1(%rdi), %rbx
	movq	%rbx, 24(%rsi)
	movw	%di, 2(%r10)
.L1180:
	addq	$1, %rcx
	cmpq	%rcx, %r9
	je	.L1162
	movl	$2, %r9d
	jmp	.L1174
.L1275:
	leaq	2(%r9), %rcx
	cmpq	%rcx, %r11
	sete	%r11b
	xorl	%eax, %eax
.L1182:
	testb	%r11b, %r11b
	je	.L1185
.L1183:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L1269:
	movl	$10, %edi
	mulw	%di
	jo	.L1170
	movzbl	%r8b, %ebx
	addw	%ax, %bx
	jc	.L1170
	movl	%ebx, %eax
	jmp	.L1247
.L1268:
	movl	$10, %edi
	mulw	%di
	jo	.L1170
	movzbl	%r8b, %ebx
	addw	%ax, %bx
	jc	.L1170
	movl	%ebx, %eax
	jmp	.L1245
.L1197:
	movq	%r11, %rdx
	movq	%rcx, %r8
	xorl	%eax, %eax
	movl	$16, %edi
	subq	%rcx, %rdx
	andl	$3, %edx
	je	.L1184
	cmpq	$1, %rdx
	je	.L1232
	cmpq	$2, %rdx
	je	.L1233
	movzbl	(%rcx), %ebx
	leal	-48(%rbx), %ebp
	cmpb	$9, %bpl
	ja	.L1186
	movl	$12, %edi
	movzbl	%bpl, %eax
	leaq	1(%rcx), %r8
.L1233:
	movzbl	(%r8), %ebx
	subl	$48, %ebx
	cmpb	$9, %bl
	ja	.L1186
	subl	$4, %edi
	js	.L1276
	leal	(%rax,%rax,4), %eax
	movzbl	%bl, %ebx
	leal	(%rbx,%rax,2), %eax
.L1256:
	addq	$1, %r8
.L1232:
	movzbl	(%r8), %ebp
	subl	$48, %ebp
	cmpb	$9, %bpl
	ja	.L1186
	subl	$4, %edi
	js	.L1277
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %ebp
	leal	0(%rbp,%rax,2), %eax
.L1258:
	addq	$1, %r8
	cmpq	%r8, %r11
	je	.L1193
.L1184:
	movzbl	(%r8), %ebx
	subl	$48, %ebx
	cmpb	$9, %bl
	ja	.L1186
	subl	$4, %edi
	js	.L1187
	leal	(%rax,%rax,4), %eax
	movzbl	%bl, %ebx
	leal	(%rbx,%rax,2), %eax
.L1188:
	movzbl	1(%r8), %edx
	leaq	1(%r8), %rbx
	movq	%rbx, %r8
	leal	-48(%rdx), %ebp
	cmpb	$9, %bpl
	ja	.L1186
	movl	%edi, %r8d
	subl	$4, %r8d
	js	.L1278
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %r8d
	leal	(%r8,%rax,2), %eax
.L1260:
	movzbl	1(%rbx), %edx
	leaq	1(%rbx), %r8
	leal	-48(%rdx), %ebp
	cmpb	$9, %bpl
	ja	.L1186
	movl	%edi, %r8d
	subl	$8, %r8d
	js	.L1279
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %r8d
	leal	(%r8,%rax,2), %eax
.L1262:
	movzbl	2(%rbx), %edx
	leaq	2(%rbx), %r8
	leal	-48(%rdx), %ebp
	cmpb	$9, %bpl
	ja	.L1186
	subl	$12, %edi
	js	.L1280
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %ebp
	leal	0(%rbp,%rax,2), %eax
.L1264:
	leaq	3(%rbx), %r8
	cmpq	%r8, %r11
	jne	.L1184
.L1193:
	cmpq	%r8, %r11
	movq	%r8, %rcx
	sete	%r11b
	jmp	.L1182
.L1186:
	cmpq	%r8, %rcx
	jne	.L1193
	jmp	.L1183
	.p2align 4,,10
	.p2align 3
.L1278:
	movl	$10, %edx
	mulw	%dx
	jo	.L1183
	movzbl	%bpl, %ebp
	addw	%ax, %bp
	jc	.L1183
	movl	%ebp, %eax
	jmp	.L1260
.L1187:
	movl	$10, %edx
	mulw	%dx
	jo	.L1183
	movzbl	%bl, %ebp
	addw	%ax, %bp
	jc	.L1183
	movl	%ebp, %eax
	jmp	.L1188
.L1280:
	movl	$10, %r8d
	mulw	%r8w
	jo	.L1183
	movzbl	%bpl, %edx
	addw	%ax, %dx
	jc	.L1183
	movl	%edx, %eax
	jmp	.L1264
.L1279:
	movl	$10, %edx
	mulw	%dx
	jo	.L1183
	movzbl	%bpl, %ebp
	addw	%ax, %bp
	jc	.L1183
	movl	%ebp, %eax
	jmp	.L1262
.L1277:
	movl	$10, %edx
	mulw	%dx
	jo	.L1183
	movzbl	%bpl, %ebx
	addw	%ax, %bx
	jc	.L1183
	movl	%ebx, %eax
	jmp	.L1258
.L1276:
	movl	$10, %edx
	mulw	%dx
	jo	.L1183
	movzbl	%bl, %ebp
	addw	%ax, %bp
	jc	.L1183
	movl	%ebp, %eax
	jmp	.L1256
.L1273:
	call	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
.L1195:
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
.L1266:
	leaq	.LC22(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE11824:
	.size	_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE, .-_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE
	.section	.text._ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_,"axG",@progbits,_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_,comdat
	.p2align 4
	.weak	_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_
	.type	_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_, @function
_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_:
.LFB12317:
	.cfi_startproc
	endbr64
	movsbw	(%rdi), %dx
	movq	%rsi, %r8
	movq	%rdi, %rcx
	leaq	1(%rdi), %rsi
	xorl	%eax, %eax
	cmpb	$48, %dl
	je	.L1283
	leal	-49(%rdx), %eax
	cmpb	$8, %al
	ja	.L1294
	leaq	1(%rdi), %rsi
	cmpq	%r8, %rsi
	je	.L1284
	movzbl	1(%rdi), %edi
	subl	$48, %edi
	cmpb	$9, %dil
	ja	.L1284
	movq	%r8, %r9
	movq	%rcx, %rsi
	xorl	%eax, %eax
	movl	$16, %r11d
	subq	%rcx, %r9
	andl	$3, %r9d
	je	.L1285
	cmpq	$1, %r9
	je	.L1314
	cmpq	$2, %r9
	je	.L1315
	subl	$48, %edx
	cmpb	$9, %dl
	ja	.L1286
	movl	$12, %r11d
	movzbl	%dl, %eax
	leaq	1(%rcx), %rsi
.L1315:
	movzbl	(%rsi), %r10d
	leal	-48(%r10), %edi
	cmpb	$9, %dil
	ja	.L1286
	subl	$4, %r11d
	js	.L1333
	leal	(%rax,%rax,4), %eax
	movzbl	%dil, %r10d
	leal	(%r10,%rax,2), %eax
.L1323:
	addq	$1, %rsi
.L1314:
	movzbl	(%rsi), %edi
	leal	-48(%rdi), %r9d
	cmpb	$9, %r9b
	ja	.L1286
	subl	$4, %r11d
	js	.L1334
	leal	(%rax,%rax,4), %eax
	movzbl	%r9b, %edi
	leal	(%rdi,%rax,2), %eax
.L1325:
	addq	$1, %rsi
	cmpq	%rsi, %r8
	je	.L1283
.L1285:
	movzbl	(%rsi), %r9d
	leal	-48(%r9), %r10d
	cmpb	$9, %r10b
	ja	.L1286
	subl	$4, %r11d
	js	.L1287
	leal	(%rax,%rax,4), %eax
	movzbl	%r10b, %r9d
	leal	(%r9,%rax,2), %eax
.L1288:
	movzbl	1(%rsi), %r10d
	leaq	1(%rsi), %rdi
	movq	%rdi, %rsi
	leal	-48(%r10), %r9d
	cmpb	$9, %r9b
	ja	.L1286
	movl	%r11d, %esi
	subl	$4, %esi
	js	.L1335
	leal	(%rax,%rax,4), %eax
	movzbl	%r9b, %r9d
	leal	(%r9,%rax,2), %eax
.L1327:
	movzbl	1(%rdi), %edx
	leaq	1(%rdi), %rsi
	leal	-48(%rdx), %r10d
	cmpb	$9, %r10b
	ja	.L1286
	movl	%r11d, %esi
	subl	$8, %esi
	js	.L1336
	leal	(%rax,%rax,4), %eax
	movzbl	%r10b, %r10d
	leal	(%r10,%rax,2), %eax
.L1329:
	movzbl	2(%rdi), %r9d
	leaq	2(%rdi), %rsi
	leal	-48(%r9), %r10d
	cmpb	$9, %r10b
	ja	.L1286
	subl	$12, %r11d
	js	.L1337
	leal	(%rax,%rax,4), %eax
	movzbl	%r10b, %r9d
	leal	(%r9,%rax,2), %eax
.L1331:
	leaq	3(%rdi), %rsi
	cmpq	%rsi, %r8
	jne	.L1285
.L1283:
	movq	%rsi, %rdx
	movzwl	%ax, %eax
	ret
	.p2align 4,,10
	.p2align 3
.L1294:
	xorl	%eax, %eax
	xorl	%esi, %esi
	movq	%rsi, %rdx
	movzwl	%ax, %eax
	ret
	.p2align 4,,10
	.p2align 3
.L1284:
	leal	-48(%rdx), %eax
	movq	%rsi, %rdx
	movzwl	%ax, %eax
	ret
.L1333:
	movl	$10, %edx
	mulw	%dx
	jo	.L1332
	movzbl	%dil, %r9d
	addw	%ax, %r9w
	jc	.L1332
	movl	%r9d, %eax
	jmp	.L1323
	.p2align 4,,10
	.p2align 3
.L1286:
	cmpq	%rsi, %rcx
	jne	.L1283
.L1332:
	xorl	%esi, %esi
	xorl	%eax, %eax
	jmp	.L1283
	.p2align 4,,10
	.p2align 3
.L1287:
	movl	$10, %edx
	mulw	%dx
	jo	.L1332
	movzbl	%r10b, %edi
	addw	%ax, %di
	jc	.L1332
	movl	%edi, %eax
	jmp	.L1288
	.p2align 4,,10
	.p2align 3
.L1335:
	movl	$10, %edx
	mulw	%dx
	jo	.L1332
	movzbl	%r9b, %r10d
	addw	%ax, %r10w
	jc	.L1332
	movl	%r10d, %eax
	jmp	.L1327
	.p2align 4,,10
	.p2align 3
.L1336:
	movl	$10, %r9d
	mulw	%r9w
	jo	.L1332
	movzbl	%r10b, %edx
	addw	%ax, %dx
	jc	.L1332
	movl	%edx, %eax
	jmp	.L1329
	.p2align 4,,10
	.p2align 3
.L1337:
	movl	$10, %esi
	mulw	%si
	jo	.L1332
	movzbl	%r10b, %edx
	addw	%ax, %dx
	jc	.L1332
	movl	%edx, %eax
	jmp	.L1331
.L1334:
	movl	$10, %edx
	mulw	%dx
	jo	.L1332
	movzbl	%r9b, %r10d
	addw	%ax, %r10w
	jc	.L1332
	movl	%r10d, %eax
	jmp	.L1325
	.cfi_endproc
.LFE12317:
	.size	_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_, .-_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_
	.section	.rodata._ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE.str1.8,"aMS",@progbits,1
	.align 8
.LC24:
	.string	"format error: missing precision after '.' in format string"
	.section	.text._ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE,"axG",@progbits,_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE
	.type	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE, @function
_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE:
.LFB11819:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rsi, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movq	%rdi, %r13
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	movq	(%rsi), %rcx
	movq	8(%rsi), %r12
	xorl	%esi, %esi
	cmpq	%rcx, %r12
	je	.L1406
	movzbl	(%rcx), %ebp
	cmpb	$125, %bpl
	je	.L1406
	cmpb	$123, %bpl
	je	.L1407
	movq	%r12, %rax
	subq	%rcx, %rax
	cmpq	$1, %rax
	jle	.L1341
	movzbl	1(%rcx), %edx
	cmpb	$62, %dl
	je	.L1408
	cmpb	$94, %dl
	je	.L1409
	movl	$1, %r15d
	cmpb	$60, %dl
	je	.L1342
	cmpb	$62, %bpl
	je	.L1417
	cmpb	$94, %bpl
	je	.L1418
	cmpb	$60, %bpl
	je	.L1419
	xorl	%r15d, %r15d
	movl	$32, %ebp
	jmp	.L1346
	.p2align 4,,10
	.p2align 3
.L1407:
	xorl	%r15d, %r15d
	movl	$32, %ebp
.L1340:
	leaq	1(%rcx), %rdi
	cmpq	%rdi, %r12
	je	.L1398
	movsbw	1(%rcx), %r9w
	cmpb	$125, %r9b
	je	.L1530
	cmpb	$48, %r9b
	je	.L1531
	leal	-49(%r9), %r10d
	cmpb	$8, %r10b
	jbe	.L1369
.L1370:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L1406:
	movl	$32, %ebp
.L1339:
	movzbl	%bpl, %edi
	movabsq	$-71776119061217281, %rbp
	salq	$48, %rdi
	andq	%rsi, %rbp
	orq	%rdi, %rbp
	movq	%rbp, %rax
	movl	%ebp, 0(%r13)
	shrq	$24, %rax
	movl	%eax, 3(%r13)
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%rcx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L1408:
	.cfi_restore_state
	movl	$2, %r15d
.L1342:
	addq	$2, %rcx
	cmpq	%rcx, %r12
	je	.L1347
.L1346:
	movzbl	(%rcx), %edi
	cmpb	$125, %dil
	je	.L1347
.L1348:
	cmpb	$48, %dil
	je	.L1532
	movzbl	%dil, %ebx
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %r8
	cmpb	$9, (%r8,%rbx)
	jbe	.L1533
	cmpb	$123, %dil
	je	.L1340
	xorl	%ebx, %ebx
	cmpq	%rcx, %r12
	je	.L1416
	xorl	%r8d, %r8d
.L1404:
	cmpb	$46, %dil
	je	.L1534
	xorl	%r9d, %r9d
	xorl	%esi, %esi
.L1383:
	cmpb	$115, %dil
	je	.L1535
.L1402:
	call	_ZNSt8__format29__failed_to_parse_format_specEv
	.p2align 4,,10
	.p2align 3
.L1417:
	movl	$2, %r15d
.L1345:
	addq	$1, %rcx
	movl	$32, %ebp
	cmpq	%rcx, %r12
	jne	.L1346
.L1347:
	movq	%r15, %rsi
	andl	$3, %esi
	jmp	.L1339
	.p2align 4,,10
	.p2align 3
.L1533:
	cmpq	%rcx, %r12
	je	.L1351
	movq	%r12, %rax
	movq	%rcx, %r8
	xorl	%ebx, %ebx
	movl	$16, %edi
	subq	%rcx, %rax
	andl	$3, %eax
	je	.L1359
	cmpq	$1, %rax
	je	.L1472
	cmpq	$2, %rax
	je	.L1473
	movzbl	(%rcx), %edx
	subl	$48, %edx
	cmpb	$9, %dl
	ja	.L1352
	movl	$12, %edi
	movzbl	%dl, %ebx
	leaq	1(%rcx), %r8
.L1473:
	movzbl	(%r8), %r10d
	leal	-48(%r10), %r9d
	cmpb	$9, %r9b
	ja	.L1352
	subl	$4, %edi
	js	.L1536
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r9b, %r10d
	leal	(%r10,%rdx,2), %ebx
.L1509:
	addq	$1, %r8
.L1472:
	movzbl	(%r8), %r9d
	leal	-48(%r9), %r11d
	cmpb	$9, %r11b
	ja	.L1352
	subl	$4, %edi
	js	.L1537
	leal	(%rbx,%rbx,4), %r10d
	movzbl	%r11b, %r9d
	leal	(%r9,%r10,2), %ebx
.L1511:
	addq	$1, %r8
	cmpq	%r8, %r12
	je	.L1512
.L1359:
	movzbl	(%r8), %r11d
	leal	-48(%r11), %r10d
	cmpb	$9, %r10b
	ja	.L1352
	subl	$4, %edi
	js	.L1353
	leal	(%rbx,%rbx,4), %r11d
	movzbl	%r10b, %r9d
	leal	(%r9,%r11,2), %ebx
.L1354:
	movzbl	1(%r8), %r10d
	leaq	1(%r8), %r11
	movq	%r11, %r8
	leal	-48(%r10), %r9d
	cmpb	$9, %r9b
	ja	.L1352
	movl	%edi, %r8d
	subl	$4, %r8d
	js	.L1538
	leal	(%rbx,%rbx,4), %r10d
	movzbl	%r9b, %r9d
	leal	(%r9,%r10,2), %ebx
.L1514:
	movzbl	1(%r11), %edx
	leaq	1(%r11), %r8
	leal	-48(%rdx), %r10d
	cmpb	$9, %r10b
	ja	.L1352
	movl	%edi, %r8d
	subl	$8, %r8d
	js	.L1539
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r10b, %r10d
	leal	(%r10,%rdx,2), %ebx
.L1516:
	movzbl	2(%r11), %r9d
	leaq	2(%r11), %r8
	leal	-48(%r9), %r10d
	cmpb	$9, %r10b
	ja	.L1352
	subl	$12, %edi
	js	.L1540
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r10b, %r9d
	leal	(%r9,%rdx,2), %ebx
.L1518:
	leaq	3(%r11), %r8
	cmpq	%r8, %r12
	jne	.L1359
.L1512:
	movq	%r12, %rcx
	movl	$1, %r8d
	jmp	.L1360
	.p2align 4,,10
	.p2align 3
.L1352:
	cmpq	%rcx, %r8
	je	.L1351
	movq	%r8, %rcx
	movl	$1, %r8d
.L1382:
	cmpq	%rcx, %r12
	je	.L1360
	movzbl	(%rcx), %edi
	cmpb	$125, %dil
	jne	.L1404
.L1360:
	sall	$7, %r8d
	movzwl	%bx, %ebx
	movabsq	$-4294901761, %r14
	movw	%r8w, %si
	salq	$16, %rbx
	orw	%r15w, %si
	andq	%rsi, %r14
	orq	%rbx, %r14
	movq	%r14, %rsi
	jmp	.L1339
	.p2align 4,,10
	.p2align 3
.L1341:
	cmpb	$62, %bpl
	je	.L1417
	cmpb	$94, %bpl
	je	.L1418
	cmpb	$60, %bpl
	je	.L1419
	movl	%ebp, %edi
	xorl	%r15d, %r15d
	movl	$32, %ebp
	jmp	.L1348
	.p2align 4,,10
	.p2align 3
.L1419:
	movl	$1, %r15d
	jmp	.L1345
	.p2align 4,,10
	.p2align 3
.L1534:
	leaq	1(%rcx), %r9
	cmpq	%r9, %r12
	je	.L1397
	movzbl	1(%rcx), %edi
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %rsi
	cmpb	$9, (%rsi,%rdi)
	ja	.L1386
	movq	%r12, %r14
	movq	%r9, %rcx
	xorl	%esi, %esi
	movl	$16, %r10d
	subq	%r9, %r14
	andl	$3, %r14d
	je	.L1394
	cmpq	$1, %r14
	je	.L1476
	cmpq	$2, %r14
	je	.L1477
	movzbl	(%r9), %eax
	subl	$48, %eax
	cmpb	$9, %al
	ja	.L1387
	movl	$12, %r10d
	movzbl	%al, %esi
	leaq	1(%r9), %rcx
.L1477:
	movzbl	(%rcx), %edx
	leal	-48(%rdx), %edi
	cmpb	$9, %dil
	ja	.L1387
	subl	$4, %r10d
	js	.L1541
	leal	(%rsi,%rsi,4), %r14d
	movzbl	%dil, %edx
	leal	(%rdx,%r14,2), %esi
.L1498:
	addq	$1, %rcx
.L1476:
	movzbl	(%rcx), %edi
	leal	-48(%rdi), %r11d
	cmpb	$9, %r11b
	ja	.L1387
	subl	$4, %r10d
	js	.L1542
	leal	(%rsi,%rsi,4), %edx
	movzbl	%r11b, %edi
	leal	(%rdi,%rdx,2), %esi
.L1500:
	addq	$1, %rcx
	cmpq	%rcx, %r12
	je	.L1501
.L1394:
	movzbl	(%rcx), %r11d
	leal	-48(%r11), %r14d
	cmpb	$9, %r14b
	ja	.L1387
	subl	$4, %r10d
	js	.L1388
	leal	(%rsi,%rsi,4), %r11d
	movzbl	%r14b, %edi
	leal	(%rdi,%r11,2), %esi
.L1389:
	movzbl	1(%rcx), %edx
	leaq	1(%rcx), %r14
	movq	%r14, %rcx
	leal	-48(%rdx), %r11d
	cmpb	$9, %r11b
	ja	.L1387
	movl	%r10d, %ecx
	subl	$4, %ecx
	js	.L1543
	leal	(%rsi,%rsi,4), %edx
	movzbl	%r11b, %r11d
	leal	(%r11,%rdx,2), %esi
.L1503:
	movzbl	1(%r14), %edi
	leaq	1(%r14), %rcx
	leal	-48(%rdi), %r11d
	cmpb	$9, %r11b
	ja	.L1387
	movl	%r10d, %ecx
	subl	$8, %ecx
	js	.L1544
	leal	(%rsi,%rsi,4), %edi
	movzbl	%r11b, %r11d
	leal	(%r11,%rdi,2), %esi
.L1505:
	movzbl	2(%r14), %edx
	leaq	2(%r14), %rcx
	leal	-48(%rdx), %edi
	cmpb	$9, %dil
	ja	.L1387
	subl	$12, %r10d
	js	.L1545
	leal	(%rsi,%rsi,4), %r11d
	movzbl	%dil, %edx
	leal	(%rdx,%r11,2), %esi
.L1507:
	leaq	3(%r14), %rcx
	cmpq	%rcx, %r12
	jne	.L1394
.L1501:
	movl	$1, %r9d
	.p2align 4,,10
	.p2align 3
.L1395:
	andl	$3, %r8d
	andl	$3, %r15d
	andl	$3, %r9d
	movzwl	%bx, %r12d
	salq	$7, %r8
	salq	$9, %r9
	movzwl	%si, %esi
	salq	$16, %r12
	orq	%r8, %r15
	salq	$32, %rsi
	orq	%r9, %r15
	orq	%r12, %r15
	orq	%r15, %rsi
	jmp	.L1339
	.p2align 4,,10
	.p2align 3
.L1409:
	movl	$3, %r15d
	jmp	.L1342
	.p2align 4,,10
	.p2align 3
.L1535:
	leaq	1(%rcx), %r10
	cmpq	%r12, %r10
	je	.L1403
	cmpb	$125, 1(%rcx)
	jne	.L1402
.L1403:
	andl	$3, %r8d
	andl	$3, %r15d
	andl	$3, %r9d
	movzwl	%si, %esi
	salq	$7, %r8
	salq	$9, %r9
	movq	%r10, %rcx
	orq	%r8, %r15
	movzwl	%bx, %r8d
	salq	$32, %rsi
	salq	$16, %r8
	orq	%r9, %r15
	orq	%r8, %r15
	orq	%r15, %rsi
	jmp	.L1339
	.p2align 4,,10
	.p2align 3
.L1418:
	movl	$3, %r15d
	jmp	.L1345
	.p2align 4,,10
	.p2align 3
.L1386:
	cmpb	$123, %dil
	je	.L1546
.L1397:
	leaq	.LC24(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.p2align 4,,10
	.p2align 3
.L1530:
	cmpl	$1, 16(%r14)
	je	.L1381
	movq	24(%r14), %rbx
	movl	$2, 16(%r14)
	leaq	1(%rbx), %r11
	movq	%r11, 24(%r14)
.L1366:
	addq	$1, %rdi
	xorl	%r8d, %r8d
	cmpq	%rcx, %rdi
	je	.L1382
	movq	%rdi, %rcx
	movl	$2, %r8d
	jmp	.L1382
	.p2align 4,,10
	.p2align 3
.L1353:
	movl	$10, %edx
	movl	%ebx, %eax
	mulw	%dx
	jo	.L1351
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1354
.L1351:
	leaq	.LC23(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.p2align 4,,10
	.p2align 3
.L1538:
	movl	$10, %edx
	movl	%ebx, %eax
	mulw	%dx
	jo	.L1351
	movzbl	%r9b, %ebx
	addw	%ax, %bx
	jnc	.L1514
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1539:
	movl	$10, %r9d
	movl	%ebx, %eax
	mulw	%r9w
	jo	.L1351
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1516
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1540:
	movl	$10, %r8d
	movl	%ebx, %eax
	mulw	%r8w
	jo	.L1351
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1518
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1416:
	xorl	%r8d, %r8d
	jmp	.L1360
.L1531:
	leaq	2(%rcx), %rdi
	cmpq	%rdi, %r12
	sete	%r8b
	xorl	%ebx, %ebx
.L1368:
	testb	%r8b, %r8b
	jne	.L1370
.L1372:
	cmpb	$125, (%rdi)
	jne	.L1370
	cmpl	$2, 16(%r14)
	je	.L1381
	movl	$1, 16(%r14)
	jmp	.L1366
.L1369:
	leaq	2(%rcx), %r11
	cmpq	%r11, %r12
	je	.L1370
	movzbl	2(%rcx), %eax
	subl	$48, %eax
	cmpb	$9, %al
	jbe	.L1413
	leal	-48(%r9), %ebx
	movq	%r11, %rdi
	jmp	.L1372
.L1537:
	movl	$10, %edx
	movl	%ebx, %eax
	mulw	%dx
	jo	.L1351
	movzbl	%r11b, %ebx
	addw	%ax, %bx
	jnc	.L1511
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1546:
	leaq	2(%rcx), %rdi
	cmpq	%rdi, %r12
	je	.L1398
	cmpb	$125, 2(%rcx)
	je	.L1547
	movq	%r12, %rsi
	movq	%r9, 16(%rsp)
	movb	%r8b, 15(%rsp)
	call	_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_
	movl	%eax, %esi
	movq	%rdx, %rdi
	testq	%rdx, %rdx
	je	.L1370
	cmpq	%rdx, %r12
	je	.L1370
	cmpb	$125, (%rdx)
	jne	.L1370
	cmpl	$2, 16(%r14)
	je	.L1381
	movl	$1, 16(%r14)
	movq	16(%rsp), %r9
	movzbl	15(%rsp), %r8d
.L1400:
	leaq	1(%rdi), %rcx
	cmpq	%rcx, %r9
	je	.L1397
	movl	$2, %r9d
.L1396:
	cmpq	%rcx, %r12
	je	.L1395
	movzbl	(%rcx), %edi
	cmpb	$125, %dil
	je	.L1395
	jmp	.L1383
.L1536:
	movl	$10, %r11d
	movl	%ebx, %eax
	mulw	%r11w
	jo	.L1351
	movzbl	%r9b, %ebx
	addw	%ax, %bx
	jnc	.L1509
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1387:
	cmpq	%rcx, %r9
	je	.L1351
	movl	$1, %r9d
	jmp	.L1396
.L1388:
	movl	$10, %edx
	movl	%esi, %eax
	mulw	%dx
	jo	.L1351
	movzbl	%r14b, %esi
	addw	%ax, %si
	jnc	.L1389
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1543:
	movl	$10, %edi
	movl	%esi, %eax
	mulw	%di
	jo	.L1351
	movzbl	%r11b, %esi
	addw	%ax, %si
	jnc	.L1503
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1544:
	movl	$10, %edx
	movl	%esi, %eax
	mulw	%dx
	jo	.L1351
	movzbl	%r11b, %esi
	addw	%ax, %si
	jnc	.L1505
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1413:
	movq	%r12, %rdx
	movq	%rdi, %r9
	xorl	%ebx, %ebx
	movl	$16, %r8d
	subq	%rdi, %rdx
	andl	$3, %edx
	je	.L1371
	cmpq	$1, %rdx
	je	.L1474
	cmpq	$2, %rdx
	je	.L1475
	movzbl	(%rdi), %r10d
	subl	$48, %r10d
	cmpb	$9, %r10b
	ja	.L1373
	movl	$12, %r8d
	movzbl	%r10b, %ebx
	leaq	1(%rdi), %r9
.L1475:
	movzbl	(%r9), %r11d
	leal	-48(%r11), %r10d
	cmpb	$9, %r10b
	ja	.L1373
	subl	$4, %r8d
	js	.L1548
	leal	(%rbx,%rbx,4), %r11d
	movzbl	%r10b, %r10d
	leal	(%r10,%r11,2), %ebx
.L1520:
	addq	$1, %r9
.L1474:
	movzbl	(%r9), %edx
	leal	-48(%rdx), %r11d
	cmpb	$9, %r11b
	ja	.L1373
	subl	$4, %r8d
	js	.L1549
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r11b, %r11d
	leal	(%r11,%rdx,2), %ebx
.L1522:
	addq	$1, %r9
	cmpq	%r9, %r12
	je	.L1380
.L1371:
	movzbl	(%r9), %r10d
	leal	-48(%r10), %r11d
	cmpb	$9, %r11b
	ja	.L1373
	subl	$4, %r8d
	js	.L1374
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r11b, %r10d
	leal	(%r10,%rdx,2), %ebx
.L1375:
	movzbl	1(%r9), %eax
	leaq	1(%r9), %r11
	movq	%r11, %r9
	leal	-48(%rax), %r10d
	cmpb	$9, %r10b
	ja	.L1373
	movl	%r8d, %r9d
	subl	$4, %r9d
	js	.L1550
	leal	(%rbx,%rbx,4), %r9d
	movzbl	%r10b, %r10d
	leal	(%r10,%r9,2), %ebx
.L1524:
	movzbl	1(%r11), %edx
	leaq	1(%r11), %r9
	leal	-48(%rdx), %r10d
	cmpb	$9, %r10b
	ja	.L1373
	movl	%r8d, %eax
	subl	$8, %eax
	js	.L1551
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r10b, %r10d
	leal	(%r10,%rdx,2), %ebx
.L1526:
	movzbl	2(%r11), %eax
	leaq	2(%r11), %r9
	leal	-48(%rax), %r10d
	cmpb	$9, %r10b
	ja	.L1373
	subl	$12, %r8d
	js	.L1552
	leal	(%rbx,%rbx,4), %edx
	movzbl	%r10b, %r10d
	leal	(%r10,%rdx,2), %ebx
.L1528:
	leaq	3(%r11), %r9
	cmpq	%r9, %r12
	jne	.L1371
.L1380:
	cmpq	%r9, %r12
	movq	%r9, %rdi
	sete	%r8b
	jmp	.L1368
.L1545:
	movl	$10, %ecx
	movl	%esi, %eax
	mulw	%cx
	jo	.L1351
	movzbl	%dil, %esi
	addw	%ax, %si
	jnc	.L1507
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1547:
	cmpl	$1, 16(%r14)
	je	.L1381
	movq	24(%r14), %rsi
	movl	$2, 16(%r14)
	leaq	1(%rsi), %rcx
	movq	%rcx, 24(%r14)
	jmp	.L1400
.L1542:
	movl	$10, %r14d
	movl	%esi, %eax
	mulw	%r14w
	jo	.L1351
	movzbl	%r11b, %esi
	addw	%ax, %si
	jnc	.L1500
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1541:
	movl	$10, %r11d
	movl	%esi, %eax
	mulw	%r11w
	jo	.L1351
	movzbl	%dil, %esi
	addw	%ax, %si
	jnc	.L1498
	jmp	.L1351
	.p2align 4,,10
	.p2align 3
.L1373:
	cmpq	%r9, %rdi
	jne	.L1380
	jmp	.L1370
.L1550:
	movl	$10, %edx
	movl	%ebx, %eax
	mulw	%dx
	jo	.L1370
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1524
	jmp	.L1370
	.p2align 4,,10
	.p2align 3
.L1374:
	movl	$10, %edx
	movl	%ebx, %eax
	mulw	%dx
	jo	.L1370
	movzbl	%r11b, %ebx
	addw	%ax, %bx
	jnc	.L1375
	jmp	.L1370
.L1552:
	movl	$10, %r9d
	movl	%ebx, %eax
	mulw	%r9w
	jo	.L1370
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1528
	jmp	.L1370
.L1551:
	movl	$10, %r9d
	movl	%ebx, %eax
	mulw	%r9w
	jo	.L1370
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1526
	jmp	.L1370
.L1549:
	movl	$10, %r10d
	movl	%ebx, %eax
	mulw	%r10w
	jo	.L1370
	movzbl	%r11b, %ebx
	addw	%ax, %bx
	jnc	.L1522
	jmp	.L1370
.L1548:
	movl	$10, %edx
	movl	%ebx, %eax
	mulw	%dx
	jo	.L1370
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jnc	.L1520
	jmp	.L1370
.L1381:
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
.L1398:
	call	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
.L1532:
	leaq	.LC22(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE11819:
	.size	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE, .-_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE
	.section	.text._ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,"axG",@progbits,_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	.type	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE, @function
_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE:
.LFB12357:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rsi, %r8
	movl	%edx, %r11d
	movq	%rdi, %r9
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	andl	$15, %r11d
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	movq	8(%rsi), %rbx
	movq	(%rsi), %rsi
	cmpq	%rbx, %rsi
	je	.L1554
	movzbl	(%rsi), %ecx
	cmpb	$125, %cl
	je	.L1554
	movl	%edx, %r10d
	xorl	%edi, %edi
	cmpb	$123, %cl
	je	.L1631
	movq	%rbx, %rax
	subq	%rsi, %rax
	cmpq	$1, %rax
	jle	.L1556
	movzbl	1(%rsi), %r12d
	cmpb	$62, %r12b
	je	.L1632
	cmpb	$94, %r12b
	je	.L1633
	movl	$1, %ebp
	cmpb	$60, %r12b
	je	.L1557
	cmpb	$62, %cl
	je	.L1649
	cmpb	$94, %cl
	je	.L1650
	cmpb	$60, %cl
	je	.L1651
	xorl	%ebp, %ebp
	movl	$32, %ecx
	jmp	.L1561
	.p2align 4,,10
	.p2align 3
.L1631:
	xorl	%r15d, %r15d
	movl	$32, %ecx
	xorl	%ebp, %ebp
.L1555:
	movzbl	(%rsi), %edx
	cmpb	$125, %dl
	jne	.L1627
.L1566:
	andl	$15, %r10d
	movzbl	%r15b, %ebx
	sall	$11, %r10d
	sall	$2, %ebx
	orl	%ebx, %ebp
	movw	%r10w, %di
	orw	%bp, %di
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1554:
	movq	%r11, %rdi
	movl	$32, %ecx
	andl	$15, %edi
	salq	$11, %rdi
.L1563:
	movzbl	%cl, %r10d
	movq	%rsi, %rax
	movabsq	$-71776119061217281, %rcx
	salq	$48, %r10
	andq	%rdi, %rcx
	orq	%r10, %rcx
	movq	%rcx, %r13
	movl	%ecx, (%r9)
	shrq	$24, %r13
	movl	%r13d, 3(%r9)
	addq	$56, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L1632:
	.cfi_restore_state
	movl	$2, %ebp
.L1557:
	addq	$2, %rsi
	cmpq	%rsi, %rbx
	je	.L1562
.L1561:
	movzbl	(%rsi), %edx
	cmpb	$125, %dl
	je	.L1562
.L1628:
	leal	-32(%rdx), %r13d
	xorl	%r15d, %r15d
	cmpb	$13, %r13b
	ja	.L1565
	leaq	CSWTCH.915(%rip), %r15
	movzbl	%r13b, %r14d
	movl	(%r15,%r14,4), %eax
	xorl	%r15d, %r15d
	testl	%eax, %eax
	jne	.L1759
.L1627:
	cmpb	$35, %dl
	jne	.L1565
	leaq	1(%rsi), %r12
	cmpq	%r12, %rbx
	je	.L1568
	movzbl	1(%rsi), %edx
	cmpb	$125, %dl
	je	.L1568
	movq	%r12, %rsi
	movl	$1, %r14d
	jmp	.L1625
	.p2align 4,,10
	.p2align 3
.L1568:
	movq	%rbp, %rdi
	andl	$3, %r15d
	andl	$15, %r11d
	movq	%r12, %rsi
	salq	$2, %r15
	andl	$3, %edi
	salq	$11, %r11
	orq	%r15, %rdi
	orq	$16, %rdi
	orq	%r11, %rdi
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1759:
	movl	%eax, %r15d
	addq	$1, %rsi
	andl	$3, %r15d
	cmpq	%rsi, %rbx
	je	.L1566
	jmp	.L1555
	.p2align 4,,10
	.p2align 3
.L1649:
	movl	$2, %ebp
.L1560:
	addq	$1, %rsi
	movl	$32, %ecx
	cmpq	%rsi, %rbx
	jne	.L1561
.L1562:
	movq	%rbp, %rdi
	andl	$15, %r11d
	andl	$3, %edi
	salq	$11, %r11
	orq	%r11, %rdi
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1565:
	xorl	%r14d, %r14d
.L1625:
	cmpb	$48, %dl
	je	.L1569
	movb	$0, 8(%rsp)
.L1570:
	movzbl	%dl, %eax
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %r12
	cmpb	$9, (%r12,%rax)
	ja	.L1573
	cmpq	%rsi, %rbx
	je	.L1574
	movq	%rbx, %r13
	movq	%rsi, %r12
	xorl	%eax, %eax
	movl	$16, %r8d
	subq	%rsi, %r13
	andl	$3, %r13d
	je	.L1752
	cmpq	$1, %r13
	je	.L1705
	cmpq	$2, %r13
	je	.L1706
	movzbl	(%rsi), %edx
	subl	$48, %edx
	cmpb	$9, %dl
	ja	.L1575
	movl	$12, %r8d
	movzbl	%dl, %eax
	leaq	1(%rsi), %r12
.L1706:
	movzbl	(%r12), %r13d
	leal	-48(%r13), %r13d
	cmpb	$9, %r13b
	ja	.L1575
	subl	$4, %r8d
	js	.L1760
	leal	(%rax,%rax,4), %eax
	movzbl	%r13b, %edx
	leal	(%rdx,%rax,2), %eax
.L1728:
	addq	$1, %r12
.L1705:
	movzbl	(%r12), %r13d
	leal	-48(%r13), %r13d
	cmpb	$9, %r13b
	ja	.L1575
	subl	$4, %r8d
	js	.L1761
	leal	(%rax,%rax,4), %eax
	movzbl	%r13b, %edx
	leal	(%rdx,%rax,2), %eax
.L1730:
	addq	$1, %r12
	cmpq	%r12, %rbx
	je	.L1731
.L1752:
	movq	%rdi, 16(%rsp)
.L1582:
	movzbl	(%r12), %edi
	leal	-48(%rdi), %edi
	cmpb	$9, %dil
	ja	.L1754
	subl	$4, %r8d
	js	.L1576
	leal	(%rax,%rax,4), %eax
	movzbl	%dil, %edi
	leal	(%rdi,%rax,2), %eax
.L1577:
	movzbl	1(%r12), %r13d
	leaq	1(%r12), %rdi
	movq	%rdi, %r12
	leal	-48(%r13), %r13d
	cmpb	$9, %r13b
	ja	.L1754
	movl	%r8d, %r12d
	subl	$4, %r12d
	js	.L1762
	leal	(%rax,%rax,4), %eax
	movzbl	%r13b, %r12d
	leal	(%r12,%rax,2), %eax
.L1733:
	movzbl	1(%rdi), %edx
	leaq	1(%rdi), %r12
	leal	-48(%rdx), %r13d
	cmpb	$9, %r13b
	ja	.L1754
	movl	%r8d, %r12d
	subl	$8, %r12d
	js	.L1763
	leal	(%rax,%rax,4), %eax
	movzbl	%r13b, %r12d
	leal	(%r12,%rax,2), %eax
.L1735:
	movzbl	2(%rdi), %edx
	leaq	2(%rdi), %r12
	leal	-48(%rdx), %r13d
	cmpb	$9, %r13b
	ja	.L1754
	subl	$12, %r8d
	js	.L1764
	leal	(%rax,%rax,4), %eax
	movzbl	%r13b, %r13d
	leal	0(%r13,%rax,2), %eax
.L1737:
	leaq	3(%rdi), %r12
	cmpq	%r12, %rbx
	jne	.L1582
.L1731:
	movq	%rbx, %rsi
	movl	$1, %r8d
.L1583:
	movq	8(%rsp), %rdx
	movq	%rbp, %rdi
	andl	$3, %r15d
	andl	$1, %r14d
	andl	$3, %edi
	salq	$2, %r15
	andl	$3, %r8d
	andl	$15, %r11d
	salq	$4, %r14
	orq	%r15, %rdi
	andl	$1, %edx
	salq	$7, %r8
	salq	$6, %rdx
	orq	%r14, %rdi
	salq	$11, %r11
	movzwl	%ax, %ebx
	orq	%rdx, %rdi
	salq	$16, %rbx
	orq	%r8, %rdi
	orq	%r11, %rdi
	orq	%rbx, %rdi
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1556:
	cmpb	$62, %cl
	je	.L1649
	cmpb	$94, %cl
	je	.L1650
	cmpb	$60, %cl
	je	.L1651
	movl	%ecx, %edx
	xorl	%ebp, %ebp
	movl	$32, %ecx
	jmp	.L1628
	.p2align 4,,10
	.p2align 3
.L1651:
	movl	$1, %ebp
	jmp	.L1560
	.p2align 4,,10
	.p2align 3
.L1633:
	movl	$3, %ebp
	jmp	.L1557
	.p2align 4,,10
	.p2align 3
.L1569:
	leaq	1(%rsi), %r13
	cmpq	%r13, %rbx
	je	.L1571
	movzbl	1(%rsi), %edx
	cmpb	$125, %dl
	jne	.L1572
.L1571:
	movq	%rbp, %rdi
	andl	$3, %r15d
	andl	$1, %r14d
	andl	$15, %r11d
	salq	$2, %r15
	andl	$3, %edi
	salq	$4, %r14
	movq	%r13, %rsi
	orq	%r15, %rdi
	salq	$11, %r11
	orq	%r14, %rdi
	orq	$64, %rdi
	orq	%r11, %rdi
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1650:
	movl	$3, %ebp
	jmp	.L1560
	.p2align 4,,10
	.p2align 3
.L1573:
	cmpb	$123, %dl
	je	.L1765
	cmpq	%rsi, %rbx
	je	.L1648
	xorl	%r8d, %r8d
	xorl	%eax, %eax
.L1624:
	cmpb	$76, %dl
	je	.L1766
	subl	$66, %edx
	cmpb	$54, %dl
	ja	.L1639
	leaq	.L1612(%rip), %r13
	movzbl	%dl, %edx
	movslq	0(%r13,%rdx,4), %r12
	addq	%r13, %r12
	notrack jmp	*%r12
	.section	.rodata._ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,"aG",@progbits,_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,comdat
	.align 4
	.align 4
.L1612:
	.long	.L1640-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1641-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1642-.L1612
	.long	.L1643-.L1612
	.long	.L1644-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1645-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1646-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1639-.L1612
	.long	.L1647-.L1612
	.section	.text._ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,"axG",@progbits,_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,comdat
	.p2align 4,,10
	.p2align 3
.L1639:
	xorl	%r13d, %r13d
.L1610:
	cmpq	%rsi, %rbx
	je	.L1621
.L1622:
	cmpb	$125, (%rsi)
	jne	.L1623
.L1621:
	sall	$5, %r13d
	movzbl	%r15b, %r15d
	movzbl	%r14b, %r14d
	sall	$7, %r8d
	sall	$2, %r15d
	sall	$4, %r14d
	orl	%ebp, %r15d
	movzbl	8(%rsp), %ebp
	sall	$11, %r11d
	orl	%r14d, %r15d
	orl	%r13d, %r15d
	sall	$6, %ebp
	orl	%ebp, %r15d
	orl	%r8d, %r15d
	movabsq	$-4294901761, %r8
	movw	%r15w, %di
	orw	%r11w, %di
	movzwl	%ax, %r11d
	andq	%rdi, %r8
	salq	$16, %r11
	orq	%r11, %r8
	movq	%r8, %rdi
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1765:
	leaq	1(%rsi), %r13
	cmpq	%r13, %rbx
	je	.L1767
	movsbw	1(%rsi), %ax
	cmpb	$125, %al
	je	.L1768
	cmpb	$48, %al
	je	.L1769
	leal	-49(%rax), %edx
	cmpb	$8, %dl
	jbe	.L1770
.L1592:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L1647:
	xorl	%r13d, %r13d
.L1611:
	addq	$1, %rsi
	movl	$5, %r11d
	jmp	.L1610
.L1646:
	xorl	%r13d, %r13d
.L1613:
	testl	%r10d, %r10d
	jne	.L1620
	addq	$1, %rsi
	xorl	%r11d, %r11d
	jmp	.L1610
.L1645:
	xorl	%r13d, %r13d
.L1614:
	addq	$1, %rsi
	movl	$4, %r11d
	jmp	.L1610
.L1644:
	xorl	%r13d, %r13d
.L1615:
	addq	$1, %rsi
	movl	$1, %r11d
	jmp	.L1610
.L1643:
	xorl	%r13d, %r13d
.L1616:
	testl	%r10d, %r10d
	je	.L1620
	addq	$1, %rsi
	movl	$7, %r11d
	jmp	.L1610
.L1642:
	xorl	%r13d, %r13d
.L1617:
	addq	$1, %rsi
	movl	$2, %r11d
	jmp	.L1610
.L1641:
	xorl	%r13d, %r13d
.L1618:
	addq	$1, %rsi
	movl	$6, %r11d
	jmp	.L1610
.L1640:
	xorl	%r13d, %r13d
.L1619:
	addq	$1, %rsi
	movl	$3, %r11d
	jmp	.L1610
	.p2align 4,,10
	.p2align 3
.L1766:
	leaq	1(%rsi), %rdx
	cmpq	%rdx, %rbx
	je	.L1608
	movzbl	1(%rsi), %esi
	cmpb	$125, %sil
	jne	.L1609
.L1608:
	movq	%rbp, %rdi
	andl	$3, %r15d
	movq	8(%rsp), %r12
	andl	$1, %r14d
	andl	$3, %edi
	salq	$2, %r15
	andl	$3, %r8d
	andl	$15, %r11d
	salq	$4, %r14
	orq	%r15, %rdi
	andl	$1, %r12d
	salq	$7, %r8
	orq	%r14, %rdi
	salq	$6, %r12
	movzwl	%ax, %eax
	movq	%rdx, %rsi
	salq	$11, %r11
	orq	$32, %rdi
	salq	$16, %rax
	orq	%r12, %rdi
	orq	%r8, %rdi
	orq	%r11, %rdi
	orq	%rax, %rdi
	jmp	.L1563
	.p2align 4,,10
	.p2align 3
.L1754:
	movq	16(%rsp), %rdi
.L1575:
	cmpq	%rsi, %r12
	je	.L1574
	movq	%r12, %rsi
	movl	$1, %r8d
.L1605:
	cmpq	%rsi, %rbx
	je	.L1583
	movzbl	(%rsi), %edx
	cmpb	$125, %dl
	jne	.L1624
	jmp	.L1583
	.p2align 4,,10
	.p2align 3
.L1576:
	movl	$10, %r13d
	mulw	%r13w
	jo	.L1574
	movzbl	%dil, %edx
	addw	%ax, %dx
	jc	.L1574
	movl	%edx, %eax
	jmp	.L1577
.L1762:
	movl	$10, %edx
	mulw	%dx
	jo	.L1574
	movzbl	%r13b, %r13d
	addw	%ax, %r13w
	jc	.L1574
	movl	%r13d, %eax
	jmp	.L1733
.L1763:
	movl	$10, %edx
	mulw	%dx
	jo	.L1574
	movzbl	%r13b, %r13d
	addw	%ax, %r13w
	jc	.L1574
	movl	%r13d, %eax
	jmp	.L1735
.L1764:
	movl	$10, %r12d
	mulw	%r12w
	jo	.L1574
	movzbl	%r13b, %edx
	addw	%ax, %dx
	jc	.L1574
	movl	%edx, %eax
	jmp	.L1737
.L1648:
	xorl	%r8d, %r8d
	xorl	%eax, %eax
	jmp	.L1583
.L1768:
	cmpl	$1, 16(%r8)
	je	.L1604
	movq	24(%r8), %rax
	movl	$2, 16(%r8)
	leaq	1(%rax), %r12
	movq	%r12, 24(%r8)
.L1589:
	leaq	1(%r13), %rdx
	xorl	%r8d, %r8d
	cmpq	%rsi, %rdx
	je	.L1605
	movq	%rdx, %rsi
	movl	$2, %r8d
	jmp	.L1605
.L1620:
	cmpq	%rsi, %rbx
	je	.L1621
.L1623:
	call	_ZNSt8__format29__failed_to_parse_format_specEv
	.p2align 4,,10
	.p2align 3
.L1769:
	leaq	2(%rsi), %r13
	xorl	%eax, %eax
.L1591:
	cmpq	%r13, %rbx
	je	.L1592
.L1630:
	cmpb	$125, 0(%r13)
	jne	.L1592
	cmpl	$2, 16(%r8)
	je	.L1604
	movl	$1, 16(%r8)
	jmp	.L1589
.L1770:
	leaq	2(%rsi), %rdx
	cmpq	%rdx, %rbx
	je	.L1592
	movzbl	2(%rsi), %r12d
	subl	$48, %r12d
	cmpb	$9, %r12b
	ja	.L1593
	movq	%rbx, %rdx
	movq	%r13, 16(%rsp)
	xorl	%eax, %eax
	subq	%r13, %rdx
	movl	$16, 36(%rsp)
	andl	$3, %edx
	je	.L1750
	cmpq	$1, %rdx
	je	.L1707
	cmpq	$2, %rdx
	je	.L1708
	movzbl	0(%r13), %r12d
	subl	$48, %r12d
	cmpb	$9, %r12b
	ja	.L1594
	leaq	1(%r13), %rdx
	movl	$12, 36(%rsp)
	movzbl	%r12b, %eax
	movq	%rdx, 16(%rsp)
.L1708:
	movq	16(%rsp), %r12
	movzbl	(%r12), %edx
	leal	-48(%rdx), %r12d
	movb	%dl, 34(%rsp)
	cmpb	$9, %r12b
	ja	.L1594
	subl	$4, 36(%rsp)
	js	.L1771
	leal	(%rax,%rax,4), %eax
	movzbl	%r12b, %edx
	leal	(%rdx,%rax,2), %eax
.L1739:
	addq	$1, 16(%rsp)
.L1707:
	movq	16(%rsp), %r12
	movzbl	(%r12), %edx
	leal	-48(%rdx), %r12d
	movb	%dl, 34(%rsp)
	cmpb	$9, %r12b
	ja	.L1594
	subl	$4, 36(%rsp)
	js	.L1772
	leal	(%rax,%rax,4), %eax
	movzbl	%r12b, %edx
	leal	(%rdx,%rax,2), %eax
.L1741:
	addq	$1, 16(%rsp)
	movq	16(%rsp), %r12
	cmpq	%r12, %rbx
	je	.L1602
	movb	%r11b, 34(%rsp)
	movq	16(%rsp), %r12
	movb	%bpl, 35(%rsp)
	movq	%r13, 24(%rsp)
	movl	36(%rsp), %r13d
.L1601:
	movzbl	(%r12), %ebp
	leal	-48(%rbp), %r11d
	cmpb	$9, %r11b
	ja	.L1757
	subl	$4, %r13d
	js	.L1595
	leal	(%rax,%rax,4), %eax
	movzbl	%r11b, %r11d
	leal	(%r11,%rax,2), %eax
.L1596:
	movzbl	1(%r12), %edx
	leaq	1(%r12), %r11
	movq	%r11, %r12
	leal	-48(%rdx), %ebp
	cmpb	$9, %bpl
	ja	.L1757
	movl	%r13d, %r12d
	subl	$4, %r12d
	js	.L1773
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %r12d
	leal	(%r12,%rax,2), %eax
.L1743:
	movzbl	1(%r11), %edx
	leaq	1(%r11), %r12
	leal	-48(%rdx), %ebp
	cmpb	$9, %bpl
	ja	.L1757
	movl	%r13d, %r12d
	subl	$8, %r12d
	js	.L1774
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %r12d
	leal	(%r12,%rax,2), %eax
.L1745:
	movzbl	2(%r11), %edx
	leaq	2(%r11), %r12
	leal	-48(%rdx), %ebp
	cmpb	$9, %bpl
	ja	.L1757
	subl	$12, %r13d
	js	.L1775
	leal	(%rax,%rax,4), %eax
	movzbl	%bpl, %ebp
	leal	0(%rbp,%rax,2), %eax
.L1747:
	leaq	3(%r11), %r12
	cmpq	%r12, %rbx
	jne	.L1601
	movzbl	34(%rsp), %r11d
	movzbl	35(%rsp), %ebp
	movq	%r12, 16(%rsp)
.L1602:
	movq	16(%rsp), %r13
	jmp	.L1591
.L1761:
	movl	$10, %edx
	mulw	%dx
	jo	.L1574
	movzbl	%r13b, %r13d
	addw	%ax, %r13w
	jc	.L1574
	movl	%r13d, %eax
	jmp	.L1730
.L1760:
	movl	$10, %edx
	mulw	%dx
	jo	.L1574
	movzbl	%r13b, %r13d
	addw	%ax, %r13w
	jc	.L1574
	movl	%r13d, %eax
	jmp	.L1728
.L1593:
	subl	$48, %eax
	movq	%rdx, %r13
	jmp	.L1630
.L1757:
	movzbl	34(%rsp), %r11d
	movq	24(%rsp), %r13
	movq	%r12, 16(%rsp)
	movzbl	35(%rsp), %ebp
.L1594:
	movq	16(%rsp), %rdx
	cmpq	%rdx, %r13
	jne	.L1602
	jmp	.L1592
.L1750:
	movq	%r13, 24(%rsp)
	movq	16(%rsp), %r12
	movl	$16, %r13d
	movb	%r11b, 34(%rsp)
	movb	%bpl, 35(%rsp)
	jmp	.L1601
.L1595:
	movl	$10, %edx
	mulw	%dx
	jo	.L1592
	movzbl	%r11b, %ebp
	addw	%ax, %bp
	jc	.L1592
	movl	%ebp, %eax
	jmp	.L1596
.L1773:
	movl	$10, %edx
	mulw	%dx
	jo	.L1592
	movzbl	%bpl, %ebp
	addw	%bp, %ax
	jnc	.L1743
	jmp	.L1592
.L1775:
	movl	$10, %r12d
	mulw	%r12w
	jo	.L1592
	movzbl	%bpl, %edx
	addw	%dx, %ax
	jnc	.L1747
	jmp	.L1592
.L1774:
	movl	$10, %edx
	mulw	%dx
	jo	.L1592
	movzbl	%bpl, %ebp
	addw	%bp, %ax
	jnc	.L1745
	jmp	.L1592
.L1772:
	movl	$10, %edx
	mulw	%dx
	jo	.L1592
	movzbl	%r12b, %r12d
	addw	%r12w, %ax
	jnc	.L1741
	jmp	.L1592
.L1771:
	movl	$10, %edx
	mulw	%dx
	jo	.L1592
	movzbl	%r12b, %r12d
	addw	%r12w, %ax
	jnc	.L1739
	jmp	.L1592
.L1609:
	subl	$66, %esi
	cmpb	$54, %sil
	ja	.L1652
	leaq	.L1629(%rip), %r12
	movzbl	%sil, %r13d
	movslq	(%r12,%r13,4), %rsi
	addq	%r12, %rsi
	notrack jmp	*%rsi
	.section	.rodata._ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,"aG",@progbits,_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,comdat
	.align 4
	.align 4
.L1629:
	.long	.L1653-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1654-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1655-.L1629
	.long	.L1656-.L1629
	.long	.L1657-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1658-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1659-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1652-.L1629
	.long	.L1660-.L1629
	.section	.text._ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,"axG",@progbits,_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE,comdat
.L1572:
	movb	$1, 8(%rsp)
	movq	%r13, %rsi
	cmpb	$48, %dl
	jne	.L1570
	leaq	.LC22(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.L1660:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1611
.L1659:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1613
.L1658:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1614
.L1657:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1615
.L1656:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1616
.L1655:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1617
.L1654:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1618
.L1653:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1619
.L1652:
	movq	%rdx, %rsi
	movl	$1, %r13d
	jmp	.L1622
.L1767:
	call	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
.L1574:
	leaq	.LC23(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.L1604:
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
	.cfi_endproc
.LFE12357:
	.size	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE, .-_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	.section	.text._ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,"axG",@progbits,_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	.type	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE, @function
_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE:
.LFB12380:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
	movq	8(%rsi), %rbp
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	movabsq	$9007199254740992, %rax
	movq	%rax, (%rsp)
	movq	(%rsi), %rax
	cmpq	%rbp, %rax
	je	.L1777
	movzbl	(%rax), %edx
	cmpb	$125, %dl
	je	.L1777
	movq	%rsi, %rbx
	cmpb	$123, %dl
	je	.L1778
	movq	%rbp, %rcx
	subq	%rax, %rcx
	cmpq	$1, %rcx
	jle	.L1779
	movzbl	1(%rax), %esi
	cmpb	$62, %sil
	je	.L1841
	cmpb	$94, %sil
	je	.L1842
	cmpb	$60, %sil
	jne	.L1781
	movl	$1, %edi
.L1780:
	movb	%dl, 6(%rsp)
	addq	$2, %rax
.L1782:
	movzbl	(%rsp), %r8d
	andl	$-4, %r8d
	orl	%edi, %r8d
	movb	%r8b, (%rsp)
	cmpq	%rax, %rbp
	je	.L1777
.L1784:
	movzbl	(%rax), %edx
	cmpb	$125, %dl
	je	.L1777
.L1838:
	leal	-32(%rdx), %r9d
	cmpb	$13, %r9b
	ja	.L1836
	movzbl	%r9b, %r10d
	leaq	CSWTCH.915(%rip), %r11
	movl	(%r11,%r10,4), %r13d
	testl	%r13d, %r13d
	jne	.L1837
	movq	%rax, %rcx
	cmpb	$35, %dl
	je	.L1789
.L1790:
	cmpb	$46, %dl
	jne	.L1929
.L1791:
	leaq	1(%rax), %r13
	cmpq	%r13, %rbp
	je	.L1808
	movzbl	1(%rax), %edi
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %r9
	cmpb	$9, (%r9,%rdi)
	ja	.L1795
	movq	%rbp, %rdx
	movq	%r13, %rcx
	xorl	%eax, %eax
	movl	$16, %r9d
	subq	%r13, %rdx
	andl	$3, %edx
	je	.L1804
	cmpq	$1, %rdx
	je	.L1895
	cmpq	$2, %rdx
	je	.L1896
	movzbl	0(%r13), %esi
	leal	-48(%rsi), %edi
	cmpb	$9, %dil
	ja	.L1796
	movl	$12, %r9d
	movzbl	%dil, %eax
	leaq	1(%r13), %rcx
.L1896:
	movzbl	(%rcx), %r8d
	leal	-48(%r8), %r10d
	cmpb	$9, %r10b
	ja	.L1796
	subl	$4, %r9d
	js	.L1930
	leal	(%rax,%rax,4), %eax
	movzbl	%r10b, %edx
	leal	(%rdx,%rax,2), %eax
.L1917:
	addq	$1, %rcx
.L1895:
	movzbl	(%rcx), %esi
	leal	-48(%rsi), %edi
	cmpb	$9, %dil
	ja	.L1796
	subl	$4, %r9d
	js	.L1931
	leal	(%rax,%rax,4), %r11d
	movzbl	%dil, %ebx
	leal	(%rbx,%r11,2), %eax
.L1919:
	addq	$1, %rcx
	cmpq	%rcx, %rbp
	je	.L1920
.L1804:
	movzbl	(%rcx), %edx
	leal	-48(%rdx), %esi
	cmpb	$9, %sil
	ja	.L1796
	subl	$4, %r9d
	js	.L1797
	leal	(%rax,%rax,4), %eax
	movzbl	%sil, %r10d
	leal	(%r10,%rax,2), %eax
.L1798:
	movzbl	1(%rcx), %ebx
	leaq	1(%rcx), %r11
	movq	%r11, %rcx
	leal	-48(%rbx), %esi
	cmpb	$9, %sil
	ja	.L1796
	movl	%r9d, %ecx
	subl	$4, %ecx
	js	.L1932
	leal	(%rax,%rax,4), %eax
	movzbl	%sil, %r8d
	leal	(%r8,%rax,2), %eax
.L1922:
	movzbl	1(%r11), %r10d
	leaq	1(%r11), %rcx
	leal	-48(%r10), %esi
	cmpb	$9, %sil
	ja	.L1796
	movl	%r9d, %ebx
	subl	$8, %ebx
	js	.L1933
	leal	(%rax,%rax,4), %edi
	movzbl	%sil, %eax
	leal	(%rax,%rdi,2), %eax
.L1924:
	movzbl	2(%r11), %r8d
	leaq	2(%r11), %rcx
	leal	-48(%r8), %r10d
	cmpb	$9, %r10b
	ja	.L1796
	subl	$12, %r9d
	js	.L1934
	leal	(%rax,%rax,4), %ecx
	movzbl	%r10b, %edx
	leal	(%rdx,%rcx,2), %eax
.L1926:
	leaq	3(%r11), %rcx
	cmpq	%rcx, %rbp
	jne	.L1804
.L1920:
	movzbl	1(%rsp), %ebp
	movw	%ax, 4(%rsp)
	andl	$-7, %ebp
	orl	$2, %ebp
	movb	%bpl, 1(%rsp)
	jmp	.L1849
	.p2align 4,,10
	.p2align 3
.L1837:
	movzbl	(%rsp), %edx
	andl	$3, %r13d
	addq	$1, %rax
	sall	$2, %r13d
	andl	$-13, %edx
	orl	%r13d, %edx
	movb	%dl, (%rsp)
	cmpq	%rax, %rbp
	jne	.L1778
	.p2align 4,,10
	.p2align 3
.L1777:
	movl	(%rsp), %ecx
	movl	3(%rsp), %esi
	movl	%ecx, (%r12)
	movl	%esi, 3(%r12)
	movq	8(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L1915
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L1778:
	.cfi_restore_state
	movzbl	(%rax), %edx
	cmpb	$125, %dl
	je	.L1777
	movq	%rax, %rcx
	cmpb	$35, %dl
	jne	.L1836
.L1789:
	addq	$1, %rax
	orb	$16, (%rsp)
	cmpq	%rax, %rbp
	je	.L1777
	movzbl	1(%rcx), %edx
	cmpb	$125, %dl
	je	.L1777
	.p2align 4,,10
	.p2align 3
.L1836:
	cmpb	$48, %dl
	jne	.L1790
	leaq	1(%rax), %rsi
	orb	$64, (%rsp)
	cmpq	%rsi, %rbp
	je	.L1848
	movzbl	1(%rax), %edx
	movq	%rsi, %rax
	cmpb	$125, %dl
	je	.L1777
	jmp	.L1790
	.p2align 4,,10
	.p2align 3
.L1841:
	movl	$2, %edi
	jmp	.L1780
	.p2align 4,,10
	.p2align 3
.L1779:
	cmpb	$62, %dl
	je	.L1851
	cmpb	$94, %dl
	je	.L1852
	cmpb	$60, %dl
	jne	.L1838
.L1853:
	movl	$1, %edi
.L1783:
	addq	$1, %rax
	jmp	.L1782
	.p2align 4,,10
	.p2align 3
.L1781:
	cmpb	$62, %dl
	je	.L1851
	cmpb	$94, %dl
	je	.L1852
	cmpb	$60, %dl
	jne	.L1784
	jmp	.L1853
	.p2align 4,,10
	.p2align 3
.L1851:
	movl	$2, %edi
	jmp	.L1783
	.p2align 4,,10
	.p2align 3
.L1842:
	movl	$3, %edi
	jmp	.L1780
	.p2align 4,,10
	.p2align 3
.L1852:
	movl	$3, %edi
	jmp	.L1783
	.p2align 4,,10
	.p2align 3
.L1929:
	movq	%rbx, %rcx
	movq	%rsp, %rdi
	movq	%rbp, %rdx
	movq	%rax, %rsi
	call	_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE
	movq	%rax, %rcx
	cmpq	%rbp, %rax
	je	.L1848
	movzbl	(%rax), %eax
	cmpb	$125, %al
	je	.L1849
	cmpb	$46, %al
	je	.L1935
.L1835:
	cmpb	$76, %al
	je	.L1936
	subl	$65, %eax
	cmpb	$38, %al
	ja	.L1822
	leaq	.L1824(%rip), %r11
	movzbl	%al, %r9d
	movslq	(%r11,%r9,4), %rdi
	addq	%r11, %rdi
	notrack jmp	*%rdi
	.section	.rodata._ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,"aG",@progbits,_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,comdat
	.align 4
	.align 4
.L1824:
	.long	.L1831-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1830-.L1824
	.long	.L1829-.L1824
	.long	.L1828-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1827-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1822-.L1824
	.long	.L1826-.L1824
	.long	.L1825-.L1824
	.long	.L1823-.L1824
	.section	.text._ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,"axG",@progbits,_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,comdat
	.p2align 4,,10
	.p2align 3
.L1848:
	movq	%rbp, %rax
	jmp	.L1777
.L1859:
	movq	%rax, %rcx
.L1827:
	movzbl	1(%rsp), %r9d
	addq	$1, %rcx
	andl	$-121, %r9d
	orl	$8, %r9d
	movb	%r9b, 1(%rsp)
	.p2align 4,,10
	.p2align 3
.L1822:
	cmpq	%rcx, %rbp
	je	.L1849
.L1832:
	cmpb	$125, (%rcx)
	jne	.L1937
.L1849:
	movq	%rcx, %rax
	jmp	.L1777
.L1862:
	movq	%rax, %rcx
.L1823:
	movzbl	1(%rsp), %edx
	addq	$1, %rcx
	andl	$-121, %edx
	orl	$56, %edx
	movb	%dl, 1(%rsp)
	jmp	.L1822
.L1861:
	movq	%rax, %rcx
.L1825:
	movzbl	1(%rsp), %eax
	addq	$1, %rcx
	andl	$-121, %eax
	orl	$40, %eax
	movb	%al, 1(%rsp)
	jmp	.L1822
.L1860:
	movq	%rax, %rcx
.L1826:
	movzbl	1(%rsp), %r13d
	addq	$1, %rcx
	andl	$-121, %r13d
	orl	$24, %r13d
	movb	%r13b, 1(%rsp)
	jmp	.L1822
.L1858:
	movq	%rax, %rcx
.L1828:
	movzbl	1(%rsp), %r11d
	addq	$1, %rcx
	andl	$-121, %r11d
	orl	$64, %r11d
	movb	%r11b, 1(%rsp)
	jmp	.L1822
.L1857:
	movq	%rax, %rcx
.L1829:
	movzbl	1(%rsp), %edi
	addq	$1, %rcx
	andl	$-121, %edi
	orl	$48, %edi
	movb	%dil, 1(%rsp)
	jmp	.L1822
.L1856:
	movq	%rax, %rcx
.L1830:
	movzbl	1(%rsp), %r8d
	addq	$1, %rcx
	andl	$-121, %r8d
	orl	$32, %r8d
	movb	%r8b, 1(%rsp)
	jmp	.L1822
.L1855:
	movq	%rax, %rcx
.L1831:
	movzbl	1(%rsp), %r10d
	addq	$1, %rcx
	andl	$-121, %r10d
	orl	$16, %r10d
	movb	%r10b, 1(%rsp)
	jmp	.L1822
	.p2align 4,,10
	.p2align 3
.L1936:
	leaq	1(%rcx), %rax
	orb	$32, (%rsp)
	cmpq	%rax, %rbp
	je	.L1777
	movzbl	1(%rcx), %r8d
	cmpb	$125, %r8b
	je	.L1777
	subl	$65, %r8d
	cmpb	$38, %r8b
	ja	.L1854
	leaq	.L1839(%rip), %rsi
	movzbl	%r8b, %r10d
	movslq	(%rsi,%r10,4), %rbx
	addq	%rsi, %rbx
	notrack jmp	*%rbx
	.section	.rodata._ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,"aG",@progbits,_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,comdat
	.align 4
	.align 4
.L1839:
	.long	.L1855-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1856-.L1839
	.long	.L1857-.L1839
	.long	.L1858-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1859-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1854-.L1839
	.long	.L1860-.L1839
	.long	.L1861-.L1839
	.long	.L1862-.L1839
	.section	.text._ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,"axG",@progbits,_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE,comdat
	.p2align 4,,10
	.p2align 3
.L1795:
	cmpb	$123, %dil
	je	.L1938
.L1808:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L1915
	leaq	.LC24(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.p2align 4,,10
	.p2align 3
.L1938:
	leaq	2(%rax), %rdi
	cmpq	%rdi, %rbp
	je	.L1939
	cmpb	$125, 2(%rax)
	je	.L1940
	movq	%rbp, %rsi
	call	_ZNSt8__format14__parse_arg_idIcEESt4pairItPKT_ES4_S4_
	movq	%rdx, %rdi
	cmpq	%rdx, %rbp
	je	.L1815
	testq	%rdx, %rdx
	je	.L1815
	cmpb	$125, (%rdx)
	jne	.L1815
	cmpl	$2, 16(%rbx)
	je	.L1818
	movl	$1, 16(%rbx)
	movw	%ax, 4(%rsp)
.L1814:
	leaq	1(%rdi), %rcx
	cmpq	%rcx, %r13
	je	.L1808
	movl	$2, %ebx
.L1806:
	movzbl	1(%rsp), %r13d
	addl	%ebx, %ebx
	andl	$-7, %r13d
	orl	%ebx, %r13d
	movb	%r13b, 1(%rsp)
	cmpq	%rcx, %rbp
	je	.L1849
	movzbl	(%rcx), %eax
	cmpb	$125, %al
	je	.L1849
	jmp	.L1835
.L1796:
	cmpq	%rcx, %r13
	je	.L1801
	movw	%ax, 4(%rsp)
	movl	$1, %ebx
	jmp	.L1806
.L1797:
	movl	$10, %edi
	mulw	%di
	jo	.L1801
	movzbl	%sil, %r8d
	addw	%ax, %r8w
	jc	.L1801
	movl	%r8d, %eax
	jmp	.L1798
.L1932:
	movl	$10, %edx
	mulw	%dx
	jo	.L1801
	movzbl	%sil, %edi
	addw	%ax, %di
	jc	.L1801
	movl	%edi, %eax
	jmp	.L1922
.L1933:
	movl	$10, %ecx
	mulw	%cx
	jo	.L1801
	movzbl	%sil, %edx
	addw	%ax, %dx
	jc	.L1801
	movl	%edx, %eax
	jmp	.L1924
.L1934:
	movl	$10, %esi
	mulw	%si
	jo	.L1801
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jc	.L1801
	movl	%ebx, %eax
	jmp	.L1926
.L1940:
	cmpl	$1, 16(%rbx)
	je	.L1818
	movq	24(%rbx), %r10
	movl	$2, 16(%rbx)
	leaq	1(%r10), %r11
	movw	%r10w, 4(%rsp)
	movq	%r11, 24(%rbx)
	jmp	.L1814
.L1931:
	movl	$10, %r8d
	mulw	%r8w
	jo	.L1801
	movzbl	%dil, %r10d
	addw	%ax, %r10w
	jc	.L1801
	movl	%r10d, %eax
	jmp	.L1919
.L1930:
	movl	$10, %r11d
	mulw	%r11w
	jo	.L1801
	movzbl	%r10b, %ebx
	addw	%ax, %bx
	jc	.L1801
	movl	%ebx, %eax
	jmp	.L1917
.L1801:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	je	.L1807
.L1915:
	call	__stack_chk_fail@PLT
.L1935:
	movq	%rcx, %rax
	jmp	.L1791
.L1854:
	movq	%rax, %rcx
	jmp	.L1832
.L1939:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L1915
	call	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
.L1818:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L1915
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
.L1937:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L1915
	call	_ZNSt8__format29__failed_to_parse_format_specEv
.L1815:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L1915
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
.L1807:
	leaq	.LC23(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE12380:
	.size	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE, .-_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv:
.LFB12506:
	.cfi_startproc
	endbr64
	movq	(%rdi), %rax
	leaq	16(%rdi), %rdx
	cmpq	%rdx, %rax
	je	.L1943
	movq	16(%rdi), %rsi
	movq	%rax, %rdi
	addq	$1, %rsi
	jmp	_ZdlPvm@PLT
	.p2align 4,,10
	.p2align 3
.L1943:
	ret
	.cfi_endproc
.LFE12506:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	.section	.text._ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_,"axG",@progbits,_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_,comdat
	.p2align 4
	.weak	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	.type	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_, @function
_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_:
.LFB12648:
	.cfi_startproc
	endbr64
	movq	%rdi, %rcx
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movl	%edx, %eax
	movq	%rsi, %r8
	movq	%fs:40, %rdx
	movq	%rdx, 24(%rsp)
	xorl	%edx, %edx
	movq	%rsi, %rdi
	subq	%rcx, %r8
	testl	%eax, %eax
	je	.L1945
	bsrl	%eax, %edx
	addl	$4, %edx
	shrl	$2, %edx
	movl	%edx, %esi
	cmpq	%r8, %rsi
	jg	.L1952
	vmovdqa	.LC25(%rip), %xmm0
	subl	$1, %edx
	leaq	(%rcx,%rsi), %rdi
	vmovdqa	%xmm0, (%rsp)
	cmpl	$255, %eax
	jbe	.L1947
	movl	%eax, %r10d
	movl	%eax, %esi
	movl	%edx, %r9d
	shrl	$8, %eax
	andl	$15, %r10d
	shrl	$4, %esi
	leal	-1(%rdx), %r8d
	movzbl	(%rsp,%r10), %r11d
	andl	$15, %esi
	leal	-2(%rdx), %r10d
	movb	%r11b, (%rcx,%r9)
	movzbl	(%rsp,%rsi), %r9d
	movb	%r9b, (%rcx,%r8)
	cmpl	$255, %eax
	jbe	.L1947
	movl	%eax, %r11d
	movl	%eax, %r9d
	leal	-3(%rdx), %r8d
	shrl	$8, %eax
	andl	$15, %r11d
	shrl	$4, %r9d
	movzbl	(%rsp,%r11), %esi
	andl	$15, %r9d
	leal	-4(%rdx), %r11d
	movb	%sil, (%rcx,%r10)
	movzbl	(%rsp,%r9), %r10d
	movb	%r10b, (%rcx,%r8)
	cmpl	$255, %eax
	jbe	.L1947
	movl	%eax, %esi
	movl	%eax, %r10d
	leal	-5(%rdx), %r8d
	shrl	$8, %eax
	andl	$15, %esi
	shrl	$4, %r10d
	movzbl	(%rsp,%rsi), %r9d
	andl	$15, %r10d
	leal	-6(%rdx), %esi
	movb	%r9b, (%rcx,%r11)
	movzbl	(%rsp,%r10), %r11d
	movb	%r11b, (%rcx,%r8)
	cmpl	$255, %eax
	jbe	.L1947
	movl	%eax, %r11d
	movl	%eax, %r9d
	leal	-7(%rdx), %edx
	shrl	$8, %eax
	shrl	$4, %r11d
	andl	$15, %r9d
	andl	$15, %r11d
	movzbl	(%rsp,%r9), %r10d
	movzbl	(%rsp,%r11), %r8d
	movb	%r10b, (%rcx,%rsi)
	movb	%r8b, (%rcx,%rdx)
.L1947:
	cmpl	$15, %eax
	jbe	.L1949
	movl	%eax, %esi
	shrl	$4, %eax
	andl	$15, %esi
	movzbl	(%rsp,%rax), %eax
	movzbl	(%rsp,%rsi), %r9d
	movb	%r9b, 1(%rcx)
.L1950:
	movb	%al, (%rcx)
	xorl	%edx, %edx
.L1946:
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L1963
	movq	%rdi, %rax
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L1945:
	.cfi_restore_state
	movl	$75, %edx
	testq	%r8, %r8
	js	.L1946
	vmovdqa	.LC25(%rip), %xmm1
	movq	%rcx, %rdi
	vmovdqa	%xmm1, (%rsp)
.L1949:
	movzbl	(%rsp,%rax), %eax
	jmp	.L1950
.L1963:
	call	__stack_chk_fail@PLT
.L1952:
	movl	$75, %edx
	jmp	.L1946
	.cfi_endproc
.LFE12648:
	.size	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_, .-_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	.section	.text._ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_,"axG",@progbits,_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_,comdat
	.p2align 4
	.weak	_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	.type	_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_, @function
_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_:
.LFB12650:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	%edx, %eax
	movq	%rsi, %r9
	movq	%rdi, %r8
	subq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$224, %rsp
	movq	%fs:40, %rdx
	movq	%rdx, 216(%rsp)
	xorl	%edx, %edx
	cmpl	$9, %eax
	jbe	.L1965
	cmpl	$99, %eax
	jbe	.L1985
	cmpl	$999, %eax
	jbe	.L1986
	cmpl	$9999, %eax
	jbe	.L2003
	movl	%eax, %edx
	movl	$5, %edi
	cmpl	$99999, %eax
	jbe	.L1975
	cmpl	$999999, %eax
	jbe	.L2004
	cmpl	$9999999, %eax
	jbe	.L2005
	cmpl	$99999999, %eax
	jbe	.L1989
	cmpq	$999999999, %rdx
	jbe	.L1990
	movl	$9, %ecx
.L1976:
	leal	1(%rcx), %r11d
	movq	%r11, %rdx
.L1966:
	cmpq	%rsi, %r11
	jg	.L1992
	vmovdqa	.LC26(%rip), %ymm7
	vmovdqa	.LC27(%rip), %ymm8
	movl	%edx, %r9d
	vmovdqa	.LC28(%rip), %ymm9
	vmovdqa	.LC29(%rip), %ymm10
	vmovdqa	.LC30(%rip), %ymm11
	vmovdqa	.LC31(%rip), %ymm12
	vmovdqu	%ymm7, (%rsp)
	vmovdqa	.LC32(%rip), %xmm13
	vmovdqu	%ymm8, 32(%rsp)
	vmovdqu	%ymm12, 160(%rsp)
	vmovdqu	%ymm9, 64(%rsp)
	vmovdqu	%ymm10, 96(%rsp)
	vmovdqu	%ymm11, 128(%rsp)
	vmovdqu	%xmm13, 185(%rsp)
	cmpl	$99, %eax
	jbe	.L1978
	subl	$1, %edx
	jmp	.L1979
	.p2align 4,,10
	.p2align 3
.L1978:
	addl	%eax, %eax
	leal	1(%rax), %edx
	movzbl	(%rsp,%rax), %eax
	movzbl	(%rsp,%rdx), %ecx
	movb	%cl, 1(%r8)
	vzeroupper
.L1981:
	movb	%al, (%r8)
	addq	%r8, %r9
	xorl	%edx, %edx
.L1971:
	movq	216(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L2006
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	movq	%r9, %rax
	ret
	.p2align 4,,10
	.p2align 3
.L1990:
	.cfi_restore_state
	movl	$9, %edi
.L1975:
	movl	%edi, %r10d
.L1974:
	cmpq	%rsi, %r10
	jg	.L1992
	vmovdqa	.LC26(%rip), %ymm1
	vmovdqa	.LC27(%rip), %ymm2
	leal	-1(%rdi), %edx
	movl	%edi, %r9d
	vmovdqa	.LC28(%rip), %ymm3
	vmovdqa	.LC29(%rip), %ymm4
	vmovdqa	.LC30(%rip), %ymm5
	vmovdqa	.LC31(%rip), %ymm6
	vmovdqu	%ymm1, (%rsp)
	vmovdqa	.LC32(%rip), %xmm0
	vmovdqu	%ymm2, 32(%rsp)
	vmovdqu	%ymm6, 160(%rsp)
	vmovdqu	%ymm3, 64(%rsp)
	vmovdqu	%ymm4, 96(%rsp)
	vmovdqu	%ymm5, 128(%rsp)
	vmovdqu	%xmm0, 185(%rsp)
	.p2align 4,,10
	.p2align 3
.L1979:
	movl	%eax, %esi
	movl	%eax, %r11d
	imulq	$1374389535, %rsi, %rcx
	movl	%eax, %esi
	shrq	$37, %rcx
	imull	$100, %ecx, %edi
	movl	%ecx, %eax
	subl	%edi, %esi
	movl	%edx, %edi
	addl	%esi, %esi
	leal	1(%rsi), %r10d
	movzbl	(%rsp,%rsi), %esi
	movzbl	(%rsp,%r10), %r10d
	movb	%r10b, (%r8,%rdi)
	leal	-1(%rdx), %edi
	leal	-2(%rdx), %r10d
	movb	%sil, (%r8,%rdi)
	cmpl	$9999, %r11d
	jbe	.L2000
	movl	%ecx, %r11d
	movl	%ecx, %esi
	imulq	$1374389535, %r11, %rdi
	shrq	$37, %rdi
	imull	$100, %edi, %r11d
	movl	%edi, %eax
	subl	%r11d, %esi
	movl	%ecx, %r11d
	addl	%esi, %esi
	leal	1(%rsi), %edi
	movzbl	(%rsp,%rsi), %esi
	movzbl	(%rsp,%rdi), %edi
	movb	%dil, (%r8,%r10)
	leal	-3(%rdx), %r10d
	subl	$4, %edx
	movb	%sil, (%r8,%r10)
	cmpl	$9999, %ecx
	ja	.L1979
.L2000:
	cmpl	$999, %r11d
	ja	.L1978
	vzeroupper
.L1980:
	addl	$48, %eax
	jmp	.L1981
	.p2align 4,,10
	.p2align 3
.L2005:
	movl	$7, %r10d
	movl	$7, %edi
	jmp	.L1974
	.p2align 4,,10
	.p2align 3
.L1989:
	movl	$8, %r10d
	movl	$8, %edi
	jmp	.L1974
	.p2align 4,,10
	.p2align 3
.L1992:
	movl	$75, %edx
	jmp	.L1971
.L1965:
	testq	%rsi, %rsi
	jle	.L1992
	movl	$1, %r9d
	jmp	.L1980
.L1985:
	movl	$2, %r11d
	movl	$2, %edx
	jmp	.L1966
.L2003:
	movl	$4, %r10d
	movl	$4, %edi
	jmp	.L1974
.L1986:
	movl	$3, %r10d
	movl	$3, %edi
	jmp	.L1974
.L2004:
	movl	$5, %ecx
	jmp	.L1976
.L2006:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE12650:
	.size	_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_, .-_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	.section	.text._ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_,"axG",@progbits,_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_,comdat
	.p2align 4
	.weak	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	.type	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_, @function
_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_:
.LFB12651:
	.cfi_startproc
	endbr64
	movq	%rsi, %r8
	movq	%rdi, %rcx
	movl	%edx, %eax
	subq	%rdi, %r8
	testl	%edx, %edx
	je	.L2025
	bsrl	%edx, %edx
	movl	$2863311531, %edi
	addl	$3, %edx
	imulq	%rdi, %rdx
	shrq	$33, %rdx
	movl	%edx, %r9d
	cmpq	%r8, %r9
	jg	.L2016
	subl	$1, %edx
	cmpl	$63, %eax
	jbe	.L2011
.L2012:
	movl	%eax, %r11d
	movl	%eax, %esi
	movl	%edx, %r10d
	shrl	$6, %eax
	shrl	$3, %r11d
	leal	-1(%rdx), %r8d
	leal	-2(%rdx), %edi
	andl	$7, %esi
	andl	$7, %r11d
	addl	$48, %esi
	addl	$48, %r11d
	movb	%sil, (%rcx,%r10)
	movb	%r11b, (%rcx,%r8)
	cmpl	$63, %eax
	jbe	.L2011
	movl	%eax, %esi
	movl	%eax, %r10d
	leal	-3(%rdx), %r11d
	shrl	$6, %eax
	shrl	$3, %esi
	andl	$7, %r10d
	andl	$7, %esi
	addl	$48, %r10d
	addl	$48, %esi
	movb	%r10b, (%rcx,%rdi)
	leal	-4(%rdx), %edi
	movb	%sil, (%rcx,%r11)
	cmpl	$63, %eax
	jbe	.L2011
	movl	%eax, %r10d
	leal	-5(%rdx), %esi
	leal	-6(%rdx), %r11d
	movl	%eax, %r8d
	shrl	$3, %r10d
	andl	$7, %r8d
	shrl	$6, %eax
	andl	$7, %r10d
	addl	$48, %r8d
	addl	$48, %r10d
	movb	%r8b, (%rcx,%rdi)
	movb	%r10b, (%rcx,%rsi)
	cmpl	$63, %eax
	jbe	.L2011
	movl	%eax, %r10d
	movl	%eax, %edi
	leal	-7(%rdx), %r8d
	shrl	$6, %eax
	shrl	$3, %r10d
	andl	$7, %edi
	subl	$8, %edx
	andl	$7, %r10d
	addl	$48, %edi
	addl	$48, %r10d
	movb	%dil, (%rcx,%r11)
	movb	%r10b, (%rcx,%r8)
	cmpl	$63, %eax
	ja	.L2012
.L2011:
	leaq	(%rcx,%r9), %rsi
	cmpl	$7, %eax
	jbe	.L2024
	movl	%eax, %edx
	shrl	$3, %eax
	andl	$7, %edx
	addl	$48, %edx
	movb	%dl, 1(%rcx)
.L2024:
	addl	$48, %eax
.L2014:
	movb	%al, (%rcx)
	xorl	%edx, %edx
	movq	%rsi, %rax
	ret
	.p2align 4,,10
	.p2align 3
.L2025:
	testq	%r8, %r8
	js	.L2016
	movq	%rdi, %rsi
	movl	$48, %eax
	jmp	.L2014
	.p2align 4,,10
	.p2align 3
.L2016:
	movl	$75, %edx
	movq	%rsi, %rax
	ret
	.cfi_endproc
.LFE12651:
	.size	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_, .-_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm:
.LFB13237:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%r8, %r15
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movq	%rsi, %r13
	addq	%rdx, %rsi
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%r8, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	%rdx, %rbp
	leaq	16(%rbx), %r14
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	movq	8(%rdi), %rax
	movq	%rsi, 24(%rsp)
	movq	%rax, %rdi
	addq	%rax, %rbp
	subq	%rsi, %rdi
	movq	%rdi, 8(%rsp)
	cmpq	(%rbx), %r14
	je	.L2039
	movq	16(%rbx), %rdx
.L2027:
	testq	%rbp, %rbp
	js	.L2053
	cmpq	%rbp, %rdx
	jnb	.L2029
	addq	%rdx, %rdx
	cmpq	%rdx, %rbp
	jnb	.L2029
	testq	%rdx, %rdx
	js	.L2030
	leaq	1(%rdx), %rdi
	movq	%rdx, %rbp
	jmp	.L2031
	.p2align 4,,10
	.p2align 3
.L2029:
	movq	%rbp, %rdi
	addq	$1, %rdi
	js	.L2030
.L2031:
	movq	%rcx, 16(%rsp)
	call	_Znwm@PLT
	testq	%r13, %r13
	movq	16(%rsp), %rcx
	movq	%rax, %r12
	je	.L2032
	movq	(%rbx), %rsi
	cmpq	$1, %r13
	je	.L2054
	movq	%r13, %rdx
	movq	%rax, %rdi
	movq	%rcx, 16(%rsp)
	call	memcpy@PLT
	movq	16(%rsp), %rcx
.L2032:
	testq	%rcx, %rcx
	je	.L2034
	testq	%r15, %r15
	je	.L2034
	leaq	(%r12,%r13), %rdi
	cmpq	$1, %r15
	je	.L2055
	movq	%r15, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
.L2034:
	movq	8(%rsp), %r9
	movq	(%rbx), %r11
	testq	%r9, %r9
	jne	.L2056
.L2036:
	cmpq	%r14, %r11
	je	.L2038
	movq	16(%rbx), %r15
	movq	%r11, %rdi
	leaq	1(%r15), %rsi
	call	_ZdlPvm@PLT
.L2038:
	movq	%r12, (%rbx)
	movq	%rbp, 16(%rbx)
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2056:
	.cfi_restore_state
	movq	24(%rsp), %rsi
	leaq	0(%r13,%r15), %r10
	leaq	(%r12,%r10), %rdi
	addq	%r11, %rsi
	cmpq	$1, %r9
	je	.L2057
	movq	%r9, %rdx
	movq	%r11, 16(%rsp)
	call	memcpy@PLT
	movq	16(%rsp), %r11
	jmp	.L2036
	.p2align 4,,10
	.p2align 3
.L2030:
	call	_ZSt17__throw_bad_allocv@PLT
	.p2align 4,,10
	.p2align 3
.L2039:
	movl	$15, %edx
	jmp	.L2027
	.p2align 4,,10
	.p2align 3
.L2054:
	movzbl	(%rsi), %esi
	movb	%sil, (%rax)
	jmp	.L2032
	.p2align 4,,10
	.p2align 3
.L2055:
	movzbl	(%rcx), %r8d
	movq	8(%rsp), %r9
	movq	(%rbx), %r11
	movb	%r8b, (%rdi)
	testq	%r9, %r9
	je	.L2036
	jmp	.L2056
	.p2align 4,,10
	.p2align 3
.L2057:
	movzbl	(%rsi), %r13d
	movb	%r13b, (%rdi)
	jmp	.L2036
.L2053:
	leaq	.LC12(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE13237:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	.section	.rodata._ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv.str1.1,"aMS",@progbits,1
.LC33:
	.string	"basic_string::append"
	.section	.text._ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv,"axG",@progbits,_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv
	.type	_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv, @function
_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv:
.LFB12678:
	.cfi_startproc
	endbr64
	movq	8(%rdi), %rcx
	movq	24(%rdi), %r8
	subq	%rcx, %r8
	jne	.L2074
	ret
	.p2align 4,,10
	.p2align 3
.L2074:
	movabsq	$9223372036854775807, %rax
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movq	%rdi, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movq	296(%rdi), %rsi
	subq	%rsi, %rax
	cmpq	%r8, %rax
	jb	.L2075
	movq	288(%rdi), %rdi
	leaq	304(%rbx), %rdx
	leaq	(%r8,%rsi), %rbp
	cmpq	%rdx, %rdi
	je	.L2065
	movq	304(%rbx), %r9
.L2061:
	cmpq	%rbp, %r9
	jb	.L2062
	addq	%rsi, %rdi
	cmpq	$1, %r8
	je	.L2076
	movq	%r8, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
.L2064:
	movq	288(%rbx), %rsi
	movq	%rbp, 296(%rbx)
	movb	$0, (%rsi,%rbp)
	movq	8(%rbx), %rdi
	movq	%rdi, 24(%rbx)
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2076:
	.cfi_restore_state
	movzbl	(%rcx), %ecx
	movb	%cl, (%rdi)
	jmp	.L2064
	.p2align 4,,10
	.p2align 3
.L2062:
	leaq	288(%rbx), %rdi
	xorl	%edx, %edx
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	jmp	.L2064
	.p2align 4,,10
	.p2align 3
.L2065:
	movl	$15, %r9d
	jmp	.L2061
.L2075:
	leaq	.LC33(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE12678:
	.size	_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv, .-_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv
	.text
	.align 2
	.p2align 4
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0:
.LFB14061:
	.cfi_startproc
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rsi, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	8(%rdi), %r12
	cmpq	%rsi, %r12
	jb	.L2089
	cmpq	%r12, %rsi
	jnb	.L2086
.L2088:
	movq	0(%rbp), %rsi
	movq	%rbx, 8(%rbp)
	movb	$0, (%rsi,%rbx)
.L2086:
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2089:
	.cfi_restore_state
	movabsq	$9223372036854775807, %rax
	movq	%rsi, %r13
	subq	%r12, %r13
	subq	%r12, %rax
	cmpq	%r13, %rax
	jb	.L2090
	movq	(%rdi), %rdi
	leaq	16(%rbp), %rdx
	cmpq	%rdx, %rdi
	je	.L2085
	movq	16(%rbp), %rcx
.L2080:
	cmpq	%rbx, %rcx
	jb	.L2091
.L2081:
	addq	%r12, %rdi
	cmpq	$1, %r13
	je	.L2092
	xorl	%esi, %esi
	movq	%r13, %rdx
	call	memset@PLT
	movq	0(%rbp), %rsi
	movq	%rbx, 8(%rbp)
	movb	$0, (%rsi,%rbx)
	jmp	.L2086
	.p2align 4,,10
	.p2align 3
.L2092:
	movb	$0, (%rdi)
	jmp	.L2088
	.p2align 4,,10
	.p2align 3
.L2091:
	movq	%rbp, %rdi
	movq	%r13, %r8
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%r12, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	movq	0(%rbp), %rdi
	jmp	.L2081
	.p2align 4,,10
	.p2align 3
.L2085:
	movl	$15, %ecx
	jmp	.L2080
.L2090:
	leaq	.LC11(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE14061:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc:
.LFB13627:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	leaq	16(%rdi), %r15
	movl	%esi, %ecx
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	movq	8(%rdi), %rbp
	movq	(%rdi), %r12
	leaq	1(%rbp), %r13
	cmpq	%r12, %r15
	je	.L2107
	movq	16(%rdi), %r14
	cmpq	%r13, %r14
	jb	.L2108
.L2095:
	movb	%cl, (%r12,%rbp)
	movq	(%rbx), %rsi
	movq	%r13, 8(%rbx)
	movb	$0, 1(%rsi,%rbp)
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2108:
	.cfi_restore_state
	testq	%r13, %r13
	js	.L2109
	addq	%r14, %r14
	cmpq	%r14, %r13
	jb	.L2110
	movq	%rbp, %rdi
	movq	%r13, %r14
	addq	$2, %rdi
	js	.L2099
.L2100:
	movl	%ecx, (%rsp)
	call	_Znwm@PLT
	testq	%rbp, %rbp
	movl	(%rsp), %ecx
	movq	%rax, %r12
	jne	.L2096
	movq	(%rbx), %r8
.L2101:
	cmpq	%r8, %r15
	je	.L2104
	movq	16(%rbx), %rdx
	movq	%r8, %rdi
	movl	%ecx, (%rsp)
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
	movl	(%rsp), %ecx
.L2104:
	movq	%r12, (%rbx)
	movq	%r14, 16(%rbx)
	jmp	.L2095
	.p2align 4,,10
	.p2align 3
.L2107:
	cmpq	$16, %r13
	jne	.L2095
	movl	$31, %edi
	movl	%esi, (%rsp)
	movl	$30, %r14d
	call	_Znwm@PLT
	movl	(%rsp), %ecx
	movq	%rax, %r12
.L2096:
	movq	(%rbx), %r8
	cmpq	$1, %rbp
	je	.L2111
	movq	%r8, %rsi
	movq	%rbp, %rdx
	movq	%r12, %rdi
	movl	%ecx, 12(%rsp)
	movq	%r8, (%rsp)
	call	memcpy@PLT
	movl	12(%rsp), %ecx
	movq	(%rsp), %r8
	jmp	.L2101
	.p2align 4,,10
	.p2align 3
.L2111:
	movzbl	(%r8), %eax
	movb	%al, (%r12)
	jmp	.L2101
	.p2align 4,,10
	.p2align 3
.L2110:
	leaq	1(%r14), %rdi
	testq	%r14, %r14
	jns	.L2100
.L2099:
	call	_ZSt17__throw_bad_allocv@PLT
.L2109:
	leaq	.LC12(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE13627:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm:
.LFB13727:
	.cfi_startproc
	endbr64
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	leaq	16(%rdi), %r12
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	movq	%rsi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	cmpq	(%rdi), %r12
	je	.L2124
	movq	16(%rdi), %rax
.L2113:
	cmpq	%rbp, %rax
	jb	.L2131
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2131:
	.cfi_restore_state
	testq	%rbp, %rbp
	js	.L2132
	addq	%rax, %rax
	cmpq	%rax, %rbp
	jb	.L2133
	movq	%rbp, %rdi
	addq	$1, %rdi
	js	.L2117
.L2118:
	call	_Znwm@PLT
	movq	8(%rbx), %rcx
	movq	(%rbx), %r14
	movq	%rax, %r13
	leaq	1(%rcx), %rdx
	testq	%rcx, %rcx
	je	.L2134
	testq	%rdx, %rdx
	jne	.L2135
.L2130:
	cmpq	%r14, %r12
	je	.L2121
	movq	16(%rbx), %rsi
	movq	%r14, %rdi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L2121:
	movq	%r13, (%rbx)
	movq	%rbp, 16(%rbx)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2135:
	.cfi_restore_state
	movq	%r14, %rsi
	movq	%rax, %rdi
	call	memcpy@PLT
	jmp	.L2130
	.p2align 4,,10
	.p2align 3
.L2133:
	testq	%rax, %rax
	js	.L2117
	leaq	1(%rax), %rdi
	movq	%rax, %rbp
	jmp	.L2118
	.p2align 4,,10
	.p2align 3
.L2124:
	movl	$15, %eax
	jmp	.L2113
	.p2align 4,,10
	.p2align 3
.L2134:
	movzbl	(%r14), %edx
	movb	%dl, (%rax)
	jmp	.L2130
	.p2align 4,,10
	.p2align 3
.L2117:
	call	_ZSt17__throw_bad_allocv@PLT
.L2132:
	leaq	.LC12(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE13727:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	.section	.text._ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE,"axG",@progbits,_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	.type	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE, @function
_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE:
.LFB13758:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movq	%rdx, %r13
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	24(%rdi), %rdi
	movq	16(%rbp), %rbx
	movq	%rdi, %rax
	subq	8(%rbp), %rax
	subq	%rax, %rbx
	cmpq	%rbx, %rsi
	jnb	.L2140
	jmp	.L2137
	.p2align 4,,10
	.p2align 3
.L2148:
	movq	%r13, %rsi
	movq	%rbx, %rdx
	addq	%rbx, %r13
	subq	%rbx, %r12
	call	memcpy@PLT
	addq	%rbx, 24(%rbp)
.L2147:
	movq	0(%rbp), %rdx
	movq	%rbp, %rdi
	call	*(%rdx)
	movq	24(%rbp), %rdi
	movq	16(%rbp), %rbx
	movq	%rdi, %rcx
	subq	8(%rbp), %rcx
	subq	%rcx, %rbx
	cmpq	%rbx, %r12
	jb	.L2137
.L2140:
	testq	%rbx, %rbx
	jne	.L2148
	movq	%rdi, 24(%rbp)
	jmp	.L2147
	.p2align 4,,10
	.p2align 3
.L2137:
	testq	%r12, %r12
	jne	.L2149
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2149:
	.cfi_restore_state
	movq	%r12, %rdx
	movq	%r13, %rsi
	call	memcpy@PLT
	addq	%r12, 24(%rbp)
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE13758:
	.size	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE, .-_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	.section	.text._ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_,"axG",@progbits,_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_,comdat
	.p2align 4
	.weak	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
	.type	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_, @function
_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_:
.LFB13794:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	movq	%rdi, %r14
	pushq	%r13
	pushq	%r12
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%r8, %r12
	pushq	%rbx
	subq	$104, %rsp
	.cfi_offset 3, -56
	movq	%rsi, 16(%rsp)
	movq	%rdx, 8(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 88(%rsp)
	xorl	%eax, %eax
	movb	$0, 48(%rsp)
	cmpl	$3, %ecx
	je	.L2316
	cmpl	$2, %ecx
	je	.L2192
	cmpq	$31, %r8
	ja	.L2317
	testq	%r8, %r8
	jne	.L2318
	cmpq	$0, 16(%rsp)
	jne	.L2314
	.p2align 4,,10
	.p2align 3
.L2160:
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L2319
	addq	$104, %rsp
	movq	%r14, %rax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2192:
	.cfi_restore_state
	movq	%r8, 32(%rsp)
	movq	$0, 40(%rsp)
.L2152:
	cmpq	$31, %r12
	ja	.L2194
	testq	%r12, %r12
	jne	.L2161
	cmpq	$0, 32(%rsp)
	je	.L2320
.L2201:
.L2172:
	jmp	.L2172
	.p2align 4,,10
	.p2align 3
.L2194:
	movl	$32, %r12d
.L2161:
	leaq	48(%rsp), %rbx
	movsbl	%r9b, %r9d
	movl	%r12d, %ecx
	movq	%rbx, 24(%rsp)
	movq	%rbx, %rdx
	cmpl	$8, %r12d
	jnb	.L2321
.L2164:
	andl	$7, %ecx
	je	.L2168
	xorl	%edi, %edi
	cmpl	$1, %ecx
	je	.L2270
	cmpl	$2, %ecx
	je	.L2271
	cmpl	$3, %ecx
	je	.L2272
	cmpl	$4, %ecx
	je	.L2273
	cmpl	$5, %ecx
	je	.L2274
	cmpl	$6, %ecx
	jne	.L2322
.L2275:
	movl	%edi, %r10d
	addl	$1, %edi
	movb	%r9b, (%rdx,%r10)
.L2274:
	movl	%edi, %r11d
	addl	$1, %edi
	movb	%r9b, (%rdx,%r11)
.L2273:
	movl	%edi, %r13d
	addl	$1, %edi
	movb	%r9b, (%rdx,%r13)
.L2272:
	movl	%edi, %r15d
	addl	$1, %edi
	movb	%r9b, (%rdx,%r15)
.L2271:
	movl	%edi, %eax
	addl	$1, %edi
	movb	%r9b, (%rdx,%rax)
.L2270:
	movl	%edi, %esi
	addl	$1, %edi
	movb	%r9b, (%rdx,%rsi)
	cmpl	%ecx, %edi
	jnb	.L2168
.L2167:
	movl	%edi, %r8d
	leal	1(%rdi), %ebx
	leal	2(%rdi), %r10d
	leal	3(%rdi), %r11d
	leal	4(%rdi), %r13d
	movb	%r9b, (%rdx,%r8)
	leal	5(%rdi), %r15d
	leal	6(%rdi), %eax
	movb	%r9b, (%rdx,%rbx)
	leal	7(%rdi), %esi
	addl	$8, %edi
	movb	%r9b, (%rdx,%r10)
	movb	%r9b, (%rdx,%r11)
	movb	%r9b, (%rdx,%r13)
	movb	%r9b, (%rdx,%r15)
	movb	%r9b, (%rdx,%rax)
	movb	%r9b, (%rdx,%rsi)
	cmpl	%ecx, %edi
	jb	.L2167
.L2168:
	movq	32(%rsp), %rcx
	testq	%rcx, %rcx
	je	.L2170
	cmpq	%rcx, %r12
	jnb	.L2171
	testq	%r12, %r12
	je	.L2201
	.p2align 4,,10
	.p2align 3
.L2179:
	movq	24(%r14), %rdi
	movq	16(%r14), %r15
	movq	%r12, %rbx
	movq	24(%rsp), %r13
	movq	%rdi, %r9
	subq	8(%r14), %r9
	subq	%r9, %r15
	cmpq	%r15, %r12
	jnb	.L2177
	jmp	.L2173
	.p2align 4,,10
	.p2align 3
.L2324:
	movq	%r13, %rsi
	movq	%r15, %rdx
	addq	%r15, %r13
	subq	%r15, %rbx
	call	memcpy@PLT
	addq	%r15, 24(%r14)
.L2312:
	movq	(%r14), %rdx
	movq	%r14, %rdi
	call	*(%rdx)
	movq	24(%r14), %rdi
	movq	16(%r14), %r15
	movq	%rdi, %r8
	subq	8(%r14), %r8
	subq	%r8, %r15
	cmpq	%r15, %rbx
	jb	.L2323
.L2177:
	testq	%r15, %r15
	jne	.L2324
	movq	%rdi, 24(%r14)
	jmp	.L2312
	.p2align 4,,10
	.p2align 3
.L2323:
	testq	%rbx, %rbx
	jne	.L2173
.L2178:
	subq	%r12, 32(%rsp)
	movq	32(%rsp), %rdi
	cmpq	%rdi, %r12
	jb	.L2179
	testq	%rdi, %rdi
	jne	.L2171
.L2170:
	cmpq	$0, 16(%rsp)
	jne	.L2157
.L2181:
	cmpq	$0, 40(%rsp)
	je	.L2160
.L2155:
	movq	40(%rsp), %r10
	cmpq	%r10, %r12
	jnb	.L2325
	testq	%r12, %r12
	jne	.L2326
.L2182:
	jmp	.L2182
	.p2align 4,,10
	.p2align 3
.L2173:
	movq	%rbx, %rdx
	movq	%r13, %rsi
	call	memcpy@PLT
	addq	%rbx, 24(%r14)
	jmp	.L2178
	.p2align 4,,10
	.p2align 3
.L2316:
	movq	%r8, %rax
	andl	$1, %r12d
	shrq	%rax
	addq	%rax, %r12
	movq	%rax, 32(%rsp)
	movq	%r12, 40(%rsp)
	jmp	.L2152
	.p2align 4,,10
	.p2align 3
.L2157:
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
	movq	%r14, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L2181
	.p2align 4,,10
	.p2align 3
.L2171:
	movq	32(%rsp), %rsi
	movq	24(%rsp), %rdx
	movq	%r14, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L2170
	.p2align 4,,10
	.p2align 3
.L2326:
	leaq	48(%rsp), %r11
	movq	%r11, 32(%rsp)
	.p2align 4,,10
	.p2align 3
.L2189:
	movq	24(%r14), %rdi
	movq	16(%r14), %rbx
	movq	%r12, %r13
	movq	32(%rsp), %r15
	movq	%rdi, %rax
	subq	8(%r14), %rax
	subq	%rax, %rbx
	cmpq	%rbx, %r12
	jnb	.L2187
	jmp	.L2183
	.p2align 4,,10
	.p2align 3
.L2328:
	movq	%r15, %rsi
	movq	%rbx, %rdx
	addq	%rbx, %r15
	subq	%rbx, %r13
	call	memcpy@PLT
	addq	%rbx, 24(%r14)
.L2313:
	movq	(%r14), %rsi
	movq	%r14, %rdi
	call	*(%rsi)
	movq	24(%r14), %rdi
	movq	16(%r14), %rbx
	movq	%rdi, %rcx
	subq	8(%r14), %rcx
	subq	%rcx, %rbx
	cmpq	%rbx, %r13
	jb	.L2327
.L2187:
	testq	%rbx, %rbx
	jne	.L2328
	movq	%rdi, 24(%r14)
	jmp	.L2313
.L2318:
	leaq	48(%rsp), %rdi
	movsbl	%r9b, %esi
	movq	%r8, %rdx
	movq	%rdi, 32(%rsp)
	call	memset@PLT
	cmpq	$0, 16(%rsp)
	movq	%r12, 40(%rsp)
	jne	.L2157
.L2158:
	movq	40(%rsp), %rsi
	movq	32(%rsp), %rdx
.L2314:
	movq	%r14, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L2160
	.p2align 4,,10
	.p2align 3
.L2327:
	testq	%r13, %r13
	jne	.L2183
.L2188:
	subq	%r12, 40(%rsp)
	movq	40(%rsp), %r9
	cmpq	%r9, %r12
	jb	.L2189
	testq	%r9, %r9
	je	.L2160
	jmp	.L2158
	.p2align 4,,10
	.p2align 3
.L2183:
	movq	%r13, %rdx
	movq	%r15, %rsi
	call	memcpy@PLT
	addq	%r13, 24(%r14)
	jmp	.L2188
.L2321:
	movzbl	%r9b, %edx
	movl	%r12d, %edi
	movl	$8, %r10d
	movabsq	$72340172838076673, %rsi
	imulq	%rsi, %rdx
	andl	$-8, %edi
	leal	-1(%rdi), %r8d
	shrl	$3, %r8d
	movq	%rdx, 48(%rsp)
	andl	$7, %r8d
	cmpl	%edi, %r10d
	jnb	.L2307
	testl	%r8d, %r8d
	je	.L2165
	cmpl	$1, %r8d
	je	.L2276
	cmpl	$2, %r8d
	je	.L2277
	cmpl	$3, %r8d
	je	.L2278
	cmpl	$4, %r8d
	je	.L2279
	cmpl	$5, %r8d
	je	.L2280
	cmpl	$6, %r8d
	je	.L2281
	movq	%rdx, 8(%rbx)
	movl	$16, %r10d
.L2281:
	movq	24(%rsp), %r13
	movl	%r10d, %r11d
	addl	$8, %r10d
	movq	%rdx, 0(%r13,%r11)
.L2280:
	movq	24(%rsp), %rax
	movl	%r10d, %r15d
	addl	$8, %r10d
	movq	%rdx, (%rax,%r15)
.L2279:
	movq	24(%rsp), %r8
	movl	%r10d, %esi
	addl	$8, %r10d
	movq	%rdx, (%r8,%rsi)
.L2278:
	movq	24(%rsp), %rbx
	movl	%r10d, %r11d
	addl	$8, %r10d
	movq	%rdx, (%rbx,%r11)
.L2277:
	movq	24(%rsp), %r15
	movl	%r10d, %r13d
	addl	$8, %r10d
	movq	%rdx, (%r15,%r13)
.L2276:
	movq	24(%rsp), %rsi
	movl	%r10d, %eax
	addl	$8, %r10d
	movq	%rdx, (%rsi,%rax)
	cmpl	%edi, %r10d
	jnb	.L2307
.L2165:
	movq	24(%rsp), %r11
	movl	%r10d, %r8d
	leal	8(%r10), %ebx
	leal	16(%r10), %r13d
	leal	24(%r10), %r15d
	leal	32(%r10), %eax
	movq	%rdx, (%r11,%r8)
	leal	40(%r10), %esi
	leal	48(%r10), %r8d
	movq	%rdx, (%r11,%rbx)
	leal	56(%r10), %ebx
	addl	$64, %r10d
	movq	%rdx, (%r11,%r13)
	movq	%rdx, (%r11,%r15)
	movq	%rdx, (%r11,%rax)
	movq	%rdx, (%r11,%rsi)
	movq	%rdx, (%r11,%r8)
	movq	%rdx, (%r11,%rbx)
	cmpl	%edi, %r10d
	jb	.L2165
.L2307:
	movq	24(%rsp), %rdx
	addq	%r10, %rdx
	jmp	.L2164
.L2317:
	vmovd	%r9d, %xmm0
	cmpq	$0, 16(%rsp)
	vpbroadcastb	%xmm0, %ymm1
	vmovdqu	%ymm1, 48(%rsp)
	jne	.L2154
	vzeroupper
.L2190:
	movq	%r12, 40(%rsp)
	movl	$32, %r12d
	jmp	.L2155
.L2322:
	movb	%r9b, (%rdx)
	movl	$1, %edi
	jmp	.L2275
.L2320:
	cmpq	$0, 16(%rsp)
	jne	.L2157
	cmpq	$0, 40(%rsp)
	je	.L2160
	jmp	.L2182
	.p2align 4,,10
	.p2align 3
.L2154:
	vzeroupper
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L2190
.L2325:
	leaq	48(%rsp), %r12
	movq	%r12, 32(%rsp)
	jmp	.L2158
.L2319:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13794:
	.size	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_, .-_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
	.section	.text._ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE,"axG",@progbits,_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE,comdat
	.p2align 4
	.weak	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE
	.type	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE, @function
_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE:
.LFB13686:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsi, %r10
	movq	%rdx, %rsi
	movq	%rcx, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movzwl	(%r8), %eax
	andw	$384, %ax
	cmpw	$128, %ax
	je	.L2358
	cmpw	$256, %ax
	je	.L2332
.L2347:
	movq	16(%rdx), %rbx
	testq	%rdi, %rdi
	jne	.L2359
	movq	%rbx, %rax
	movq	-8(%rbp), %rbx
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2358:
	.cfi_restore_state
	movzwl	2(%r8), %ebx
.L2331:
	cmpq	%rbx, %rsi
	jnb	.L2347
	movzbl	(%r8), %r11d
	movq	16(%rdx), %rax
	movq	%r10, %rdx
	movl	%r11d, %ecx
	andl	$3, %ecx
	andl	$3, %r11d
	cmove	%r9d, %ecx
	subq	%rsi, %rbx
	movsbl	6(%r8), %r9d
	movq	%rdi, %rsi
	movq	%rbx, %r8
	movq	-8(%rbp), %rbx
	movq	%rax, %rdi
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	jmp	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
	.p2align 4,,10
	.p2align 3
.L2359:
	.cfi_restore_state
	movq	%rdi, %rsi
	movq	%r10, %rdx
	movq	%rbx, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	movq	%rbx, %rax
	movq	-8(%rbp), %rbx
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2332:
	.cfi_restore_state
	movzbl	(%rcx), %ebx
	movzwl	2(%r8), %r11d
	movl	%ebx, %ecx
	andl	$15, %ebx
	andl	$15, %ecx
	cmpq	%rbx, %r11
	jb	.L2360
	testb	%cl, %cl
	jne	.L2337
	movq	(%rdx), %rax
	shrq	$4, %rax
	cmpq	%rax, %r11
	jb	.L2361
.L2337:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L2360:
	movq	(%rdx), %rbx
	leaq	(%r11,%r11,4), %rcx
	salq	$4, %r11
	addq	8(%rdx), %r11
	vmovdqa	(%r11), %xmm1
	shrq	$4, %rbx
	shrq	%cl, %rbx
	vmovdqa	%xmm1, 32(%rsp)
	andl	$31, %ebx
.L2336:
	movb	%bl, 48(%rsp)
	movzbl	%bl, %r11d
	leaq	.L2340(%rip), %rbx
	vmovdqu	32(%rsp), %ymm0
	movslq	(%rbx,%r11,4), %rax
	vmovdqu	%ymm0, (%rsp)
	addq	%rbx, %rax
	notrack jmp	*%rax
	.section	.rodata._ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE,"aG",@progbits,_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE,comdat
	.align 4
	.align 4
.L2340:
	.long	.L2355-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2344-.L2340
	.long	.L2343-.L2340
	.long	.L2342-.L2340
	.long	.L2341-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.long	.L2345-.L2340
	.section	.text._ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE,"axG",@progbits,_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE,comdat
.L2342:
	movq	(%rsp), %rbx
	testq	%rbx, %rbx
	js	.L2345
.L2357:
	vzeroupper
	jmp	.L2331
.L2343:
	movl	(%rsp), %ebx
	vzeroupper
	jmp	.L2331
.L2344:
	movslq	(%rsp), %rbx
	testl	%ebx, %ebx
	jns	.L2357
.L2345:
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
	.p2align 4,,10
	.p2align 3
.L2341:
	movq	(%rsp), %rbx
	jmp	.L2357
.L2355:
	vzeroupper
	jmp	.L2337
	.p2align 4,,10
	.p2align 3
.L2361:
	salq	$5, %r11
	addq	8(%rdx), %r11
	vmovdqu	(%r11), %xmm2
	movzbl	16(%r11), %ebx
	vmovdqa	%xmm2, 32(%rsp)
	jmp	.L2336
	.cfi_endproc
.LFE13686:
	.size	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE, .-_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE
	.section	.text.unlikely
.LCOLDB34:
	.text
.LHOTB34:
	.p2align 4
	.type	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0, @function
_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0:
.LFB14064:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %r10
	movq	%rsi, %rdi
	movq	%rdx, %rsi
	movq	%rcx, %rax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movzwl	(%rcx), %edx
	andw	$384, %dx
	cmpw	$128, %dx
	je	.L2386
	cmpw	$256, %dx
	je	.L2387
.L2379:
	movq	16(%rsi), %rbx
	movq	%rdi, %rdx
	movl	$1, %esi
	movq	%rbx, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	movq	%rbx, %rax
	movq	-8(%rbp), %rbx
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2386:
	.cfi_restore_state
	movzwl	2(%rcx), %edx
.L2364:
	cmpq	$1, %rdx
	jbe	.L2379
	movzbl	(%rax), %r9d
	movq	16(%rsi), %rbx
	leaq	-1(%rdx), %r8
	movq	%rdi, %rdx
	movl	$1, %r11d
	movq	%r10, %rsi
	movl	%r9d, %ecx
	movq	%rbx, %rdi
	movq	-8(%rbp), %rbx
	andl	$3, %ecx
	andl	$3, %r9d
	movsbl	6(%rax), %r9d
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	cmove	%r11d, %ecx
	jmp	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
	.p2align 4,,10
	.p2align 3
.L2387:
	.cfi_restore_state
	movzbl	(%rsi), %ebx
	movzwl	2(%rcx), %r8d
	movl	%ebx, %ecx
	andl	$15, %ebx
	andl	$15, %ecx
	cmpq	%rbx, %r8
	jb	.L2388
	testb	%cl, %cl
	jne	.L2369
	movq	(%rsi), %r9
	shrq	$4, %r9
	cmpq	%r9, %r8
	jnb	.L2369
	salq	$5, %r8
	addq	8(%rsi), %r8
	vmovdqu	(%r8), %xmm2
	movzbl	16(%r8), %r11d
	vmovdqa	%xmm2, 32(%rsp)
.L2368:
	leaq	.L2372(%rip), %r8
	movzbl	%r11b, %edx
	movb	%r11b, 48(%rsp)
	vmovdqu	32(%rsp), %ymm0
	movslq	(%r8,%rdx,4), %rbx
	vmovdqu	%ymm0, (%rsp)
	addq	%r8, %rbx
	notrack jmp	*%rbx
	.section	.rodata
	.align 4
	.align 4
.L2372:
	.long	.L2383-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2376-.L2372
	.long	.L2375-.L2372
	.long	.L2374-.L2372
	.long	.L2373-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.long	.L2377-.L2372
	.text
.L2374:
	movq	(%rsp), %rdx
	testq	%rdx, %rdx
	js	.L2377
.L2385:
	vzeroupper
	jmp	.L2364
.L2375:
	movl	(%rsp), %edx
	vzeroupper
	jmp	.L2364
.L2376:
	movslq	(%rsp), %rdx
	testl	%edx, %edx
	jns	.L2385
	jmp	.L2377
	.p2align 4,,10
	.p2align 3
.L2373:
	movq	(%rsp), %rdx
	jmp	.L2385
.L2383:
	vzeroupper
	jmp	.L2369
	.p2align 4,,10
	.p2align 3
.L2388:
	movq	(%rsi), %r11
	leaq	(%r8,%r8,4), %rcx
	salq	$4, %r8
	addq	8(%rsi), %r8
	vmovdqa	(%r8), %xmm1
	shrq	$4, %r11
	shrq	%cl, %r11
	vmovdqa	%xmm1, 32(%rsp)
	andl	$31, %r11d
	jmp	.L2368
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0.cold, @function
_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0.cold:
.LFSB14064:
.L2377:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -24
	.cfi_offset 6, -16
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L2369:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.cfi_endproc
.LFE14064:
	.text
	.size	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0, .-_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
	.section	.text.unlikely
	.size	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0.cold, .-_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0.cold
.LCOLDE34:
	.text
.LHOTE34:
	.section	.text._ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE,"axG",@progbits,_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE
	.type	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE, @function
_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE:
.LFB13685:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	.cfi_offset 13, -24
	movq	%rcx, %r13
	pushq	%r12
	pushq	%rbx
	subq	$88, %rsp
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movzwl	(%rdi), %eax
	movzbl	1(%rdi), %ecx
	testw	$1920, %ax
	jne	.L2390
	movq	16(%r13), %r13
	testq	%rsi, %rsi
	jne	.L2428
.L2392:
	addq	$88, %rsp
	movq	%r13, %rax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2390:
	.cfi_restore_state
	movq	%rdi, %rbx
	movq	%rsi, %r12
	andl	$6, %ecx
	je	.L2393
	cmpb	$2, %cl
	je	.L2429
	cmpb	$4, %cl
	jne	.L2393
	movzbl	0(%r13), %ecx
	movzwl	4(%rdi), %r8d
	movl	%ecx, %edi
	andl	$15, %ecx
	andl	$15, %edi
	cmpq	%rcx, %r8
	jb	.L2430
	testb	%dil, %dil
	je	.L2431
.L2398:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L2429:
	movzwl	4(%rdi), %r12d
.L2395:
	cmpq	%r12, %rsi
	cmovbe	%rsi, %r12
.L2393:
	andw	$384, %ax
	cmpw	$128, %ax
	je	.L2432
	cmpw	$256, %ax
	je	.L2410
.L2413:
	movq	16(%r13), %r13
	testq	%r12, %r12
	je	.L2392
	movq	%r12, %rsi
.L2428:
	movq	%r13, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	addq	$88, %rsp
	movq	%r13, %rax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2432:
	.cfi_restore_state
	movzwl	2(%rbx), %r8d
.L2409:
	cmpq	%r8, %r12
	jnb	.L2413
	movzbl	(%rbx), %esi
	movl	$1, %eax
	movq	16(%r13), %rdi
	movsbl	6(%rbx), %r9d
	movl	%esi, %ecx
	andl	$3, %ecx
	testb	$3, %sil
	movq	%r12, %rsi
	cmove	%eax, %ecx
	subq	%r12, %r8
	call	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
	addq	$88, %rsp
	movq	%rax, %r13
	popq	%rbx
	popq	%r12
	movq	%r13, %rax
	popq	%r13
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2410:
	.cfi_restore_state
	movzwl	2(%rbx), %edi
	movq	%r13, %rsi
	movq	%rdx, 8(%rsp)
	call	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
	movq	8(%rsp), %rdx
	movq	%rax, %r8
	jmp	.L2409
	.p2align 4,,10
	.p2align 3
.L2430:
	movq	0(%r13), %r11
	leaq	(%r8,%r8,4), %rcx
	salq	$4, %r8
	addq	8(%r13), %r8
	vmovdqa	(%r8), %xmm1
	shrq	$4, %r11
	shrq	%cl, %r11
	vmovdqa	%xmm1, 48(%rsp)
	movq	%r11, %r10
	andl	$31, %r10d
.L2397:
	leaq	.L2401(%rip), %r8
	movzbl	%r10b, %r12d
	movb	%r10b, 64(%rsp)
	vmovdqu	48(%rsp), %ymm0
	movslq	(%r8,%r12,4), %rdi
	vmovdqu	%ymm0, 16(%rsp)
	addq	%r8, %rdi
	notrack jmp	*%rdi
	.section	.rodata._ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE,"aG",@progbits,_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE,comdat
	.align 4
	.align 4
.L2401:
	.long	.L2425-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2405-.L2401
	.long	.L2404-.L2401
	.long	.L2403-.L2401
	.long	.L2402-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.long	.L2406-.L2401
	.section	.text._ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE,"axG",@progbits,_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE,comdat
.L2403:
	movq	16(%rsp), %r12
	testq	%r12, %r12
	js	.L2406
.L2427:
	vzeroupper
	jmp	.L2395
.L2404:
	movl	16(%rsp), %r12d
	vzeroupper
	jmp	.L2395
.L2405:
	movslq	16(%rsp), %r12
	testl	%r12d, %r12d
	jns	.L2427
.L2406:
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
	.p2align 4,,10
	.p2align 3
.L2402:
	movq	16(%rsp), %r12
	jmp	.L2427
.L2425:
	vzeroupper
	jmp	.L2398
	.p2align 4,,10
	.p2align 3
.L2431:
	movq	0(%r13), %r9
	shrq	$4, %r9
	cmpq	%r9, %r8
	jnb	.L2398
	salq	$5, %r8
	addq	8(%r13), %r8
	vmovdqu	(%r8), %xmm2
	movzbl	16(%r8), %r10d
	vmovdqa	%xmm2, 48(%rsp)
	jmp	.L2397
	.cfi_endproc
.LFE13685:
	.size	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE, .-_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE
	.section	.text._ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_,"axG",@progbits,_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_,comdat
	.p2align 4
	.weak	_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_
	.type	_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_, @function
_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_:
.LFB13857:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movl	%esi, %r12d
	pushq	%rbx
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	movq	%rdx, %rdi
	movq	%rcx, %rdx
	andq	$-32, %rsp
	movsbq	(%rdi), %rax
	leal	-1(%rax), %ecx
	cmpb	$125, %cl
	ja	.L2434
	movq	%r9, %rsi
	subq	%r8, %rsi
	cmpq	%rax, %rsi
	jle	.L2434
	subq	$1, %rdx
	xorl	%r11d, %r11d
	movq	%rdx, %r10
	andl	$3, %r10d
	je	.L2436
	subq	%rax, %r9
	cmpq	%rdx, %r11
	jb	.L2820
.L2435:
	movq	%r9, %r15
	movl	$1, %edx
	subq	%r8, %r15
	cmpq	%rax, %r15
	jle	.L2793
.L2439:
	subq	%rax, %r9
	leaq	1(%rdx), %rcx
	movq	%r9, %rsi
	movq	%rcx, %rdx
	subq	%r8, %rsi
	cmpq	%rsi, %rax
	jge	.L2793
	subq	%rax, %r9
	addq	$1, %rdx
	movq	%r9, %r13
	subq	%r8, %r13
	cmpq	%r13, %rax
	jge	.L2793
	subq	%rax, %r9
	leaq	2(%rcx), %rdx
	movq	%r9, %r10
	subq	%r8, %r10
	cmpq	%r10, %rax
	jge	.L2793
	subq	%rax, %r9
	leaq	3(%rcx), %rdx
	movq	%r9, %r14
	subq	%r8, %r14
	cmpq	%r14, %rax
	jge	.L2793
	subq	%rax, %r9
	leaq	4(%rcx), %rdx
	movq	%r9, %r15
	subq	%r8, %r15
	cmpq	%r15, %rax
	jge	.L2793
	subq	%rax, %r9
	leaq	5(%rcx), %rdx
	movq	%r9, %rsi
	subq	%r8, %rsi
	cmpq	%rsi, %rax
	jge	.L2793
	subq	%rax, %r9
	leaq	6(%rcx), %rdx
	movq	%r9, %r13
	subq	%r8, %r13
	cmpq	%r13, %rax
	jge	.L2793
	subq	%rax, %r9
	leaq	7(%rcx), %rdx
	movq	%r9, %rcx
	subq	%r8, %rcx
	cmpq	%rcx, %rax
	jl	.L2439
.L2793:
	leaq	(%rdi,%r11), %r13
	jmp	.L2438
.L2820:
	movsbq	1(%rdi), %rax
	leaq	1(%rdi), %r13
	movl	$1, %r11d
	leal	-1(%rax), %r14d
	cmpb	$125, %r14b
	ja	.L2483
	movq	%r9, %r15
	subq	%r8, %r15
	cmpq	%rax, %r15
	jle	.L2483
	cmpq	$1, %r10
	je	.L2436
	cmpq	$2, %r10
	je	.L2708
	leaq	2(%rdi), %r13
	subq	%rax, %r9
	movl	$2, %r11d
	movsbq	0(%r13), %rax
	leal	-1(%rax), %ecx
	cmpb	$125, %cl
	ja	.L2483
	movq	%r9, %rsi
	subq	%r8, %rsi
	cmpq	%rax, %rsi
	jle	.L2483
.L2708:
	subq	%rax, %r9
	addq	$1, %r11
	jmp	.L2819
	.p2align 4,,10
	.p2align 3
.L2436:
	subq	%rax, %r9
	cmpq	%rdx, %r11
	jnb	.L2435
	leaq	1(%r11), %r10
	leaq	(%rdi,%r10), %r13
	movq	%r10, %r11
	movsbq	0(%r13), %rax
	leal	-1(%rax), %r14d
	cmpb	$125, %r14b
	ja	.L2483
	movq	%r9, %r15
	subq	%r8, %r15
	cmpq	%rax, %r15
	jle	.L2483
	addq	$1, %r11
	subq	%rax, %r9
	leaq	(%rdi,%r11), %r13
	movsbq	0(%r13), %rcx
	leal	-1(%rcx), %esi
	cmpb	$125, %sil
	ja	.L2483
	movq	%r9, %rax
	subq	%r8, %rax
	cmpq	%rcx, %rax
	jle	.L2483
	leaq	2(%r10), %r11
	subq	%rcx, %r9
	leaq	(%rdi,%r11), %r13
	movsbq	0(%r13), %r14
	leal	-1(%r14), %r15d
	cmpb	$125, %r15b
	ja	.L2483
	movq	%r9, %rcx
	subq	%r8, %rcx
	cmpq	%r14, %rcx
	jle	.L2483
	subq	%r14, %r9
	leaq	3(%r10), %r11
.L2819:
	leaq	(%rdi,%r11), %r13
	movsbq	0(%r13), %rax
	leal	-1(%rax), %r10d
	cmpb	$125, %r10b
	ja	.L2483
	movq	%r9, %r14
	subq	%r8, %r14
	cmpq	%rax, %r14
	jg	.L2436
	.p2align 4,,10
	.p2align 3
.L2483:
	xorl	%edx, %edx
.L2438:
	leaq	-1(%rdx), %r10
	leaq	-1(%r11), %rsi
	movq	%r10, -8(%rsp)
	cmpq	%r9, %r8
	je	.L2821
.L2472:
	movq	%r9, %rax
	subq	%r8, %rax
	leaq	-1(%rax), %rcx
	cmpq	$14, %rcx
	jbe	.L2475
	leaq	1(%r8), %r14
	movq	%rbx, %r15
	subq	%r14, %r15
	xorl	%r14d, %r14d
	cmpq	$30, %r15
	ja	.L2822
.L2676:
	movq	%rax, %r15
	andl	$7, %r15d
	je	.L2801
	cmpq	$1, %r15
	je	.L2683
	cmpq	$2, %r15
	je	.L2684
	cmpq	$3, %r15
	je	.L2685
	cmpq	$4, %r15
	je	.L2686
	cmpq	$5, %r15
	je	.L2687
	cmpq	$6, %r15
	jne	.L2823
.L2688:
	movzbl	(%r8,%r14), %r10d
	movb	%r10b, (%rbx,%r14)
	addq	$1, %r14
.L2687:
	movzbl	(%r8,%r14), %ecx
	movb	%cl, (%rbx,%r14)
	addq	$1, %r14
.L2686:
	movzbl	(%r8,%r14), %r15d
	movb	%r15b, (%rbx,%r14)
	addq	$1, %r14
.L2685:
	movzbl	(%r8,%r14), %r10d
	movb	%r10b, (%rbx,%r14)
	addq	$1, %r14
.L2684:
	movzbl	(%r8,%r14), %ecx
	movb	%cl, (%rbx,%r14)
	addq	$1, %r14
.L2683:
	movzbl	(%r8,%r14), %r15d
	movb	%r15b, (%rbx,%r14)
	addq	$1, %r14
	cmpq	%rax, %r14
	je	.L2449
.L2801:
	movq	-8(%rsp), %r10
.L2448:
	movzbl	(%r8,%r14), %ecx
	movb	%cl, (%rbx,%r14)
	movzbl	1(%r8,%r14), %r15d
	movb	%r15b, 1(%rbx,%r14)
	movzbl	2(%r8,%r14), %ecx
	movb	%cl, 2(%rbx,%r14)
	movzbl	3(%r8,%r14), %r15d
	movb	%r15b, 3(%rbx,%r14)
	movzbl	4(%r8,%r14), %ecx
	movb	%cl, 4(%rbx,%r14)
	movzbl	5(%r8,%r14), %r15d
	movb	%r15b, 5(%rbx,%r14)
	movzbl	6(%r8,%r14), %ecx
	movb	%cl, 6(%rbx,%r14)
	movzbl	7(%r8,%r14), %r15d
	movb	%r15b, 7(%rbx,%r14)
	addq	$8, %r14
	cmpq	%rax, %r14
	jne	.L2448
	movq	%r10, -8(%rsp)
.L2449:
	addq	%rbx, %rax
.L2440:
	testq	%rdx, %rdx
	je	.L2450
	movq	-8(%rsp), %r10
	movq	%r11, -16(%rsp)
	movq	%rsi, -8(%rsp)
	.p2align 4,,10
	.p2align 3
.L2460:
	movb	%r12b, (%rax)
	movzbl	0(%r13), %edx
	leaq	1(%rax), %r14
	testb	%dl, %dl
	jle	.L2478
	leal	-1(%rdx), %r11d
	cmpb	$14, %r11b
	jbe	.L2452
	movq	%rax, %rsi
	subq	%r9, %rsi
	cmpq	$30, %rsi
	ja	.L2824
.L2452:
	movq	%rdx, %r15
	movzbl	%dl, %r8d
	xorl	%ebx, %ebx
	andl	$7, %r15d
	je	.L2458
	cmpq	$1, %r15
	je	.L2689
	cmpq	$2, %r15
	je	.L2690
	cmpq	$3, %r15
	je	.L2691
	cmpq	$4, %r15
	je	.L2692
	cmpq	$5, %r15
	je	.L2693
	cmpq	$6, %r15
	je	.L2694
	movzbl	(%r9), %ecx
	movl	$1, %ebx
	movb	%cl, 1(%rax)
.L2694:
	movzbl	(%r9,%rbx), %r11d
	movb	%r11b, 1(%rax,%rbx)
	addq	$1, %rbx
.L2693:
	movzbl	(%r9,%rbx), %esi
	movb	%sil, 1(%rax,%rbx)
	addq	$1, %rbx
.L2692:
	movzbl	(%r9,%rbx), %r15d
	movb	%r15b, 1(%rax,%rbx)
	addq	$1, %rbx
.L2691:
	movzbl	(%r9,%rbx), %ecx
	movb	%cl, 1(%rax,%rbx)
	addq	$1, %rbx
.L2690:
	movzbl	(%r9,%rbx), %r11d
	movb	%r11b, 1(%rax,%rbx)
	addq	$1, %rbx
.L2689:
	movzbl	(%r9,%rbx), %esi
	movb	%sil, 1(%rax,%rbx)
	addq	$1, %rbx
	cmpq	%rbx, %r8
	je	.L2459
.L2458:
	movzbl	(%r9,%rbx), %r15d
	movb	%r15b, 1(%rax,%rbx)
	movzbl	1(%r9,%rbx), %ecx
	movb	%cl, 2(%rax,%rbx)
	movzbl	2(%r9,%rbx), %r11d
	movb	%r11b, 3(%rax,%rbx)
	movzbl	3(%r9,%rbx), %esi
	movb	%sil, 4(%rax,%rbx)
	movzbl	4(%r9,%rbx), %r15d
	movb	%r15b, 5(%rax,%rbx)
	movzbl	5(%r9,%rbx), %ecx
	movb	%cl, 6(%rax,%rbx)
	movzbl	6(%r9,%rbx), %r11d
	movb	%r11b, 7(%rax,%rbx)
	movzbl	7(%r9,%rbx), %esi
	movb	%sil, 8(%rax,%rbx)
	addq	$8, %rbx
	cmpq	%rbx, %r8
	jne	.L2458
	.p2align 4,,10
	.p2align 3
.L2459:
	movsbq	%dl, %rax
	addq	%rax, %r9
	addq	%r14, %rax
.L2451:
	subq	$1, %r10
	jnb	.L2460
	movq	-16(%rsp), %r11
	movq	-8(%rsp), %rsi
.L2450:
	testq	%r11, %r11
	je	.L2816
	.p2align 4,,10
	.p2align 3
.L2471:
	movb	%r12b, (%rax)
	movzbl	(%rdi,%rsi), %r13d
	leaq	1(%rax), %rbx
	testb	%r13b, %r13b
	jle	.L2480
	leal	-1(%r13), %r10d
	cmpb	$14, %r10b
	jbe	.L2463
	movq	%rax, %r14
	subq	%r9, %r14
	cmpq	$30, %r14
	ja	.L2825
.L2463:
	movq	%r13, %r15
	movzbl	%r13b, %r8d
	xorl	%r14d, %r14d
	andl	$7, %r15d
	je	.L2469
	cmpq	$1, %r15
	je	.L2695
	cmpq	$2, %r15
	je	.L2696
	cmpq	$3, %r15
	je	.L2697
	cmpq	$4, %r15
	je	.L2698
	cmpq	$5, %r15
	je	.L2699
	cmpq	$6, %r15
	je	.L2700
	movzbl	(%r9), %edx
	movl	$1, %r14d
	movb	%dl, 1(%rax)
.L2700:
	movzbl	(%r9,%r14), %r11d
	movb	%r11b, 1(%rax,%r14)
	addq	$1, %r14
.L2699:
	movzbl	(%r9,%r14), %r10d
	movb	%r10b, 1(%rax,%r14)
	addq	$1, %r14
.L2698:
	movzbl	(%r9,%r14), %ecx
	movb	%cl, 1(%rax,%r14)
	addq	$1, %r14
.L2697:
	movzbl	(%r9,%r14), %r15d
	movb	%r15b, 1(%rax,%r14)
	addq	$1, %r14
.L2696:
	movzbl	(%r9,%r14), %edx
	movb	%dl, 1(%rax,%r14)
	addq	$1, %r14
.L2695:
	movzbl	(%r9,%r14), %r11d
	movb	%r11b, 1(%rax,%r14)
	addq	$1, %r14
	cmpq	%r14, %r8
	je	.L2470
.L2469:
	movzbl	(%r9,%r14), %r10d
	movb	%r10b, 1(%rax,%r14)
	movzbl	1(%r9,%r14), %ecx
	movb	%cl, 2(%rax,%r14)
	movzbl	2(%r9,%r14), %r15d
	movb	%r15b, 3(%rax,%r14)
	movzbl	3(%r9,%r14), %edx
	movb	%dl, 4(%rax,%r14)
	movzbl	4(%r9,%r14), %r11d
	movb	%r11b, 5(%rax,%r14)
	movzbl	5(%r9,%r14), %r10d
	movb	%r10b, 6(%rax,%r14)
	movzbl	6(%r9,%r14), %ecx
	movb	%cl, 7(%rax,%r14)
	movzbl	7(%r9,%r14), %r15d
	movb	%r15b, 8(%rax,%r14)
	addq	$8, %r14
	cmpq	%r14, %r8
	jne	.L2469
	.p2align 4,,10
	.p2align 3
.L2470:
	movsbq	%r13b, %rax
	addq	%rax, %r9
	addq	%rbx, %rax
.L2462:
	subq	$1, %rsi
	jnb	.L2471
.L2816:
	vzeroupper
.L2433:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2824:
	.cfi_restore_state
	movl	%edx, %ebx
	cmpb	$30, %r11b
	jle	.L2479
	vmovdqu	(%r9), %ymm7
	shrb	$5, %bl
	vmovdqu	%ymm7, 1(%rax)
	cmpb	$1, %bl
	je	.L2454
	vmovdqu	32(%r9), %ymm0
	vmovdqu	%ymm0, 33(%rax)
	cmpb	$3, %bl
	jne	.L2454
	vmovdqu	64(%r9), %ymm5
	vmovdqu	%ymm5, 65(%rax)
.L2454:
	movq	%rdx, %rcx
	movl	%edx, %r11d
	movl	%edx, %esi
	andl	$224, %ecx
	andl	$-32, %r11d
	andl	$31, %esi
	leaq	(%r14,%rcx), %r8
	addq	%r9, %rcx
	testb	$31, %dl
	je	.L2459
	leal	-1(%rsi), %r15d
	movl	%esi, %ebx
	cmpb	$14, %r15b
	jbe	.L2456
.L2453:
	movzbl	%r11b, %r11d
	movl	%ebx, %r15d
	vmovdqu	(%r9,%r11), %xmm1
	andl	$-16, %r15d
	subl	%r15d, %esi
	vmovdqu	%xmm1, 1(%rax,%r11)
	movq	%rbx, %rax
	andl	$240, %eax
	addq	%rax, %r8
	addq	%rax, %rcx
	andl	$15, %ebx
	je	.L2459
.L2456:
	movzbl	(%rcx), %ebx
	movb	%bl, (%r8)
	cmpb	$1, %sil
	je	.L2459
	movzbl	1(%rcx), %r11d
	movb	%r11b, 1(%r8)
	cmpb	$2, %sil
	je	.L2459
	movzbl	2(%rcx), %r15d
	movb	%r15b, 2(%r8)
	cmpb	$3, %sil
	je	.L2459
	movzbl	3(%rcx), %eax
	movb	%al, 3(%r8)
	cmpb	$4, %sil
	je	.L2459
	movzbl	4(%rcx), %ebx
	movb	%bl, 4(%r8)
	cmpb	$5, %sil
	je	.L2459
	movzbl	5(%rcx), %r11d
	movb	%r11b, 5(%r8)
	cmpb	$6, %sil
	je	.L2459
	movzbl	6(%rcx), %r15d
	movb	%r15b, 6(%r8)
	cmpb	$7, %sil
	je	.L2459
	movzbl	7(%rcx), %eax
	movb	%al, 7(%r8)
	cmpb	$8, %sil
	je	.L2459
	movzbl	8(%rcx), %ebx
	movb	%bl, 8(%r8)
	cmpb	$9, %sil
	je	.L2459
	movzbl	9(%rcx), %r11d
	movb	%r11b, 9(%r8)
	cmpb	$10, %sil
	je	.L2459
	movzbl	10(%rcx), %r15d
	movb	%r15b, 10(%r8)
	cmpb	$11, %sil
	je	.L2459
	movzbl	11(%rcx), %eax
	movb	%al, 11(%r8)
	cmpb	$12, %sil
	je	.L2459
	movzbl	12(%rcx), %ebx
	movb	%bl, 12(%r8)
	cmpb	$13, %sil
	je	.L2459
	movzbl	13(%rcx), %r11d
	movb	%r11b, 13(%r8)
	cmpb	$14, %sil
	je	.L2459
	movzbl	14(%rcx), %esi
	movb	%sil, 14(%r8)
	jmp	.L2459
	.p2align 4,,10
	.p2align 3
.L2825:
	movl	%r13d, %r15d
	cmpb	$30, %r10b
	jle	.L2481
	vmovdqu	(%r9), %ymm6
	movl	%r13d, %r10d
	shrb	$5, %r10b
	vmovdqu	%ymm6, 1(%rax)
	cmpb	$1, %r10b
	je	.L2465
	vmovdqu	32(%r9), %ymm4
	vmovdqu	%ymm4, 33(%rax)
	cmpb	$3, %r10b
	jne	.L2465
	vmovdqu	64(%r9), %ymm2
	vmovdqu	%ymm2, 65(%rax)
.L2465:
	movq	%r13, %rdx
	movl	%r13d, %r11d
	movl	%r13d, %ecx
	andl	$224, %edx
	andl	$-32, %r11d
	andl	$31, %ecx
	leaq	(%rbx,%rdx), %r8
	addq	%r9, %rdx
	testb	$31, %r13b
	je	.L2470
	leal	-1(%rcx), %r14d
	movl	%ecx, %r15d
	cmpb	$14, %r14b
	jbe	.L2467
.L2464:
	movzbl	%r11b, %r11d
	movl	%r15d, %r10d
	vmovdqu	(%r9,%r11), %xmm3
	andl	$-16, %r10d
	subl	%r10d, %ecx
	vmovdqu	%xmm3, 1(%rax,%r11)
	movq	%r15, %rax
	andl	$240, %eax
	addq	%rax, %r8
	addq	%rax, %rdx
	andl	$15, %r15d
	je	.L2470
.L2467:
	movzbl	(%rdx), %r15d
	movb	%r15b, (%r8)
	cmpb	$1, %cl
	je	.L2470
	movzbl	1(%rdx), %r14d
	movb	%r14b, 1(%r8)
	cmpb	$2, %cl
	je	.L2470
	movzbl	2(%rdx), %r11d
	movb	%r11b, 2(%r8)
	cmpb	$3, %cl
	je	.L2470
	movzbl	3(%rdx), %r10d
	movb	%r10b, 3(%r8)
	cmpb	$4, %cl
	je	.L2470
	movzbl	4(%rdx), %eax
	movb	%al, 4(%r8)
	cmpb	$5, %cl
	je	.L2470
	movzbl	5(%rdx), %r15d
	movb	%r15b, 5(%r8)
	cmpb	$6, %cl
	je	.L2470
	movzbl	6(%rdx), %r14d
	movb	%r14b, 6(%r8)
	cmpb	$7, %cl
	je	.L2470
	movzbl	7(%rdx), %r11d
	movb	%r11b, 7(%r8)
	cmpb	$8, %cl
	je	.L2470
	movzbl	8(%rdx), %r10d
	movb	%r10b, 8(%r8)
	cmpb	$9, %cl
	je	.L2470
	movzbl	9(%rdx), %eax
	movb	%al, 9(%r8)
	cmpb	$10, %cl
	je	.L2470
	movzbl	10(%rdx), %r15d
	movb	%r15b, 10(%r8)
	cmpb	$11, %cl
	je	.L2470
	movzbl	11(%rdx), %r14d
	movb	%r14b, 11(%r8)
	cmpb	$12, %cl
	je	.L2470
	movzbl	12(%rdx), %r11d
	movb	%r11b, 12(%r8)
	cmpb	$13, %cl
	je	.L2470
	movzbl	13(%rdx), %r10d
	movb	%r10b, 13(%r8)
	cmpb	$14, %cl
	je	.L2470
	movzbl	14(%rdx), %ecx
	movb	%cl, 14(%r8)
	jmp	.L2470
	.p2align 4,,10
	.p2align 3
.L2478:
	movq	%r14, %rax
	jmp	.L2451
	.p2align 4,,10
	.p2align 3
.L2480:
	movq	%rbx, %rax
	jmp	.L2462
.L2822:
	cmpq	$30, %rcx
	jbe	.L2477
	movq	%rax, %r15
	xorl	%ecx, %ecx
	andq	$-32, %r15
	leaq	-32(%r15), %r10
	shrq	$5, %r10
	addq	$1, %r10
	andl	$7, %r10d
	je	.L2802
	cmpq	$1, %r10
	je	.L2677
	cmpq	$2, %r10
	je	.L2678
	cmpq	$3, %r10
	je	.L2679
	cmpq	$4, %r10
	je	.L2680
	cmpq	$5, %r10
	je	.L2681
	cmpq	$6, %r10
	jne	.L2826
.L2682:
	vmovdqu	(%r8,%rcx), %ymm0
	vmovdqu	%ymm0, (%rbx,%rcx)
	addq	$32, %rcx
.L2681:
	vmovdqu	(%r8,%rcx), %ymm5
	vmovdqu	%ymm5, (%rbx,%rcx)
	addq	$32, %rcx
.L2680:
	vmovdqu	(%r8,%rcx), %ymm1
	vmovdqu	%ymm1, (%rbx,%rcx)
	addq	$32, %rcx
.L2679:
	vmovdqu	(%r8,%rcx), %ymm6
	vmovdqu	%ymm6, (%rbx,%rcx)
	addq	$32, %rcx
.L2678:
	vmovdqu	(%r8,%rcx), %ymm4
	vmovdqu	%ymm4, (%rbx,%rcx)
	addq	$32, %rcx
.L2677:
	vmovdqu	(%r8,%rcx), %ymm2
	vmovdqu	%ymm2, (%rbx,%rcx)
	addq	$32, %rcx
	cmpq	%r15, %rcx
	je	.L2792
.L2802:
	movq	-8(%rsp), %r14
.L2443:
	vmovdqu	(%r8,%rcx), %ymm3
	vmovdqu	%ymm3, (%rbx,%rcx)
	vmovdqu	32(%r8,%rcx), %ymm8
	vmovdqu	%ymm8, 32(%rbx,%rcx)
	vmovdqu	64(%r8,%rcx), %ymm9
	vmovdqu	%ymm9, 64(%rbx,%rcx)
	vmovdqu	96(%r8,%rcx), %ymm10
	vmovdqu	%ymm10, 96(%rbx,%rcx)
	vmovdqu	128(%r8,%rcx), %ymm11
	vmovdqu	%ymm11, 128(%rbx,%rcx)
	vmovdqu	160(%r8,%rcx), %ymm12
	vmovdqu	%ymm12, 160(%rbx,%rcx)
	vmovdqu	192(%r8,%rcx), %ymm13
	vmovdqu	%ymm13, 192(%rbx,%rcx)
	vmovdqu	224(%r8,%rcx), %ymm14
	vmovdqu	%ymm14, 224(%rbx,%rcx)
	addq	$256, %rcx
	cmpq	%r15, %rcx
	jne	.L2443
	movq	%r14, -8(%rsp)
.L2792:
	leaq	(%rbx,%r15), %r14
	leaq	(%r8,%r15), %rcx
	cmpq	%rax, %r15
	je	.L2449
	movq	%rax, %r10
	subq	%r15, %r10
	movq	%r10, -16(%rsp)
	subq	$1, %r10
	cmpq	$14, %r10
	jbe	.L2446
.L2442:
	vmovdqu	(%r8,%r15), %xmm15
	vmovdqu	%xmm15, (%rbx,%r15)
	movq	-16(%rsp), %r15
	movq	%r15, %r8
	andq	$-16, %r8
	addq	%r8, %r14
	addq	%r8, %rcx
	andl	$15, %r15d
	je	.L2449
.L2446:
	movzbl	(%rcx), %r10d
	leaq	1(%rcx), %r15
	movb	%r10b, (%r14)
	cmpq	%r15, %r9
	je	.L2449
	movzbl	1(%rcx), %r8d
	leaq	2(%rcx), %r10
	movb	%r8b, 1(%r14)
	cmpq	%r10, %r9
	je	.L2449
	movzbl	2(%rcx), %r15d
	leaq	3(%rcx), %r8
	movb	%r15b, 2(%r14)
	cmpq	%r8, %r9
	je	.L2449
	movzbl	3(%rcx), %r10d
	leaq	4(%rcx), %r15
	movb	%r10b, 3(%r14)
	cmpq	%r15, %r9
	je	.L2449
	movzbl	4(%rcx), %r8d
	leaq	5(%rcx), %r10
	movb	%r8b, 4(%r14)
	cmpq	%r10, %r9
	je	.L2449
	movzbl	5(%rcx), %r15d
	leaq	6(%rcx), %r8
	movb	%r15b, 5(%r14)
	cmpq	%r8, %r9
	je	.L2449
	movzbl	6(%rcx), %r10d
	leaq	7(%rcx), %r15
	movb	%r10b, 6(%r14)
	cmpq	%r15, %r9
	je	.L2449
	movzbl	7(%rcx), %r8d
	leaq	8(%rcx), %r10
	movb	%r8b, 7(%r14)
	cmpq	%r10, %r9
	je	.L2449
	movzbl	8(%rcx), %r15d
	leaq	9(%rcx), %r8
	movb	%r15b, 8(%r14)
	cmpq	%r8, %r9
	je	.L2449
	movzbl	9(%rcx), %r10d
	leaq	10(%rcx), %r15
	movb	%r10b, 9(%r14)
	cmpq	%r15, %r9
	je	.L2449
	movzbl	10(%rcx), %r8d
	leaq	11(%rcx), %r10
	movb	%r8b, 10(%r14)
	cmpq	%r10, %r9
	je	.L2449
	movzbl	11(%rcx), %r15d
	leaq	12(%rcx), %r8
	movb	%r15b, 11(%r14)
	cmpq	%r8, %r9
	je	.L2449
	movzbl	12(%rcx), %r10d
	leaq	13(%rcx), %r15
	movb	%r10b, 12(%r14)
	cmpq	%r15, %r9
	je	.L2449
	movzbl	13(%rcx), %r8d
	leaq	14(%rcx), %r10
	movb	%r8b, 13(%r14)
	cmpq	%r10, %r9
	je	.L2449
	movzbl	14(%rcx), %ecx
	movb	%cl, 14(%r14)
	jmp	.L2449
.L2475:
	xorl	%r14d, %r14d
	jmp	.L2676
.L2481:
	movl	%r13d, %ecx
	movq	%r9, %rdx
	movq	%rbx, %r8
	xorl	%r11d, %r11d
	jmp	.L2464
.L2479:
	movl	%edx, %esi
	movq	%r9, %rcx
	movq	%r14, %r8
	xorl	%r11d, %r11d
	jmp	.L2453
.L2823:
	movzbl	(%r8), %r14d
	movb	%r14b, (%rbx)
	movl	$1, %r14d
	jmp	.L2688
.L2434:
	movq	%rbx, %rax
	cmpq	%r8, %r9
	je	.L2433
	movq	%rdi, %r13
	movq	$-1, %rsi
	xorl	%edx, %edx
	xorl	%r11d, %r11d
	movq	$-1, -8(%rsp)
	jmp	.L2472
.L2821:
	movq	%rbx, %rax
	jmp	.L2440
.L2826:
	vmovdqu	(%r8), %ymm7
	movl	$32, %ecx
	vmovdqu	%ymm7, (%rbx)
	jmp	.L2682
.L2477:
	movq	%rax, -16(%rsp)
	movq	%r8, %rcx
	movq	%rbx, %r14
	xorl	%r15d, %r15d
	jmp	.L2442
	.cfi_endproc
.LFE13857:
	.size	_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_, .-_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_
	.section	.text._ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	.type	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_, @function
_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_:
.LFB13779:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13779
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	movq	%rsi, %r14
	pushq	%r13
	.cfi_offset 13, -40
	movq	%rdx, %r13
	pushq	%r12
	.cfi_offset 12, -48
	movq	%r8, %r12
	pushq	%r10
	pushq	%rbx
	.cfi_offset 10, -56
	.cfi_offset 3, -64
	movq	%rdi, %rbx
	subq	$160, %rsp
	movq	%rcx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movzwl	(%rdi), %eax
	andw	$384, %ax
	cmpw	$128, %ax
	je	.L2918
	cmpw	$256, %ax
	je	.L2919
	movq	$0, -128(%rbp)
	movq	%rsi, %r15
	movb	$0, -120(%rbp)
	movq	%rdx, -168(%rbp)
	testb	$32, (%rdi)
	jne	.L2883
	movq	16(%r8), %r12
.L2878:
	testq	%r15, %r15
	jne	.L2920
.L2867:
	cmpb	$0, -120(%rbp)
	jne	.L2921
.L2872:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L2915
	leaq	-48(%rbp), %rsp
	movq	%r12, %rax
	popq	%rbx
	popq	%r10
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2918:
	.cfi_restore_state
	movzwl	2(%rdi), %eax
	movq	%rax, -176(%rbp)
.L2829:
	movq	$0, -128(%rbp)
	movq	%r14, %r15
	movb	$0, -120(%rbp)
	movq	%r13, -168(%rbp)
	testb	$32, (%rbx)
	jne	.L2877
.L2846:
	movq	-176(%rbp), %rsi
	movq	16(%r12), %r12
	cmpq	%rsi, %r15
	jnb	.L2878
	movzbl	(%rbx), %r8d
	movsbl	6(%rbx), %r9d
	movq	-176(%rbp), %rbx
	movl	%r8d, %ecx
	subq	%r15, %rbx
	andl	$3, %ecx
	jne	.L2869
	testb	$64, %r8b
	je	.L2880
	cmpq	$0, -184(%rbp)
	movl	$48, %r9d
	movl	$2, %ecx
	je	.L2869
	movq	-184(%rbp), %rdi
	movq	%rdi, %rsi
	cmpq	%rdi, %r15
	jb	.L2922
.L2870:
	movq	-168(%rbp), %rdx
	movq	%r12, %rdi
.LEHB2:
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
.L2871:
	movq	-184(%rbp), %rcx
	movl	$48, %r9d
	addq	%rcx, -168(%rbp)
	subq	%rcx, %r15
	movl	$2, %ecx
	.p2align 4,,10
	.p2align 3
.L2869:
	movq	-168(%rbp), %rdx
	movq	%rbx, %r8
	movq	%r15, %rsi
	movq	%r12, %rdi
	call	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
	cmpb	$0, -120(%rbp)
	movq	%rax, %r12
	je	.L2872
.L2921:
	leaq	-128(%rbp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L2872
	.p2align 4,,10
	.p2align 3
.L2883:
	movq	$0, -176(%rbp)
.L2877:
	cmpb	$0, 32(%r12)
	leaq	24(%r12), %rsi
	je	.L2923
.L2847:
	leaq	-160(%rbp), %rdi
	movq	%rdi, -192(%rbp)
	call	_ZNSt6localeC1ERKS_@PLT
	cmpb	$0, -120(%rbp)
	jne	.L2924
	movq	-192(%rbp), %rsi
	leaq	-128(%rbp), %rdi
	movq	%rdi, -200(%rbp)
	call	_ZNSt6localeC1ERKS_@PLT
	movb	$1, -120(%rbp)
.L2849:
	movq	-192(%rbp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	cmpb	$0, -120(%rbp)
	je	.L2925
.L2850:
	movq	-200(%rbp), %rsi
	leaq	-96(%rbp), %rdi
	movq	%rdi, -192(%rbp)
	call	_ZNKSt6locale4nameB5cxx11Ev@PLT
	cmpq	$1, -88(%rbp)
	movq	-96(%rbp), %rdi
	leaq	-80(%rbp), %rsi
	je	.L2926
.L2852:
	cmpq	%rsi, %rdi
	je	.L2855
	movq	-80(%rbp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L2855:
	leaq	_ZNSt7__cxx118numpunctIcE2idE(%rip), %rdi
	call	_ZNKSt6locale2id5_M_idEv@PLT
	movq	-128(%rbp), %r9
	movq	8(%r9), %r10
	movq	(%r10,%rax,8), %rsi
	movq	%rsi, -208(%rbp)
	testq	%rsi, %rsi
	je	.L2857
	movq	(%rsi), %r11
	movq	-192(%rbp), %rdi
	call	*32(%r11)
	movq	-88(%rbp), %rdx
	movq	%rdx, -200(%rbp)
	testq	%rdx, %rdx
	je	.L2860
	movq	-184(%rbp), %rax
	movq	%r14, %r15
	movq	%rsp, %r8
	subq	%rax, %r15
	leaq	39(%rax,%r15,2), %rdi
	movq	%rdi, %rsi
	andq	$-4096, %rdi
	andq	$-16, %rsi
	subq	%rdi, %r8
.L2861:
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r8, %rsp
	je	.L2862
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L2861
	.p2align 4,,10
	.p2align 3
.L2880:
	movl	$32, %r9d
	movl	$2, %ecx
	jmp	.L2869
	.p2align 4,,10
	.p2align 3
.L2924:
	movq	-192(%rbp), %rsi
	leaq	-128(%rbp), %rdi
	movq	%rdi, -200(%rbp)
	call	_ZNSt6localeaSERKS_@PLT
	jmp	.L2849
	.p2align 4,,10
	.p2align 3
.L2920:
	movq	-168(%rbp), %rdx
	movq	%r15, %rsi
	movq	%r12, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
.LEHE2:
	jmp	.L2867
	.p2align 4,,10
	.p2align 3
.L2919:
	movzbl	(%r8), %eax
	movzwl	2(%rdi), %edx
	movl	%eax, %ecx
	andl	$15, %eax
	andl	$15, %ecx
	cmpq	%rax, %rdx
	jb	.L2927
	testb	%cl, %cl
	jne	.L2833
	movq	(%r8), %rsi
	shrq	$4, %rsi
	cmpq	%rsi, %rdx
	jb	.L2928
.L2833:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L2915
.LEHB3:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
.LEHE3:
	.p2align 4,,10
	.p2align 3
.L2926:
	cmpb	$67, (%rdi)
	jne	.L2852
	cmpq	%rsi, %rdi
	je	.L2846
	movq	-80(%rbp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
	jmp	.L2846
	.p2align 4,,10
	.p2align 3
.L2925:
	movq	-200(%rbp), %rdi
	call	_ZNSt6localeC1Ev@PLT
	movb	$1, -120(%rbp)
	jmp	.L2850
	.p2align 4,,10
	.p2align 3
.L2862:
	andl	$4095, %esi
	subq	%rsi, %rsp
	testq	%rsi, %rsi
	je	.L2863
	orq	$0, -8(%rsp,%rsi)
.L2863:
	leaq	31(%rsp), %rdi
	movq	-184(%rbp), %rdx
	andq	$-32, %rdi
	movq	%rdi, -168(%rbp)
	testq	%rdx, %rdx
	je	.L2864
	movq	%r13, %rsi
	call	memcpy@PLT
.L2864:
	movq	-208(%rbp), %rdi
	movq	-184(%rbp), %r9
	movq	(%rdi), %rax
	leaq	0(%r13,%r9), %r15
	addq	%r14, %r13
	movq	-96(%rbp), %r14
.LEHB4:
	call	*24(%rax)
.LEHE4:
	movq	-168(%rbp), %r10
	movq	%r13, %r9
	movq	%r15, %r8
	movsbl	%al, %esi
	movq	-184(%rbp), %r11
	movq	-200(%rbp), %rcx
	movq	%r14, %rdx
	leaq	(%r10,%r11), %rdi
	call	_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_
	movq	-168(%rbp), %r13
	movq	%rax, %r15
	subq	%r13, %r15
.L2860:
	movq	-192(%rbp), %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	jmp	.L2846
	.p2align 4,,10
	.p2align 3
.L2923:
	movq	%rsi, %rdi
	movq	%rsi, -192(%rbp)
	call	_ZNSt6localeC1Ev@PLT
	movb	$1, 32(%r12)
	movq	-192(%rbp), %rsi
	jmp	.L2847
	.p2align 4,,10
	.p2align 3
.L2922:
	testq	%r15, %r15
	je	.L2871
	movq	%r15, %rsi
	jmp	.L2870
	.p2align 4,,10
	.p2align 3
.L2927:
	movq	(%r8), %r8
	leaq	(%rdx,%rdx,4), %rcx
	salq	$4, %rdx
	addq	8(%r12), %rdx
	vmovdqa	(%rdx), %xmm1
	shrq	$4, %r8
	shrq	%cl, %r8
	vmovdqa	%xmm1, -128(%rbp)
	andl	$31, %r8d
.L2832:
	leaq	.L2836(%rip), %r10
	movzbl	%r8b, %r9d
	movb	%r8b, -112(%rbp)
	vmovdqu	-128(%rbp), %ymm0
	movslq	(%r10,%r9,4), %r11
	vmovdqu	%ymm0, -160(%rbp)
	addq	%r10, %r11
	notrack jmp	*%r11
	.section	.rodata._ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,"aG",@progbits,_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,comdat
	.align 4
	.align 4
.L2836:
	.long	.L2911-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2840-.L2836
	.long	.L2839-.L2836
	.long	.L2838-.L2836
	.long	.L2837-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.long	.L2835-.L2836
	.section	.text._ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,comdat
.L2838:
	movq	-160(%rbp), %rdx
	testq	%rdx, %rdx
	js	.L2835
.L2914:
	movq	%rdx, -176(%rbp)
	vzeroupper
	jmp	.L2829
.L2839:
	movl	-160(%rbp), %r15d
	movq	%r15, -176(%rbp)
	vzeroupper
	jmp	.L2829
.L2840:
	movslq	-160(%rbp), %rdx
	testl	%edx, %edx
	jns	.L2914
.L2835:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L2929
	leaq	.LC18(%rip), %rdi
	vzeroupper
.LEHB5:
	call	_ZSt20__throw_format_errorPKc
.LEHE5:
	.p2align 4,,10
	.p2align 3
.L2837:
	movq	-160(%rbp), %rdx
	jmp	.L2914
.L2911:
	vzeroupper
	jmp	.L2833
	.p2align 4,,10
	.p2align 3
.L2928:
	salq	$5, %rdx
	addq	8(%r8), %rdx
	vmovdqu	(%rdx), %xmm2
	vmovdqa	%xmm2, -128(%rbp)
	movzbl	16(%rdx), %edi
	movb	%dil, -112(%rbp)
	movzbl	16(%rdx), %r8d
	jmp	.L2832
.L2929:
	vzeroupper
.L2915:
	call	__stack_chk_fail@PLT
.L2857:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L2915
.LEHB6:
	call	_ZSt16__throw_bad_castv@PLT
.LEHE6:
.L2884:
	endbr64
	movq	%rax, %r12
	jmp	.L2874
.L2873:
	movq	-192(%rbp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
.L2874:
	cmpb	$0, -120(%rbp)
	jne	.L2930
	vzeroupper
.L2875:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L2915
	movq	%r12, %rdi
.LEHB7:
	call	_Unwind_Resume@PLT
.LEHE7:
.L2885:
	endbr64
	movq	%rax, %r12
	jmp	.L2873
.L2930:
	leaq	-128(%rbp), %rdi
	vzeroupper
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L2875
	.cfi_endproc
.LFE13779:
	.section	.gcc_except_table._ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,"aG",@progbits,_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,comdat
.LLSDA13779:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13779-.LLSDACSB13779
.LLSDACSB13779:
	.uleb128 .LEHB2-.LFB13779
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L2884-.LFB13779
	.uleb128 0
	.uleb128 .LEHB3-.LFB13779
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB4-.LFB13779
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L2885-.LFB13779
	.uleb128 0
	.uleb128 .LEHB5-.LFB13779
	.uleb128 .LEHE5-.LEHB5
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB6-.LFB13779
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L2884-.LFB13779
	.uleb128 0
	.uleb128 .LEHB7-.LFB13779
	.uleb128 .LEHE7-.LEHB7
	.uleb128 0
	.uleb128 0
.LLSDACSE13779:
	.section	.text._ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_,comdat
	.size	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_, .-_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	.section	.rodata._ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_.str1.1,"aMS",@progbits,1
.LC35:
	.string	"0b"
.LC36:
	.string	"0B"
.LC37:
	.string	"0"
.LC38:
	.string	"0X"
.LC39:
	.string	"0x"
	.section	.rodata._ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_.str1.8,"aMS",@progbits,1
	.align 8
.LC40:
	.string	"format error: integer not representable as character"
	.section	.text._ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13667:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	%esi, %eax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movq	%rdx, %r13
	pushq	%r12
	.cfi_offset 12, -48
	movq	%rdi, %r12
	pushq	%rbx
	subq	$264, %rsp
	.cfi_offset 3, -56
	movzbl	1(%rdi), %ecx
	movq	%fs:40, %rdx
	movq	%rdx, 248(%rsp)
	xorl	%edx, %edx
	movl	%ecx, %r15d
	andl	$120, %r15d
	cmpb	$56, %r15b
	je	.L3079
	shrb	$3, %cl
	andl	$15, %ecx
	cmpb	$4, %cl
	je	.L2936
	ja	.L2937
	cmpb	$1, %cl
	jbe	.L2938
	cmpb	$16, %r15b
	leaq	.LC35(%rip), %r11
	leaq	.LC36(%rip), %rbx
	cmovne	%rbx, %r11
	testb	%sil, %sil
	jne	.L3080
	leaq	24(%rsp), %r14
	leaq	23(%rsp), %r9
	movl	$48, %esi
.L2943:
	movzbl	(%r12), %ebx
	movb	%sil, 23(%rsp)
	testb	$16, %bl
	je	.L2978
.L2977:
	movq	$-2, %rdx
	movl	$2, %r8d
.L2947:
	addq	%r9, %rdx
	movl	%r8d, %r10d
	testl	%r8d, %r8d
	je	.L2948
	xorl	%ecx, %ecx
	leal	-1(%r8), %r15d
	movl	$1, %r8d
	movzbl	(%r11,%rcx), %edi
	andl	$7, %r15d
	movb	%dil, (%rdx,%rcx)
	cmpl	%r10d, %r8d
	jnb	.L2948
	testl	%r15d, %r15d
	je	.L2963
	cmpl	$1, %r15d
	je	.L3046
	cmpl	$2, %r15d
	je	.L3047
	cmpl	$3, %r15d
	je	.L3048
	cmpl	$4, %r15d
	je	.L3049
	cmpl	$5, %r15d
	je	.L3050
	cmpl	$6, %r15d
	je	.L3051
	movl	$1, %eax
	movl	$2, %r8d
	movzbl	(%r11,%rax), %esi
	movb	%sil, (%rdx,%rax)
.L3051:
	movl	%r8d, %r15d
	addl	$1, %r8d
	movzbl	(%r11,%r15), %ecx
	movb	%cl, (%rdx,%r15)
.L3050:
	movl	%r8d, %eax
	addl	$1, %r8d
	movzbl	(%r11,%rax), %edi
	movb	%dil, (%rdx,%rax)
.L3049:
	movl	%r8d, %esi
	addl	$1, %r8d
	movzbl	(%r11,%rsi), %r15d
	movb	%r15b, (%rdx,%rsi)
.L3048:
	movl	%r8d, %ecx
	addl	$1, %r8d
	movzbl	(%r11,%rcx), %eax
	movb	%al, (%rdx,%rcx)
.L3047:
	movl	%r8d, %esi
	addl	$1, %r8d
	movzbl	(%r11,%rsi), %edi
	movb	%dil, (%rdx,%rsi)
.L3046:
	movl	%r8d, %r15d
	addl	$1, %r8d
	movzbl	(%r11,%r15), %ecx
	movb	%cl, (%rdx,%r15)
	cmpl	%r10d, %r8d
	jnb	.L2948
.L2963:
	movl	%r8d, %eax
	leal	1(%r8), %r15d
	leal	2(%r8), %ecx
	movzbl	(%r11,%rax), %esi
	movzbl	(%r11,%r15), %edi
	movb	%sil, (%rdx,%rax)
	leal	3(%r8), %esi
	movzbl	(%r11,%rcx), %eax
	movb	%dil, (%rdx,%r15)
	movzbl	(%r11,%rsi), %r15d
	movb	%al, (%rdx,%rcx)
	leal	4(%r8), %ecx
	leal	5(%r8), %eax
	movb	%r15b, (%rdx,%rsi)
	movzbl	(%r11,%rcx), %edi
	leal	6(%r8), %r15d
	movzbl	(%r11,%rax), %esi
	movb	%dil, (%rdx,%rcx)
	movzbl	(%r11,%r15), %ecx
	movb	%sil, (%rdx,%rax)
	leal	7(%r8), %eax
	addl	$8, %r8d
	movzbl	(%r11,%rax), %edi
	movb	%cl, (%rdx,%r15)
	movb	%dil, (%rdx,%rax)
	cmpl	%r10d, %r8d
	jb	.L2963
	jmp	.L2948
	.p2align 4,,10
	.p2align 3
.L3079:
	testb	%sil, %sil
	js	.L2933
	movb	%sil, 19(%rsp)
	movq	%rdi, %rcx
	leaq	19(%rsp), %rsi
	movq	%r13, %rdx
	movl	$1, %edi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
.L2934:
	movq	248(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L3078
	addq	$264, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L2938:
	.cfi_restore_state
	movzbl	%sil, %r11d
	testb	%sil, %sil
	jne	.L2949
	movb	$48, 23(%rsp)
	leaq	24(%rsp), %r14
	leaq	23(%rsp), %r9
.L2950:
	movzbl	(%r12), %ebx
	movq	%r9, %rdx
.L2948:
	shrb	$2, %bl
	movl	$43, %esi
	andl	$3, %ebx
	cmpl	$1, %ebx
	je	.L2967
.L3082:
	cmpl	$3, %ebx
	je	.L2979
.L2966:
	movq	%r14, %rsi
	movq	%r9, %rcx
	movq	%r13, %r8
	movq	%r12, %rdi
	subq	%rdx, %rsi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L2934
	.p2align 4,,10
	.p2align 3
.L2936:
	testb	%sil, %sil
	je	.L2972
	movzbl	%sil, %ebx
	movl	$2863311531, %esi
	bsrl	%ebx, %edx
	leal	3(%rdx), %r14d
	imulq	%rsi, %r14
	shrq	$33, %r14
	cmpl	$63, %ebx
	jbe	.L2955
	movl	%ebx, %edi
	andl	$7, %eax
	shrl	$6, %ebx
	shrl	$3, %edi
	addl	$48, %eax
	andl	$7, %edi
	movb	%al, 25(%rsp)
	addl	$48, %edi
	movb	%dil, 24(%rsp)
.L2956:
	addl	$48, %ebx
.L2957:
	leaq	23(%rsp), %r9
	movl	%r14d, %r14d
	movl	$1, %ecx
	movl	$1, %r8d
	addq	%r9, %r14
	leaq	.LC37(%rip), %r11
.L2954:
	movb	%bl, 23(%rsp)
	movzbl	(%r12), %ebx
.L2958:
	testb	$16, %bl
	je	.L2978
	testb	%cl, %cl
	jne	.L3081
.L2978:
	shrb	$2, %bl
	movq	%r9, %rdx
	movl	$43, %esi
	andl	$3, %ebx
	cmpl	$1, %ebx
	jne	.L3082
.L2967:
	movb	%sil, -1(%rdx)
	subq	$1, %rdx
	jmp	.L2966
	.p2align 4,,10
	.p2align 3
.L2949:
	cmpl	$9, %r11d
	jbe	.L2971
	cmpl	$99, %r11d
	jbe	.L3083
	movl	%r11d, %r14d
	vmovdqa	.LC26(%rip), %ymm7
	vmovdqa	.LC27(%rip), %ymm0
	imulq	$1374389535, %r14, %rsi
	vmovdqa	.LC28(%rip), %ymm1
	vmovdqa	.LC29(%rip), %ymm2
	movl	$3, %r14d
	vmovdqa	.LC30(%rip), %ymm3
	vmovdqa	.LC31(%rip), %ymm4
	vmovdqu	%ymm7, 32(%rsp)
	vmovdqa	.LC32(%rip), %xmm5
	vmovdqu	%ymm0, 64(%rsp)
	shrq	$37, %rsi
	vmovdqu	%ymm1, 96(%rsp)
	vmovdqu	%ymm4, 192(%rsp)
	imull	$100, %esi, %r9d
	vmovdqu	%ymm2, 128(%rsp)
	vmovdqu	%ymm3, 160(%rsp)
	vmovdqu	%xmm5, 217(%rsp)
	subl	%r9d, %r11d
	leal	1(%r11,%r11), %edx
	movzwl	31(%rsp,%rdx), %r10d
	movw	%r10w, 24(%rsp)
	vzeroupper
.L2951:
	addl	$48, %esi
.L2953:
	leaq	23(%rsp), %r9
	movb	%sil, 23(%rsp)
	addq	%r9, %r14
	jmp	.L2950
	.p2align 4,,10
	.p2align 3
.L2937:
	cmpb	$40, %r15b
	je	.L3084
	testb	%sil, %sil
	jne	.L2974
	movb	$48, 23(%rsp)
	movzbl	(%rdi), %ebx
	leaq	24(%rsp), %r14
	leaq	.LC38(%rip), %r11
	leaq	23(%rsp), %r9
	cmpb	$48, %r15b
	je	.L2961
.L2960:
	testb	$16, %bl
	jne	.L2977
	jmp	.L2978
	.p2align 4,,10
	.p2align 3
.L2979:
	movl	$32, %esi
	subq	$1, %rdx
	movb	%sil, (%rdx)
	jmp	.L2966
	.p2align 4,,10
	.p2align 3
.L2972:
	movl	$48, %ebx
	xorl	%ecx, %ecx
	leaq	24(%rsp), %r14
	xorl	%r8d, %r8d
	xorl	%r11d, %r11d
	leaq	23(%rsp), %r9
	jmp	.L2954
	.p2align 4,,10
	.p2align 3
.L3080:
	movzbl	%sil, %esi
	movl	$32, %r14d
	movl	$31, %edx
	bsrl	%esi, %r8d
	xorl	$31, %r8d
	subl	%r8d, %r14d
	subl	%r8d, %edx
	je	.L2946
	andl	$1, %eax
	movl	%edx, %r10d
	movl	%esi, %r15d
	movl	$30, %edi
	addl	$48, %eax
	shrl	%r15d
	movb	%al, 23(%rsp,%r10)
	subl	%r8d, %edi
	je	.L2946
	andl	$1, %r15d
	movl	%edi, %r9d
	movl	%esi, %ecx
	movl	$29, %eax
	addl	$48, %r15d
	shrl	$2, %ecx
	movb	%r15b, 23(%rsp,%r9)
	subl	%r8d, %eax
	je	.L2946
	andl	$1, %ecx
	movl	%eax, %edx
	movl	%esi, %edi
	movl	$28, %ebx
	addl	$48, %ecx
	shrl	$3, %edi
	movb	%cl, 23(%rsp,%rdx)
	subl	%r8d, %ebx
	je	.L2946
	andl	$1, %edi
	movl	%ebx, %r15d
	movl	%esi, %r9d
	movl	$27, %r10d
	addl	$48, %edi
	shrl	$4, %r9d
	movb	%dil, 23(%rsp,%r15)
	subl	%r8d, %r10d
	je	.L2946
	andl	$1, %r9d
	movl	%r10d, %eax
	movl	%esi, %ebx
	movl	$26, %ecx
	addl	$48, %r9d
	shrl	$5, %ebx
	movb	%r9b, 23(%rsp,%rax)
	subl	%r8d, %ecx
	je	.L2946
	andl	$1, %ebx
	movl	%ecx, %edx
	shrl	$6, %esi
	addl	$48, %ebx
	movb	%bl, 23(%rsp,%rdx)
	cmpl	$24, %r8d
	jne	.L2946
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 24(%rsp)
.L2946:
	leaq	23(%rsp), %r9
	movslq	%r14d, %r14
	movl	$49, %esi
	addq	%r9, %r14
	jmp	.L2943
	.p2align 4,,10
	.p2align 3
.L3084:
	testb	%sil, %sil
	jne	.L2973
	movb	$48, 23(%rsp)
	movzbl	(%rdi), %ebx
	leaq	24(%rsp), %r14
	leaq	.LC39(%rip), %r11
	leaq	23(%rsp), %r9
	jmp	.L2960
	.p2align 4,,10
	.p2align 3
.L2974:
	leaq	.LC38(%rip), %r9
.L2959:
	leaq	23(%rsp), %r8
	movzbl	%al, %edx
	leaq	31(%rsp), %rsi
	movq	%r9, (%rsp)
	movq	%r8, %rdi
	movq	%r8, 8(%rsp)
	call	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	cmpb	$48, %r15b
	movzbl	(%r12), %ebx
	movq	8(%rsp), %r9
	movq	(%rsp), %r11
	movq	%rax, %r14
	jne	.L2960
	cmpq	%r9, %rax
	je	.L2976
.L2961:
	movq	%r14, %r10
	movq	%r9, %r15
	subq	%r9, %r10
	andl	$7, %r10d
	je	.L2962
	cmpq	$1, %r10
	je	.L3039
	cmpq	$2, %r10
	je	.L3040
	cmpq	$3, %r10
	je	.L3041
	cmpq	$4, %r10
	je	.L3042
	cmpq	$5, %r10
	je	.L3043
	cmpq	$6, %r10
	je	.L3044
	movsbl	(%r9), %edi
	movq	%r11, (%rsp)
	leaq	24(%rsp), %r15
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	8(%rsp), %r9
	movq	(%rsp), %r11
	movb	%al, (%r9)
.L3044:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$1, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r9
	movq	8(%rsp), %r11
	movb	%al, -1(%r15)
.L3043:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$1, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r9
	movq	8(%rsp), %r11
	movb	%al, -1(%r15)
.L3042:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$1, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r9
	movq	8(%rsp), %r11
	movb	%al, -1(%r15)
.L3041:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$1, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r9
	movq	8(%rsp), %r11
	movb	%al, -1(%r15)
.L3040:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$1, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r9
	movq	8(%rsp), %r11
	movb	%al, -1(%r15)
.L3039:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$1, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movq	8(%rsp), %r11
	movq	(%rsp), %r9
	movb	%al, -1(%r15)
	cmpq	%r14, %r15
	je	.L2976
.L2962:
	movsbl	(%r15), %edi
	movq	%r9, (%rsp)
	addq	$8, %r15
	movq	%r11, 8(%rsp)
	call	toupper@PLT
	movsbl	-7(%r15), %edi
	movb	%al, -8(%r15)
	call	toupper@PLT
	movsbl	-6(%r15), %edi
	movb	%al, -7(%r15)
	call	toupper@PLT
	movsbl	-5(%r15), %edi
	movb	%al, -6(%r15)
	call	toupper@PLT
	movsbl	-4(%r15), %edi
	movb	%al, -5(%r15)
	call	toupper@PLT
	movsbl	-3(%r15), %edi
	movb	%al, -4(%r15)
	call	toupper@PLT
	movsbl	-2(%r15), %edi
	movb	%al, -3(%r15)
	call	toupper@PLT
	movsbl	-1(%r15), %edi
	movb	%al, -2(%r15)
	call	toupper@PLT
	movq	8(%rsp), %r11
	movq	(%rsp), %r9
	movb	%al, -1(%r15)
	cmpq	%r14, %r15
	jne	.L2962
.L2976:
	movl	$1, %ecx
	movl	$2, %r8d
	jmp	.L2958
	.p2align 4,,10
	.p2align 3
.L2973:
	leaq	.LC39(%rip), %r9
	jmp	.L2959
	.p2align 4,,10
	.p2align 3
.L3083:
	addl	%r11d, %r11d
	vmovdqa	.LC26(%rip), %ymm6
	vmovdqa	.LC27(%rip), %ymm8
	movl	$2, %r14d
	vmovdqa	.LC28(%rip), %ymm9
	vmovdqa	.LC29(%rip), %ymm10
	leal	1(%r11), %r8d
	vmovdqa	.LC30(%rip), %ymm11
	vmovdqa	.LC31(%rip), %ymm12
	vmovdqu	%ymm6, 32(%rsp)
	vmovdqa	.LC32(%rip), %xmm13
	vmovdqu	%ymm8, 64(%rsp)
	vmovdqu	%ymm12, 192(%rsp)
	vmovdqu	%ymm9, 96(%rsp)
	vmovdqu	%ymm10, 128(%rsp)
	vmovdqu	%ymm11, 160(%rsp)
	vmovdqu	%xmm13, 217(%rsp)
	movzbl	32(%rsp,%r8), %ebx
	movzbl	32(%rsp,%r11), %esi
	movb	%bl, 24(%rsp)
	vzeroupper
	jmp	.L2953
	.p2align 4,,10
	.p2align 3
.L2971:
	movl	$1, %r14d
	jmp	.L2951
	.p2align 4,,10
	.p2align 3
.L3081:
	movq	%r8, %rdx
	negq	%rdx
	jmp	.L2947
.L2933:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L2935
.L3078:
	call	__stack_chk_fail@PLT
.L2955:
	cmpl	$7, %ebx
	jbe	.L2956
	andl	$7, %eax
	shrl	$3, %ebx
	addl	$48, %eax
	addl	$48, %ebx
	movb	%al, 24(%rsp)
	jmp	.L2957
.L2935:
	leaq	.LC40(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE13667:
	.size	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.text._ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13672:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %r8
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%rdi, %r12
	pushq	%rbx
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$360, %rsp
	movq	%fs:40, %rax
	movq	%rax, 344(%rsp)
	xorl	%eax, %eax
	movzbl	1(%rdi), %eax
	movl	%eax, %edx
	andl	$120, %edx
	cmpb	$56, %dl
	je	.L3299
	shrb	$3, %al
	andl	$15, %eax
	testq	%rsi, %rsi
	js	.L3300
	movq	%rsi, %rcx
	cmpb	$4, %al
	je	.L3096
	ja	.L3097
	cmpb	$1, %al
	jbe	.L3098
	cmpb	$16, %dl
	leaq	.LC35(%rip), %r9
	leaq	.LC36(%rip), %rsi
	cmovne	%rsi, %r9
	testq	%rbx, %rbx
	jne	.L3094
	leaq	52(%rsp), %r14
	leaq	51(%rsp), %r15
	movl	$48, %ecx
.L3103:
	movzbl	(%r12), %r13d
	movb	%cl, 51(%rsp)
	testb	$16, %r13b
	je	.L3156
.L3155:
	movq	$-2, %rdx
	movl	$2, %eax
.L3107:
	addq	%r15, %rdx
	movl	%eax, %esi
	testl	%eax, %eax
	je	.L3108
	xorl	%r11d, %r11d
	leal	-1(%rax), %r10d
	movzbl	(%r9,%r11), %edi
	andl	$7, %r10d
	movb	%dil, (%rdx,%r11)
	movl	$1, %r11d
	cmpl	%eax, %r11d
	jnb	.L3108
	testl	%r10d, %r10d
	je	.L3135
	cmpl	$1, %r10d
	je	.L3257
	cmpl	$2, %r10d
	je	.L3258
	cmpl	$3, %r10d
	je	.L3259
	cmpl	$4, %r10d
	je	.L3260
	cmpl	$5, %r10d
	je	.L3261
	cmpl	$6, %r10d
	je	.L3262
	movl	$1, %eax
	movl	$2, %r11d
	movzbl	(%r9,%rax), %ecx
	movb	%cl, (%rdx,%rax)
.L3262:
	movl	%r11d, %r10d
	addl	$1, %r11d
	movzbl	(%r9,%r10), %edi
	movb	%dil, (%rdx,%r10)
.L3261:
	movl	%r11d, %eax
	addl	$1, %r11d
	movzbl	(%r9,%rax), %ecx
	movb	%cl, (%rdx,%rax)
.L3260:
	movl	%r11d, %r10d
	addl	$1, %r11d
	movzbl	(%r9,%r10), %edi
	movb	%dil, (%rdx,%r10)
.L3259:
	movl	%r11d, %eax
	addl	$1, %r11d
	movzbl	(%r9,%rax), %ecx
	movb	%cl, (%rdx,%rax)
.L3258:
	movl	%r11d, %r10d
	addl	$1, %r11d
	movzbl	(%r9,%r10), %edi
	movb	%dil, (%rdx,%r10)
.L3257:
	movl	%r11d, %eax
	addl	$1, %r11d
	movzbl	(%r9,%rax), %ecx
	movb	%cl, (%rdx,%rax)
	cmpl	%esi, %r11d
	jnb	.L3108
.L3135:
	movl	%r11d, %r10d
	leal	1(%r11), %eax
	movzbl	(%r9,%r10), %edi
	movzbl	(%r9,%rax), %ecx
	movb	%dil, (%rdx,%r10)
	leal	2(%r11), %r10d
	movb	%cl, (%rdx,%rax)
	leal	3(%r11), %eax
	movzbl	(%r9,%r10), %edi
	movzbl	(%r9,%rax), %ecx
	movb	%dil, (%rdx,%r10)
	leal	4(%r11), %r10d
	movb	%cl, (%rdx,%rax)
	leal	5(%r11), %eax
	movzbl	(%r9,%r10), %edi
	movzbl	(%r9,%rax), %ecx
	movb	%dil, (%rdx,%r10)
	leal	6(%r11), %r10d
	movb	%cl, (%rdx,%rax)
	leal	7(%r11), %eax
	movzbl	(%r9,%r10), %edi
	addl	$8, %r11d
	movzbl	(%r9,%rax), %ecx
	movb	%dil, (%rdx,%r10)
	movb	%cl, (%rdx,%rax)
	cmpl	%esi, %r11d
	jb	.L3135
	.p2align 4,,10
	.p2align 3
.L3108:
	shrb	$2, %r13b
	leaq	-1(%rdx), %rsi
	andl	$3, %r13d
	testq	%rbx, %rbx
	jns	.L3109
	movb	$45, -1(%rdx)
	movq	%rsi, %rdx
.L3137:
	movq	%r14, %rsi
	movq	%r15, %rcx
	movq	%r12, %rdi
	subq	%rdx, %rsi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L3088
	.p2align 4,,10
	.p2align 3
.L3300:
	movq	%rsi, %rcx
	negq	%rcx
	cmpb	$4, %al
	je	.L3091
	ja	.L3092
	cmpb	$1, %al
	jbe	.L3093
	cmpb	$16, %dl
	leaq	.LC35(%rip), %r9
	leaq	.LC36(%rip), %r13
	cmovne	%r13, %r9
.L3094:
	bsrq	%rcx, %r14
	movl	$64, %esi
	movl	$63, %r15d
	xorq	$63, %r14
	subl	%r14d, %esi
	subl	%r14d, %r15d
	je	.L3106
	movl	%r15d, %r10d
	movl	$62, %r11d
	leaq	48(%rsp,%r10), %rax
	leaq	47(%rsp,%r10), %rdi
	subl	%r14d, %r11d
	subq	%r11, %rdi
	movq	%rax, %rdx
	subq	%rdi, %rdx
	andl	$7, %edx
	je	.L3105
	cmpq	$1, %rdx
	je	.L3241
	cmpq	$2, %rdx
	je	.L3242
	cmpq	$3, %rdx
	je	.L3243
	cmpq	$4, %rdx
	je	.L3244
	cmpq	$5, %rdx
	je	.L3245
	cmpq	$6, %rdx
	je	.L3246
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3246:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3245:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3244:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3243:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3242:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3241:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
	cmpq	%rdi, %rax
	je	.L3106
.L3105:
	movl	%ecx, %edx
	movq	%rcx, %r13
	movq	%rcx, %r14
	movq	%rcx, %r15
	andl	$1, %edx
	movq	%rcx, %r10
	movq	%rcx, %r11
	shrq	%r13
	addl	$48, %edx
	shrq	$2, %r14
	andl	$1, %r13d
	subq	$8, %rax
	movb	%dl, 11(%rax)
	movq	%rcx, %rdx
	shrq	$3, %r15
	andl	$1, %r14d
	shrq	$6, %rdx
	shrq	$4, %r10
	andl	$1, %r15d
	addl	$48, %r13d
	andl	$1, %edx
	shrq	$5, %r11
	andl	$1, %r10d
	addl	$48, %r14d
	addl	$48, %edx
	andl	$1, %r11d
	addl	$48, %r15d
	addl	$48, %r10d
	movb	%dl, 5(%rax)
	movl	%ecx, %edx
	addl	$48, %r11d
	shrq	$8, %rcx
	shrb	$7, %dl
	movb	%r13b, 10(%rax)
	addl	$48, %edx
	movb	%r14b, 9(%rax)
	movb	%r15b, 8(%rax)
	movb	%r10b, 7(%rax)
	movb	%r11b, 6(%rax)
	movb	%dl, 4(%rax)
	cmpq	%rdi, %rax
	jne	.L3105
.L3106:
	movslq	%esi, %rcx
	leaq	51(%rsp), %r15
	leaq	(%r15,%rcx), %r14
	movl	$49, %ecx
	jmp	.L3103
	.p2align 4,,10
	.p2align 3
.L3299:
	leaq	128(%rsi), %rdi
	cmpq	$255, %rdi
	ja	.L3087
	movb	%sil, 47(%rsp)
	movq	%r12, %rcx
	leaq	47(%rsp), %rsi
	movq	%r8, %rdx
	movl	$1, %edi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
.L3088:
	movq	344(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L3298
	addq	$360, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L3097:
	.cfi_restore_state
	cmpb	$40, %dl
	je	.L3301
	testq	%rsi, %rsi
	jne	.L3152
	movb	$48, 51(%rsp)
	movzbl	(%rdi), %r13d
	cmpb	$48, %dl
	je	.L3153
	leaq	52(%rsp), %r14
	leaq	.LC38(%rip), %r9
	leaq	51(%rsp), %r15
	jmp	.L3128
	.p2align 4,,10
	.p2align 3
.L3092:
	cmpb	$40, %dl
	leaq	.LC39(%rip), %r9
	leaq	.LC38(%rip), %r10
	cmovne	%r10, %r9
.L3095:
	bsrq	%rcx, %r14
	vmovdqa	.LC25(%rip), %xmm0
	leal	4(%r14), %edi
	vmovdqa	%xmm0, 128(%rsp)
	shrl	$2, %edi
	leal	-1(%rdi), %r15d
	cmpq	$255, %rcx
	jbe	.L3130
.L3131:
	movq	%rcx, %r13
	movq	%rcx, %rsi
	movl	%r15d, %eax
	shrq	$8, %rcx
	shrq	$4, %r13
	andl	$15, %esi
	leal	-1(%r15), %r10d
	movzbl	128(%rsp,%rsi), %r11d
	andl	$15, %r13d
	movzbl	128(%rsp,%r13), %r14d
	movb	%r11b, 51(%rsp,%rax)
	leal	-2(%r15), %eax
	movb	%r14b, 51(%rsp,%r10)
	cmpq	$255, %rcx
	jbe	.L3130
	movq	%rcx, %r13
	movq	%rcx, %rsi
	leal	-3(%r15), %r10d
	shrq	$8, %rcx
	shrq	$4, %r13
	andl	$15, %esi
	movzbl	128(%rsp,%rsi), %r11d
	andl	$15, %r13d
	movzbl	128(%rsp,%r13), %r14d
	movb	%r11b, 51(%rsp,%rax)
	leal	-4(%r15), %eax
	movb	%r14b, 51(%rsp,%r10)
	cmpq	$255, %rcx
	jbe	.L3130
	movq	%rcx, %r13
	movq	%rcx, %rsi
	leal	-5(%r15), %r10d
	shrq	$8, %rcx
	shrq	$4, %r13
	andl	$15, %esi
	movzbl	128(%rsp,%rsi), %r11d
	andl	$15, %r13d
	movzbl	128(%rsp,%r13), %r14d
	movb	%r11b, 51(%rsp,%rax)
	leal	-6(%r15), %eax
	movb	%r14b, 51(%rsp,%r10)
	cmpq	$255, %rcx
	jbe	.L3130
	movq	%rcx, %r13
	movq	%rcx, %rsi
	leal	-7(%r15), %r10d
	shrq	$8, %rcx
	shrq	$4, %r13
	andl	$15, %esi
	subl	$8, %r15d
	movzbl	128(%rsp,%rsi), %r11d
	andl	$15, %r13d
	movzbl	128(%rsp,%r13), %r14d
	movb	%r11b, 51(%rsp,%rax)
	movb	%r14b, 51(%rsp,%r10)
	cmpq	$255, %rcx
	ja	.L3131
	.p2align 4,,10
	.p2align 3
.L3130:
	cmpq	$15, %rcx
	ja	.L3302
	movzbl	128(%rsp,%rcx), %ecx
.L3133:
	leaq	51(%rsp), %r15
	movl	%edi, %esi
	movb	%cl, 51(%rsp)
	movzbl	(%r12), %r13d
	leaq	(%r15,%rsi), %r14
	cmpb	$48, %dl
	je	.L3303
.L3128:
	testb	$16, %r13b
	jne	.L3155
.L3156:
	movq	%r15, %rdx
	jmp	.L3108
	.p2align 4,,10
	.p2align 3
.L3098:
	testq	%rsi, %rsi
	jne	.L3093
	movzbl	(%rdi), %r13d
	movb	$48, 51(%rsp)
	leaq	51(%rsp), %r15
	leaq	52(%rsp), %r14
	movq	%r15, %rdx
	leaq	50(%rsp), %rsi
	shrb	$2, %r13b
	andl	$3, %r13d
.L3109:
	movzbl	%r13b, %ebx
	cmpl	$1, %ebx
	je	.L3304
	cmpl	$3, %ebx
	jne	.L3137
	movb	$32, -1(%rdx)
.L3140:
	movq	%rsi, %rdx
	jmp	.L3137
	.p2align 4,,10
	.p2align 3
.L3304:
	movb	$43, -1(%rdx)
	jmp	.L3140
	.p2align 4,,10
	.p2align 3
.L3096:
	testq	%rsi, %rsi
	jne	.L3091
	xorl	%r11d, %r11d
	xorl	%eax, %eax
	xorl	%r9d, %r9d
	movl	$48, %edi
	leaq	52(%rsp), %r14
	leaq	51(%rsp), %r15
.L3122:
	movb	%dil, 51(%rsp)
	movzbl	(%r12), %r13d
.L3127:
	testb	$16, %r13b
	je	.L3156
	movq	%rax, %rdx
	negq	%rdx
	testb	%r11b, %r11b
	jne	.L3107
	movq	%r15, %rdx
	jmp	.L3108
	.p2align 4,,10
	.p2align 3
.L3093:
	cmpq	$9, %rcx
	jbe	.L3146
	cmpq	$99, %rcx
	jbe	.L3305
	cmpq	$999, %rcx
	jbe	.L3147
	cmpq	$9999, %rcx
	jbe	.L3148
	movq	%rcx, %rdx
	movl	$1, %esi
	movabsq	$3777893186295716171, %r9
	jmp	.L3114
	.p2align 4,,10
	.p2align 3
.L3118:
	cmpq	$999999, %r11
	jbe	.L3306
	cmpq	$9999999, %r11
	jbe	.L3307
	cmpq	$99999999, %r11
	jbe	.L3308
.L3114:
	movq	%rdx, %rax
	movq	%rdx, %r11
	movl	%esi, %r13d
	addl	$4, %esi
	mulq	%r9
	shrq	$11, %rdx
	cmpq	$99999, %r11
	ja	.L3118
.L3116:
	cmpl	$64, %esi
	ja	.L3149
	leal	-1(%rsi), %r15d
.L3113:
	vmovdqa	.LC26(%rip), %ymm1
	vmovdqa	.LC27(%rip), %ymm2
	movabsq	$2951479051793528259, %rdi
	vmovdqa	.LC28(%rip), %ymm3
	vmovdqa	.LC29(%rip), %ymm4
	vmovdqa	.LC30(%rip), %ymm5
	vmovdqa	.LC31(%rip), %ymm6
	vmovdqu	%ymm1, 128(%rsp)
	vmovdqa	.LC32(%rip), %xmm7
	vmovdqu	%ymm2, 160(%rsp)
	vmovdqu	%ymm6, 288(%rsp)
	vmovdqu	%ymm3, 192(%rsp)
	vmovdqu	%ymm4, 224(%rsp)
	vmovdqu	%ymm5, 256(%rsp)
	vmovdqu	%xmm7, 313(%rsp)
.L3120:
	movq	%rcx, %rax
	movq	%rcx, %r9
	shrq	$2, %rax
	mulq	%rdi
	movl	%r15d, %eax
	movq	%rdx, %r11
	movq	%rdx, %r14
	shrq	$2, %r11
	andq	$-4, %r14
	addq	%r11, %r14
	leaq	(%r14,%r14,4), %r10
	movq	%rcx, %r14
	movq	%r11, %rcx
	salq	$2, %r10
	subq	%r10, %r9
	leal	-1(%r15), %r10d
	addq	%r9, %r9
	movzbl	129(%rsp,%r9), %r13d
	movzbl	128(%rsp,%r9), %r9d
	movb	%r13b, 51(%rsp,%rax)
	movb	%r9b, 51(%rsp,%r10)
	leal	-2(%r15), %r10d
	cmpq	$9999, %r14
	jbe	.L3294
	shrq	$4, %rdx
	movq	%r11, %r9
	movq	%rdx, %rax
	mulq	%rdi
	movq	%rdx, %r13
	andq	$-4, %rdx
	shrq	$2, %r13
	addq	%r13, %rdx
	movq	%r13, %rcx
	leaq	(%rdx,%rdx,4), %r14
	salq	$2, %r14
	subq	%r14, %r9
	movq	%r11, %r14
	addq	%r9, %r9
	movzbl	129(%rsp,%r9), %eax
	movzbl	128(%rsp,%r9), %edx
	movb	%al, 51(%rsp,%r10)
	leal	-3(%r15), %r10d
	subl	$4, %r15d
	movb	%dl, 51(%rsp,%r10)
	cmpq	$9999, %r11
	ja	.L3120
.L3294:
	cmpq	$999, %r14
	ja	.L3112
	vzeroupper
.L3110:
	addl	$48, %ecx
	jmp	.L3121
	.p2align 4,,10
	.p2align 3
.L3091:
	bsrq	%rcx, %r9
	movl	$2863311531, %r11d
	leal	3(%r9), %r10d
	imulq	%r11, %r10
	shrq	$33, %r10
	leal	-1(%r10), %r13d
	cmpq	$63, %rcx
	jbe	.L3123
.L3124:
	movq	%rcx, %rax
	movq	%rcx, %r15
	movl	%r13d, %r14d
	shrq	$6, %rcx
	shrq	$3, %rax
	leal	-1(%r13), %edx
	leal	-2(%r13), %esi
	andl	$7, %r15d
	andl	$7, %eax
	addl	$48, %r15d
	addl	$48, %eax
	movb	%r15b, 51(%rsp,%r14)
	movb	%al, 51(%rsp,%rdx)
	cmpq	$63, %rcx
	jbe	.L3123
	movq	%rcx, %r9
	leal	-3(%r13), %r11d
	leal	-4(%r13), %r14d
	movq	%rcx, %rdi
	shrq	$3, %r9
	andl	$7, %edi
	shrq	$6, %rcx
	andl	$7, %r9d
	addl	$48, %edi
	addl	$48, %r9d
	movb	%dil, 51(%rsp,%rsi)
	movb	%r9b, 51(%rsp,%r11)
	cmpq	$63, %rcx
	jbe	.L3123
	movq	%rcx, %rax
	leal	-5(%r13), %edx
	leal	-6(%r13), %esi
	movq	%rcx, %r15
	shrq	$3, %rax
	andl	$7, %r15d
	shrq	$6, %rcx
	andl	$7, %eax
	addl	$48, %r15d
	addl	$48, %eax
	movb	%r15b, 51(%rsp,%r14)
	movb	%al, 51(%rsp,%rdx)
	cmpq	$63, %rcx
	jbe	.L3123
	movq	%rcx, %r9
	movq	%rcx, %rdi
	leal	-7(%r13), %r11d
	shrq	$6, %rcx
	shrq	$3, %r9
	andl	$7, %edi
	subl	$8, %r13d
	andl	$7, %r9d
	addl	$48, %edi
	addl	$48, %r9d
	movb	%dil, 51(%rsp,%rsi)
	movb	%r9b, 51(%rsp,%r11)
	cmpq	$63, %rcx
	ja	.L3124
	.p2align 4,,10
	.p2align 3
.L3123:
	leal	48(%rcx), %edi
	cmpq	$7, %rcx
	jbe	.L3126
	movq	%rcx, %r13
	shrq	$3, %rcx
	andl	$7, %r13d
	movq	%rcx, %rdi
	addl	$48, %r13d
	addl	$48, %edi
	movb	%r13b, 52(%rsp)
.L3126:
	leaq	51(%rsp), %r15
	movl	%r10d, %ecx
	movl	$1, %r11d
	movl	$1, %eax
	leaq	(%r15,%rcx), %r14
	leaq	.LC37(%rip), %r9
	jmp	.L3122
	.p2align 4,,10
	.p2align 3
.L3302:
	movq	%rcx, %r15
	shrq	$4, %rcx
	andl	$15, %r15d
	movzbl	128(%rsp,%r15), %eax
	movb	%al, 52(%rsp)
	movzbl	128(%rsp,%rcx), %ecx
	jmp	.L3133
.L3305:
	vmovdqa	.LC26(%rip), %ymm8
	vmovdqa	.LC27(%rip), %ymm9
	movl	$2, %esi
	vmovdqa	.LC28(%rip), %ymm10
	vmovdqa	.LC29(%rip), %ymm11
	vmovdqa	.LC30(%rip), %ymm12
	vmovdqa	.LC31(%rip), %ymm13
	vmovdqu	%ymm8, 128(%rsp)
	vmovdqa	.LC32(%rip), %xmm14
	vmovdqu	%ymm9, 160(%rsp)
	vmovdqu	%ymm13, 288(%rsp)
	vmovdqu	%ymm10, 192(%rsp)
	vmovdqu	%ymm11, 224(%rsp)
	vmovdqu	%ymm12, 256(%rsp)
	vmovdqu	%xmm14, 313(%rsp)
	.p2align 4,,10
	.p2align 3
.L3112:
	addq	%rcx, %rcx
	movzbl	129(%rsp,%rcx), %r15d
	movzbl	128(%rsp,%rcx), %ecx
	movb	%r15b, 52(%rsp)
	vzeroupper
.L3121:
	movb	%cl, 51(%rsp)
	leaq	51(%rsp), %r15
	leaq	(%r15,%rsi), %r14
.L3119:
	movzbl	(%r12), %r13d
	movq	%r15, %rdx
	jmp	.L3108
	.p2align 4,,10
	.p2align 3
.L3153:
	leaq	.LC38(%rip), %r9
	leaq	52(%rsp), %r14
	leaq	51(%rsp), %r15
.L3129:
	movq	%r14, %rdi
	movq	%r15, %rdx
	subq	%r15, %rdi
	andl	$7, %edi
	je	.L3134
	cmpq	$1, %rdi
	je	.L3250
	cmpq	$2, %rdi
	je	.L3251
	cmpq	$3, %rdi
	je	.L3252
	cmpq	$4, %rdi
	je	.L3253
	cmpq	$5, %rdi
	je	.L3254
	cmpq	$6, %rdi
	je	.L3255
	movsbl	(%r15), %edi
	movq	%r8, 16(%rsp)
	movq	%r9, 24(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %r8
	movq	24(%rsp), %r9
	leaq	52(%rsp), %rdx
	movb	%al, (%r15)
.L3255:
	movsbl	(%rdx), %edi
	movq	%r8, 8(%rsp)
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	8(%rsp), %r8
	movq	16(%rsp), %r9
	movb	%al, (%rdx)
	addq	$1, %rdx
.L3254:
	movsbl	(%rdx), %edi
	movq	%r8, 8(%rsp)
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	8(%rsp), %r8
	movq	16(%rsp), %r9
	movb	%al, (%rdx)
	addq	$1, %rdx
.L3253:
	movsbl	(%rdx), %edi
	movq	%r8, 8(%rsp)
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	8(%rsp), %r8
	movq	16(%rsp), %r9
	movb	%al, (%rdx)
	addq	$1, %rdx
.L3252:
	movsbl	(%rdx), %edi
	movq	%r8, 8(%rsp)
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	8(%rsp), %r8
	movq	16(%rsp), %r9
	movb	%al, (%rdx)
	addq	$1, %rdx
.L3251:
	movsbl	(%rdx), %edi
	movq	%r8, 8(%rsp)
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	8(%rsp), %r8
	movq	16(%rsp), %r9
	movb	%al, (%rdx)
	addq	$1, %rdx
.L3250:
	movsbl	(%rdx), %edi
	movq	%r8, 8(%rsp)
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	16(%rsp), %r9
	movq	8(%rsp), %r8
	movb	%al, (%rdx)
	addq	$1, %rdx
	cmpq	%r14, %rdx
	je	.L3295
.L3134:
	movsbl	(%rdx), %edi
	movq	%r9, 16(%rsp)
	movq	%rdx, 24(%rsp)
	movq	%r8, 8(%rsp)
	call	toupper@PLT
	movq	24(%rsp), %r8
	movb	%al, (%r8)
	movsbl	1(%r8), %edi
	call	toupper@PLT
	movq	24(%rsp), %r10
	movb	%al, 1(%r10)
	movsbl	2(%r10), %edi
	call	toupper@PLT
	movq	24(%rsp), %rcx
	movb	%al, 2(%rcx)
	movsbl	3(%rcx), %edi
	call	toupper@PLT
	movq	24(%rsp), %rsi
	movb	%al, 3(%rsi)
	movsbl	4(%rsi), %edi
	call	toupper@PLT
	movq	24(%rsp), %r11
	movb	%al, 4(%r11)
	movsbl	5(%r11), %edi
	call	toupper@PLT
	movq	24(%rsp), %rdi
	movb	%al, 5(%rdi)
	movsbl	6(%rdi), %edi
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movb	%al, 6(%rdx)
	movsbl	7(%rdx), %edi
	call	toupper@PLT
	movq	24(%rsp), %rdx
	movq	16(%rsp), %r9
	movq	8(%rsp), %r8
	movb	%al, 7(%rdx)
	addq	$8, %rdx
	cmpq	%r14, %rdx
	jne	.L3134
.L3295:
	movl	$1, %r11d
	movl	$2, %eax
	jmp	.L3127
	.p2align 4,,10
	.p2align 3
.L3301:
	testq	%rsi, %rsi
	jne	.L3151
	movb	$48, 51(%rsp)
	movzbl	(%rdi), %r13d
	leaq	52(%rsp), %r14
	leaq	.LC39(%rip), %r9
	leaq	51(%rsp), %r15
	jmp	.L3128
	.p2align 4,,10
	.p2align 3
.L3303:
	testl	%edi, %edi
	jne	.L3129
	movl	$1, %r11d
	movl	$2, %eax
	movq	%r15, %r14
	jmp	.L3127
	.p2align 4,,10
	.p2align 3
.L3306:
	leal	5(%r13), %esi
	jmp	.L3116
	.p2align 4,,10
	.p2align 3
.L3307:
	leal	6(%r13), %esi
	jmp	.L3116
	.p2align 4,,10
	.p2align 3
.L3308:
	leal	7(%r13), %esi
	jmp	.L3116
	.p2align 4,,10
	.p2align 3
.L3152:
	leaq	.LC38(%rip), %r9
	jmp	.L3095
.L3149:
	leaq	115(%rsp), %r14
	leaq	51(%rsp), %r15
	jmp	.L3119
.L3151:
	leaq	.LC39(%rip), %r9
	jmp	.L3095
.L3146:
	movl	$1, %esi
	jmp	.L3110
.L3148:
	movl	$4, %esi
	movl	$3, %r15d
	jmp	.L3113
.L3147:
	movl	$3, %esi
	movl	$2, %r15d
	jmp	.L3113
.L3087:
	movq	344(%rsp), %rax
	subq	%fs:40, %rax
	je	.L3089
.L3298:
	call	__stack_chk_fail@PLT
.L3089:
	leaq	.LC40(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE13672:
	.size	_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.text._ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13674:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %r8
	movq	%rsi, %rcx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$344, %rsp
	movq	%fs:40, %rax
	movq	%rax, 328(%rsp)
	xorl	%eax, %eax
	movzbl	1(%rdi), %eax
	movl	%eax, %edx
	andl	$120, %edx
	cmpb	$56, %dl
	je	.L3515
	shrb	$3, %al
	andl	$15, %eax
	cmpb	$4, %al
	je	.L3314
	ja	.L3315
	cmpb	$1, %al
	jbe	.L3316
	cmpb	$16, %dl
	leaq	.LC35(%rip), %r9
	leaq	.LC36(%rip), %rdi
	cmovne	%rdi, %r9
	testq	%rsi, %rsi
	jne	.L3516
	leaq	36(%rsp), %r13
	leaq	35(%rsp), %r15
	movl	$48, %r12d
.L3321:
	movb	%r12b, 35(%rsp)
	movzbl	(%rbx), %r12d
	testb	$16, %r12b
	je	.L3373
.L3372:
	movq	$-2, %rdx
	movl	$2, %ecx
.L3325:
	addq	%r15, %rdx
	movl	%ecx, %esi
	testl	%ecx, %ecx
	je	.L3326
	xorl	%eax, %eax
	leal	-1(%rcx), %r14d
	movzbl	(%r9,%rax), %edi
	andl	$7, %r14d
	movb	%dil, (%rdx,%rax)
	movl	$1, %eax
	cmpl	%ecx, %eax
	jnb	.L3326
	testl	%r14d, %r14d
	je	.L3355
	cmpl	$1, %r14d
	je	.L3473
	cmpl	$2, %r14d
	je	.L3474
	cmpl	$3, %r14d
	je	.L3475
	cmpl	$4, %r14d
	je	.L3476
	cmpl	$5, %r14d
	je	.L3477
	cmpl	$6, %r14d
	je	.L3478
	movl	$1, %r10d
	movl	$2, %eax
	movzbl	(%r9,%r10), %r11d
	movb	%r11b, (%rdx,%r10)
.L3478:
	movl	%eax, %ecx
	addl	$1, %eax
	movzbl	(%r9,%rcx), %r14d
	movb	%r14b, (%rdx,%rcx)
.L3477:
	movl	%eax, %r10d
	addl	$1, %eax
	movzbl	(%r9,%r10), %edi
	movb	%dil, (%rdx,%r10)
.L3476:
	movl	%eax, %r11d
	addl	$1, %eax
	movzbl	(%r9,%r11), %ecx
	movb	%cl, (%rdx,%r11)
.L3475:
	movl	%eax, %r14d
	addl	$1, %eax
	movzbl	(%r9,%r14), %r10d
	movb	%r10b, (%rdx,%r14)
.L3474:
	movl	%eax, %r11d
	addl	$1, %eax
	movzbl	(%r9,%r11), %edi
	movb	%dil, (%rdx,%r11)
.L3473:
	movl	%eax, %ecx
	addl	$1, %eax
	movzbl	(%r9,%rcx), %r14d
	movb	%r14b, (%rdx,%rcx)
	cmpl	%esi, %eax
	jnb	.L3326
.L3355:
	movl	%eax, %r10d
	leal	1(%rax), %ecx
	leal	2(%rax), %r14d
	movzbl	(%r9,%r10), %r11d
	movzbl	(%r9,%rcx), %edi
	movb	%r11b, (%rdx,%r10)
	leal	3(%rax), %r11d
	movzbl	(%r9,%r14), %r10d
	movb	%dil, (%rdx,%rcx)
	movzbl	(%r9,%r11), %ecx
	movb	%r10b, (%rdx,%r14)
	leal	4(%rax), %r14d
	leal	5(%rax), %r10d
	movb	%cl, (%rdx,%r11)
	movzbl	(%r9,%r14), %edi
	leal	6(%rax), %ecx
	movzbl	(%r9,%r10), %r11d
	movb	%dil, (%rdx,%r14)
	movzbl	(%r9,%rcx), %r14d
	movb	%r11b, (%rdx,%r10)
	leal	7(%rax), %r10d
	addl	$8, %eax
	movzbl	(%r9,%r10), %edi
	movb	%r14b, (%rdx,%rcx)
	movb	%dil, (%rdx,%r10)
	cmpl	%esi, %eax
	jb	.L3355
	jmp	.L3326
	.p2align 4,,10
	.p2align 3
.L3515:
	cmpq	$127, %rsi
	ja	.L3311
	movb	%sil, 31(%rsp)
	movq	%rdi, %rcx
	leaq	31(%rsp), %rsi
	movq	%r8, %rdx
	movl	$1, %edi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
.L3312:
	movq	328(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L3514
	addq	$344, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L3316:
	.cfi_restore_state
	testq	%rsi, %rsi
	jne	.L3327
	movb	$48, 35(%rsp)
	leaq	36(%rsp), %r13
	leaq	35(%rsp), %r15
.L3328:
	movzbl	(%rbx), %r12d
	movq	%r15, %rdx
.L3326:
	shrb	$2, %r12b
	movl	$43, %esi
	andl	$3, %r12d
	cmpl	$1, %r12d
	je	.L3359
	cmpl	$3, %r12d
	je	.L3374
.L3358:
	movq	%r13, %rsi
	movq	%r15, %rcx
	movq	%rbx, %rdi
	subq	%rdx, %rsi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L3312
	.p2align 4,,10
	.p2align 3
.L3314:
	testq	%rsi, %rsi
	je	.L3367
	bsrq	%rsi, %rsi
	movl	$2863311531, %r10d
	leal	3(%rsi), %r9d
	imulq	%r10, %r9
	shrq	$33, %r9
	leal	-1(%r9), %r11d
	cmpq	$63, %rcx
	jbe	.L3342
.L3343:
	movq	%rcx, %r14
	movq	%rcx, %r13
	movl	%r11d, %r12d
	shrq	$6, %rcx
	shrq	$3, %r14
	leal	-1(%r11), %r15d
	leal	-2(%r11), %eax
	andl	$7, %r13d
	andl	$7, %r14d
	addl	$48, %r13d
	addl	$48, %r14d
	movb	%r13b, 35(%rsp,%r12)
	movb	%r14b, 35(%rsp,%r15)
	cmpq	$63, %rcx
	jbe	.L3342
	movq	%rcx, %rsi
	leal	-3(%r11), %edi
	leal	-4(%r11), %r10d
	movq	%rcx, %rdx
	shrq	$3, %rsi
	andl	$7, %edx
	shrq	$6, %rcx
	andl	$7, %esi
	addl	$48, %edx
	addl	$48, %esi
	movb	%dl, 35(%rsp,%rax)
	movb	%sil, 35(%rsp,%rdi)
	cmpq	$63, %rcx
	jbe	.L3342
	movq	%rcx, %r13
	leal	-5(%r11), %r14d
	leal	-6(%r11), %r15d
	movq	%rcx, %r12
	shrq	$3, %r13
	andl	$7, %r12d
	shrq	$6, %rcx
	andl	$7, %r13d
	addl	$48, %r12d
	addl	$48, %r13d
	movb	%r12b, 35(%rsp,%r10)
	movb	%r13b, 35(%rsp,%r14)
	cmpq	$63, %rcx
	jbe	.L3342
	movq	%rcx, %rdx
	movq	%rcx, %rax
	leal	-7(%r11), %esi
	shrq	$6, %rcx
	shrq	$3, %rdx
	andl	$7, %eax
	subl	$8, %r11d
	andl	$7, %edx
	addl	$48, %eax
	addl	$48, %edx
	movb	%al, 35(%rsp,%r15)
	movb	%dl, 35(%rsp,%rsi)
	cmpq	$63, %rcx
	ja	.L3343
	.p2align 4,,10
	.p2align 3
.L3342:
	leal	48(%rcx), %edi
	cmpq	$7, %rcx
	ja	.L3517
.L3345:
	movl	%r9d, %ecx
	leaq	35(%rsp), %r15
	movl	$1, %edx
	leaq	(%r15,%rcx), %r13
	leaq	.LC37(%rip), %r9
	movl	$1, %ecx
.L3341:
	movb	%dil, 35(%rsp)
	movzbl	(%rbx), %r12d
.L3346:
	testb	$16, %r12b
	je	.L3373
	testb	%dl, %dl
	jne	.L3518
.L3373:
	movq	%r15, %rdx
	jmp	.L3326
	.p2align 4,,10
	.p2align 3
.L3374:
	movl	$32, %esi
.L3359:
	movb	%sil, -1(%rdx)
	subq	$1, %rdx
	jmp	.L3358
	.p2align 4,,10
	.p2align 3
.L3315:
	cmpb	$40, %dl
	je	.L3519
	testq	%rsi, %rsi
	jne	.L3369
	movb	$48, 35(%rsp)
	movzbl	(%rdi), %r12d
	cmpb	$48, %dl
	je	.L3370
	leaq	36(%rsp), %r13
	leaq	.LC38(%rip), %r9
	leaq	35(%rsp), %r15
	jmp	.L3348
	.p2align 4,,10
	.p2align 3
.L3327:
	cmpq	$9, %rsi
	jbe	.L3363
	cmpq	$99, %rsi
	jbe	.L3520
	cmpq	$999, %rsi
	jbe	.L3364
	cmpq	$9999, %rsi
	jbe	.L3365
	movabsq	$3777893186295716171, %r9
	movq	%rsi, %rdx
	movl	$1, %esi
	jmp	.L3333
	.p2align 4,,10
	.p2align 3
.L3337:
	cmpq	$999999, %r13
	jbe	.L3521
	cmpq	$9999999, %r13
	jbe	.L3522
	cmpq	$99999999, %r13
	jbe	.L3523
.L3333:
	movq	%rdx, %rax
	movq	%rdx, %r13
	movl	%esi, %r15d
	addl	$4, %esi
	mulq	%r9
	shrq	$11, %rdx
	cmpq	$99999, %r13
	ja	.L3337
.L3335:
	cmpl	$64, %esi
	ja	.L3366
	leal	-1(%rsi), %r14d
.L3332:
	vmovdqa	.LC26(%rip), %ymm1
	vmovdqa	.LC27(%rip), %ymm2
	movabsq	$2951479051793528259, %r11
	vmovdqa	.LC28(%rip), %ymm3
	vmovdqa	.LC29(%rip), %ymm4
	vmovdqa	.LC30(%rip), %ymm5
	vmovdqa	.LC31(%rip), %ymm6
	vmovdqu	%ymm1, 112(%rsp)
	vmovdqa	.LC32(%rip), %xmm7
	vmovdqu	%ymm2, 144(%rsp)
	vmovdqu	%ymm6, 272(%rsp)
	vmovdqu	%ymm3, 176(%rsp)
	vmovdqu	%ymm4, 208(%rsp)
	vmovdqu	%ymm5, 240(%rsp)
	vmovdqu	%xmm7, 297(%rsp)
.L3339:
	movq	%rcx, %rax
	movq	%rcx, %r15
	movq	%rcx, %r13
	movl	%r14d, %r9d
	shrq	$2, %rax
	mulq	%r11
	movq	%rdx, %rdi
	movq	%rdx, %r12
	shrq	$2, %rdi
	andq	$-4, %r12
	addq	%rdi, %r12
	movq	%rdi, %rcx
	leaq	(%r12,%r12,4), %r10
	leal	-1(%r14), %r12d
	salq	$2, %r10
	subq	%r10, %r15
	addq	%r15, %r15
	movzbl	113(%rsp,%r15), %eax
	movzbl	112(%rsp,%r15), %r10d
	leal	-2(%r14), %r15d
	movb	%al, 35(%rsp,%r9)
	movb	%r10b, 35(%rsp,%r12)
	cmpq	$9999, %r13
	jbe	.L3510
	shrq	$4, %rdx
	movq	%rdi, %r12
	movq	%rdx, %rax
	mulq	%r11
	movq	%rdx, %r9
	andq	$-4, %rdx
	shrq	$2, %r9
	addq	%r9, %rdx
	movq	%r9, %rcx
	leaq	(%rdx,%rdx,4), %r13
	salq	$2, %r13
	subq	%r13, %r12
	movq	%rdi, %r13
	addq	%r12, %r12
	movzbl	113(%rsp,%r12), %r10d
	movzbl	112(%rsp,%r12), %eax
	movb	%r10b, 35(%rsp,%r15)
	leal	-3(%r14), %r15d
	subl	$4, %r14d
	movb	%al, 35(%rsp,%r15)
	cmpq	$9999, %rdi
	ja	.L3339
.L3510:
	cmpq	$999, %r13
	jbe	.L3512
.L3331:
	addq	%rcx, %rcx
	movzbl	113(%rsp,%rcx), %r14d
	movzbl	112(%rsp,%rcx), %ecx
	movb	%r14b, 36(%rsp)
	vzeroupper
.L3340:
	leaq	35(%rsp), %r15
	movb	%cl, 35(%rsp)
	leaq	(%r15,%rsi), %r13
	jmp	.L3328
	.p2align 4,,10
	.p2align 3
.L3367:
	movl	$48, %edi
	xorl	%edx, %edx
	leaq	36(%rsp), %r13
	xorl	%r9d, %r9d
	leaq	35(%rsp), %r15
	jmp	.L3341
	.p2align 4,,10
	.p2align 3
.L3516:
	bsrq	%rsi, %r13
	movl	$64, %r12d
	movl	$63, %r15d
	xorq	$63, %r13
	subl	%r13d, %r12d
	subl	%r13d, %r15d
	je	.L3324
	movl	%r15d, %r10d
	movl	$62, %r11d
	leaq	32(%rsp,%r10), %rax
	leaq	31(%rsp,%r10), %rdi
	subl	%r13d, %r11d
	subq	%r11, %rdi
	movq	%rax, %rdx
	subq	%rdi, %rdx
	andl	$7, %edx
	je	.L3323
	cmpq	$1, %rdx
	je	.L3457
	cmpq	$2, %rdx
	je	.L3458
	cmpq	$3, %rdx
	je	.L3459
	cmpq	$4, %rdx
	je	.L3460
	cmpq	$5, %rdx
	je	.L3461
	cmpq	$6, %rdx
	je	.L3462
	movl	%esi, %edx
	shrq	%rcx
	subq	$1, %rax
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3462:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3461:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3460:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3459:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3458:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
.L3457:
	movl	%ecx, %edx
	subq	$1, %rax
	shrq	%rcx
	andl	$1, %edx
	addl	$48, %edx
	movb	%dl, 4(%rax)
	cmpq	%rdi, %rax
	je	.L3324
.L3323:
	movl	%ecx, %edx
	movq	%rcx, %rsi
	movq	%rcx, %r14
	movq	%rcx, %r13
	andl	$1, %edx
	movq	%rcx, %r15
	movq	%rcx, %r10
	movq	%rcx, %r11
	shrq	%rsi
	addl	$48, %edx
	shrq	$2, %r14
	subq	$8, %rax
	movb	%dl, 11(%rax)
	shrq	$3, %r13
	movl	%ecx, %edx
	shrq	$4, %r15
	shrq	$5, %r10
	shrq	$6, %r11
	andl	$1, %esi
	andl	$1, %r14d
	andl	$1, %r13d
	andl	$1, %r15d
	andl	$1, %r10d
	andl	$1, %r11d
	shrb	$7, %dl
	addl	$48, %esi
	addl	$48, %r14d
	addl	$48, %r13d
	addl	$48, %r15d
	addl	$48, %r10d
	addl	$48, %r11d
	addl	$48, %edx
	movb	%sil, 10(%rax)
	shrq	$8, %rcx
	movb	%r14b, 9(%rax)
	movb	%r13b, 8(%rax)
	movb	%r15b, 7(%rax)
	movb	%r10b, 6(%rax)
	movb	%r11b, 5(%rax)
	movb	%dl, 4(%rax)
	cmpq	%rdi, %rax
	jne	.L3323
.L3324:
	movslq	%r12d, %rcx
	leaq	35(%rsp), %r15
	movl	$49, %r12d
	leaq	(%r15,%rcx), %r13
	jmp	.L3321
	.p2align 4,,10
	.p2align 3
.L3519:
	testq	%rsi, %rsi
	jne	.L3368
	movb	$48, 35(%rsp)
	movzbl	(%rdi), %r12d
	leaq	36(%rsp), %r13
	leaq	.LC39(%rip), %r9
	leaq	35(%rsp), %r15
	jmp	.L3348
	.p2align 4,,10
	.p2align 3
.L3369:
	leaq	.LC38(%rip), %r9
.L3347:
	bsrq	%rcx, %rdi
	vmovdqa	.LC25(%rip), %xmm0
	leal	4(%rdi), %r14d
	shrl	$2, %r14d
	vmovdqa	%xmm0, 112(%rsp)
	leal	-1(%r14), %r12d
	cmpq	$255, %rcx
	jbe	.L3350
.L3351:
	movq	%rcx, %rax
	movq	%rcx, %r10
	movl	%r12d, %r13d
	shrq	$8, %rcx
	shrq	$4, %rax
	andl	$15, %r10d
	leal	-1(%r12), %r11d
	andl	$15, %eax
	movzbl	112(%rsp,%r10), %r15d
	leal	-2(%r12), %edi
	movzbl	112(%rsp,%rax), %esi
	movb	%r15b, 35(%rsp,%r13)
	movb	%sil, 35(%rsp,%r11)
	cmpq	$255, %rcx
	jbe	.L3350
	movq	%rcx, %r15
	leal	-3(%r12), %eax
	leal	-4(%r12), %esi
	movq	%rcx, %r13
	shrq	$4, %r15
	andl	$15, %r13d
	shrq	$8, %rcx
	andl	$15, %r15d
	movzbl	112(%rsp,%r13), %r10d
	movzbl	112(%rsp,%r15), %r11d
	movb	%r10b, 35(%rsp,%rdi)
	movb	%r11b, 35(%rsp,%rax)
	cmpq	$255, %rcx
	jbe	.L3350
	movq	%rcx, %r15
	leal	-5(%r12), %r10d
	leal	-6(%r12), %r11d
	movq	%rcx, %rdi
	shrq	$4, %r15
	andl	$15, %edi
	shrq	$8, %rcx
	andl	$15, %r15d
	movzbl	112(%rsp,%rdi), %r13d
	movzbl	112(%rsp,%r15), %eax
	movb	%r13b, 35(%rsp,%rsi)
	movb	%al, 35(%rsp,%r10)
	cmpq	$255, %rcx
	jbe	.L3350
	movq	%rcx, %r13
	movq	%rcx, %rsi
	leal	-7(%r12), %r15d
	shrq	$8, %rcx
	shrq	$4, %r13
	andl	$15, %esi
	subl	$8, %r12d
	andl	$15, %r13d
	movzbl	112(%rsp,%rsi), %edi
	movzbl	112(%rsp,%r13), %r10d
	movb	%dil, 35(%rsp,%r11)
	movb	%r10b, 35(%rsp,%r15)
	cmpq	$255, %rcx
	ja	.L3351
	.p2align 4,,10
	.p2align 3
.L3350:
	cmpq	$15, %rcx
	ja	.L3524
	movzbl	112(%rsp,%rcx), %ecx
.L3353:
	leaq	35(%rsp), %r15
	movl	%r14d, %r11d
	movb	%cl, 35(%rsp)
	movzbl	(%rbx), %r12d
	leaq	(%r15,%r11), %r13
	cmpb	$48, %dl
	je	.L3525
.L3348:
	testb	$16, %r12b
	jne	.L3372
	jmp	.L3373
	.p2align 4,,10
	.p2align 3
.L3517:
	movq	%rcx, %r11
	movq	%rcx, %rdi
	andl	$7, %r11d
	shrq	$3, %rdi
	addl	$48, %r11d
	addl	$48, %edi
	movb	%r11b, 36(%rsp)
	jmp	.L3345
	.p2align 4,,10
	.p2align 3
.L3524:
	movq	%rcx, %r12
	shrq	$4, %rcx
	andl	$15, %r12d
	movzbl	112(%rsp,%r12), %eax
	movb	%al, 36(%rsp)
	movzbl	112(%rsp,%rcx), %ecx
	jmp	.L3353
	.p2align 4,,10
	.p2align 3
.L3512:
	vzeroupper
.L3329:
	addl	$48, %ecx
	jmp	.L3340
	.p2align 4,,10
	.p2align 3
.L3368:
	leaq	.LC39(%rip), %r9
	jmp	.L3347
	.p2align 4,,10
	.p2align 3
.L3370:
	leaq	.LC38(%rip), %r9
	leaq	36(%rsp), %r13
	leaq	35(%rsp), %r15
.L3349:
	movq	%r13, %rsi
	movq	%r15, %r14
	subq	%r15, %rsi
	andl	$7, %esi
	je	.L3354
	cmpq	$1, %rsi
	je	.L3466
	cmpq	$2, %rsi
	je	.L3467
	cmpq	$3, %rsi
	je	.L3468
	cmpq	$4, %rsi
	je	.L3469
	cmpq	$5, %rsi
	je	.L3470
	cmpq	$6, %rsi
	je	.L3471
	movsbl	(%r15), %edi
	movq	%r8, (%rsp)
	leaq	36(%rsp), %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r8
	movq	8(%rsp), %r9
	movb	%al, (%r15)
.L3471:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$1, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r8
	movq	8(%rsp), %r9
	movb	%al, -1(%r14)
.L3470:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$1, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r8
	movq	8(%rsp), %r9
	movb	%al, -1(%r14)
.L3469:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$1, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r8
	movq	8(%rsp), %r9
	movb	%al, -1(%r14)
.L3468:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$1, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r8
	movq	8(%rsp), %r9
	movb	%al, -1(%r14)
.L3467:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$1, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	(%rsp), %r8
	movq	8(%rsp), %r9
	movb	%al, -1(%r14)
.L3466:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$1, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movq	8(%rsp), %r9
	movq	(%rsp), %r8
	movb	%al, -1(%r14)
	cmpq	%r13, %r14
	je	.L3511
.L3354:
	movsbl	(%r14), %edi
	movq	%r8, (%rsp)
	addq	$8, %r14
	movq	%r9, 8(%rsp)
	call	toupper@PLT
	movsbl	-7(%r14), %edi
	movb	%al, -8(%r14)
	call	toupper@PLT
	movsbl	-6(%r14), %edi
	movb	%al, -7(%r14)
	call	toupper@PLT
	movsbl	-5(%r14), %edi
	movb	%al, -6(%r14)
	call	toupper@PLT
	movsbl	-4(%r14), %edi
	movb	%al, -5(%r14)
	call	toupper@PLT
	movsbl	-3(%r14), %edi
	movb	%al, -4(%r14)
	call	toupper@PLT
	movsbl	-2(%r14), %edi
	movb	%al, -3(%r14)
	call	toupper@PLT
	movsbl	-1(%r14), %edi
	movb	%al, -2(%r14)
	call	toupper@PLT
	movq	8(%rsp), %r9
	movq	(%rsp), %r8
	movb	%al, -1(%r14)
	cmpq	%r13, %r14
	jne	.L3354
.L3511:
	movl	$1, %edx
	movl	$2, %ecx
	jmp	.L3346
	.p2align 4,,10
	.p2align 3
.L3525:
	testl	%r14d, %r14d
	jne	.L3349
	movl	$1, %edx
	movl	$2, %ecx
	movq	%r15, %r13
	jmp	.L3346
	.p2align 4,,10
	.p2align 3
.L3521:
	leal	5(%r15), %esi
	jmp	.L3335
	.p2align 4,,10
	.p2align 3
.L3522:
	leal	6(%r15), %esi
	jmp	.L3335
	.p2align 4,,10
	.p2align 3
.L3523:
	leal	7(%r15), %esi
	jmp	.L3335
	.p2align 4,,10
	.p2align 3
.L3518:
	movq	%rcx, %rdx
	negq	%rdx
	jmp	.L3325
.L3366:
	leaq	99(%rsp), %r13
	leaq	35(%rsp), %r15
	jmp	.L3328
.L3363:
	movl	$1, %esi
	jmp	.L3329
.L3520:
	vmovdqa	.LC26(%rip), %ymm8
	vmovdqa	.LC27(%rip), %ymm9
	movl	$2, %esi
	vmovdqa	.LC28(%rip), %ymm10
	vmovdqa	.LC29(%rip), %ymm11
	vmovdqa	.LC30(%rip), %ymm12
	vmovdqa	.LC31(%rip), %ymm13
	vmovdqu	%ymm8, 112(%rsp)
	vmovdqa	.LC32(%rip), %xmm14
	vmovdqu	%ymm9, 144(%rsp)
	vmovdqu	%ymm13, 272(%rsp)
	vmovdqu	%ymm10, 176(%rsp)
	vmovdqu	%ymm11, 208(%rsp)
	vmovdqu	%ymm12, 240(%rsp)
	vmovdqu	%xmm14, 297(%rsp)
	jmp	.L3331
.L3365:
	movl	$4, %esi
	movl	$3, %r14d
	jmp	.L3332
.L3364:
	movl	$3, %esi
	movl	$2, %r14d
	jmp	.L3332
.L3311:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	je	.L3313
.L3514:
	call	__stack_chk_fail@PLT
.L3313:
	leaq	.LC40(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE13674:
	.size	_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.globl	__udivti3
	.section	.text._ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13688:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	movq	%rdi, %r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$456, %rsp
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdx, 56(%rsp)
	movzbl	1(%rdi), %edx
	movq	%rsi, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 440(%rsp)
	xorl	%eax, %eax
	movl	%edx, %eax
	andl	$120, %eax
	cmpb	$56, %al
	je	.L3754
	movq	48(%rsp), %rbx
	movq	56(%rsp), %rsi
	shrb	$3, %dl
	andl	$15, %edx
	movq	%rbx, %r12
	movq	%rsi, %r13
	testq	%rsi, %rsi
	js	.L3755
	cmpb	$4, %dl
	je	.L3537
	ja	.L3538
	cmpb	$1, %dl
	jbe	.L3539
	cmpb	$16, %al
	leaq	.LC35(%rip), %rbx
	leaq	.LC36(%rip), %r8
	movq	%r12, %r9
	cmovne	%r8, %rbx
	orq	%rsi, %r9
	jne	.L3535
	movl	$48, %r10d
	leaq	84(%rsp), %r15
	leaq	83(%rsp), %r13
.L3544:
	movzbl	(%r14), %r12d
	movb	%r10b, 83(%rsp)
	testb	$16, %r12b
	je	.L3603
.L3602:
	movq	$-2, %rdx
	movl	$2, %eax
.L3549:
	addq	%r13, %rdx
	movl	%eax, %r8d
	testl	%eax, %eax
	je	.L3550
	xorl	%r10d, %r10d
	leal	-1(%rax), %r11d
	movl	$1, %r9d
	movzbl	(%rbx,%r10), %edi
	andl	$7, %r11d
	movb	%dil, (%rdx,%r10)
	cmpl	%eax, %r9d
	jnb	.L3550
	testl	%r11d, %r11d
	je	.L3581
	cmpl	$1, %r11d
	je	.L3703
	cmpl	$2, %r11d
	je	.L3704
	cmpl	$3, %r11d
	je	.L3705
	cmpl	$4, %r11d
	je	.L3706
	cmpl	$5, %r11d
	je	.L3707
	cmpl	$6, %r11d
	je	.L3708
	movl	$1, %esi
	movl	$2, %r9d
	movzbl	(%rbx,%rsi), %ecx
	movb	%cl, (%rdx,%rsi)
.L3708:
	movl	%r9d, %eax
	addl	$1, %r9d
	movzbl	(%rbx,%rax), %r11d
	movb	%r11b, (%rdx,%rax)
.L3707:
	movl	%r9d, %r10d
	addl	$1, %r9d
	movzbl	(%rbx,%r10), %edi
	movb	%dil, (%rdx,%r10)
.L3706:
	movl	%r9d, %esi
	addl	$1, %r9d
	movzbl	(%rbx,%rsi), %ecx
	movb	%cl, (%rdx,%rsi)
.L3705:
	movl	%r9d, %eax
	addl	$1, %r9d
	movzbl	(%rbx,%rax), %r11d
	movb	%r11b, (%rdx,%rax)
.L3704:
	movl	%r9d, %r10d
	addl	$1, %r9d
	movzbl	(%rbx,%r10), %edi
	movb	%dil, (%rdx,%r10)
.L3703:
	movl	%r9d, %esi
	addl	$1, %r9d
	movzbl	(%rbx,%rsi), %ecx
	movb	%cl, (%rdx,%rsi)
	cmpl	%r8d, %r9d
	jnb	.L3550
.L3581:
	movl	%r9d, %eax
	leal	1(%r9), %r10d
	leal	2(%r9), %esi
	movzbl	(%rbx,%rax), %r11d
	movzbl	(%rbx,%r10), %edi
	movzbl	(%rbx,%rsi), %ecx
	movb	%r11b, (%rdx,%rax)
	leal	3(%r9), %eax
	movb	%dil, (%rdx,%r10)
	leal	4(%r9), %r10d
	movzbl	(%rbx,%rax), %r11d
	movzbl	(%rbx,%r10), %edi
	movb	%cl, (%rdx,%rsi)
	leal	5(%r9), %esi
	movb	%r11b, (%rdx,%rax)
	leal	6(%r9), %eax
	movzbl	(%rbx,%rsi), %ecx
	movb	%dil, (%rdx,%r10)
	leal	7(%r9), %r10d
	movzbl	(%rbx,%rax), %r11d
	addl	$8, %r9d
	movzbl	(%rbx,%r10), %edi
	movb	%cl, (%rdx,%rsi)
	movb	%r11b, (%rdx,%rax)
	movb	%dil, (%rdx,%r10)
	cmpl	%r8d, %r9d
	jb	.L3581
	.p2align 4,,10
	.p2align 3
.L3550:
	shrb	$2, %r12b
	leaq	-1(%rdx), %rsi
	andl	$3, %r12d
	cmpq	$0, 56(%rsp)
	jns	.L3551
	movb	$45, -1(%rdx)
	movq	%rsi, %rdx
.L3583:
	movq	%r15, %rsi
	movq	%r13, %rcx
	movq	40(%rsp), %r8
	movq	%r14, %rdi
	subq	%rdx, %rsi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L3529
	.p2align 4,,10
	.p2align 3
.L3755:
	negq	%rbx
	adcq	$0, %rsi
	movq	%rbx, %r12
	negq	%rsi
	movq	%rsi, %r13
	cmpb	$4, %dl
	je	.L3532
	ja	.L3533
	cmpb	$1, %dl
	jbe	.L3534
	cmpb	$16, %al
	leaq	.LC35(%rip), %rbx
	leaq	.LC36(%rip), %rdx
	cmovne	%rdx, %rbx
.L3535:
	testq	%r13, %r13
	jne	.L3756
	bsrq	%r12, %r8
	movl	$128, %eax
	movl	$127, %r11d
	xorq	$63, %r8
	addl	$64, %r8d
	subl	%r8d, %eax
	subl	%r8d, %r11d
	je	.L3591
.L3546:
	movl	%r11d, %esi
	subl	$1, %r11d
	leaq	80(%rsp,%rsi), %r15
	leaq	79(%rsp,%rsi), %rdx
	subq	%r11, %rdx
	movq	%r15, %rcx
	subq	%rdx, %rcx
	andl	$7, %ecx
	je	.L3745
	cmpq	$1, %rcx
	je	.L3688
	cmpq	$2, %rcx
	je	.L3689
	cmpq	$3, %rcx
	je	.L3690
	cmpq	$4, %rcx
	je	.L3691
	cmpq	$5, %rcx
	je	.L3692
	cmpq	$6, %rcx
	je	.L3693
	movl	%r12d, %edi
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %edi
	shrq	%r13
	addl	$48, %edi
	movb	%dil, 4(%r15)
.L3693:
	movl	%r12d, %r9d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r9d
	shrq	%r13
	addl	$48, %r9d
	movb	%r9b, 4(%r15)
.L3692:
	movl	%r12d, %r8d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r8d
	shrq	%r13
	addl	$48, %r8d
	movb	%r8b, 4(%r15)
.L3691:
	movl	%r12d, %r11d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r11d
	shrq	%r13
	addl	$48, %r11d
	movb	%r11b, 4(%r15)
.L3690:
	movl	%r12d, %r10d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r10d
	shrq	%r13
	addl	$48, %r10d
	movb	%r10b, 4(%r15)
.L3689:
	movl	%r12d, %esi
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %esi
	shrq	%r13
	addl	$48, %esi
	movb	%sil, 4(%r15)
.L3688:
	movl	%r12d, %ecx
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %ecx
	shrq	%r13
	addl	$48, %ecx
	movb	%cl, 4(%r15)
	cmpq	%rdx, %r15
	je	.L3547
.L3745:
	movq	%rbx, %rdi
.L3548:
	movq	%r12, %r9
	movq	%r12, %r11
	movq	%r12, %rsi
	movq	%r12, %r8
	shrdq	$1, %r13, %r9
	movq	%r12, %r10
	movq	%r12, %rcx
	movl	%r12d, %ebx
	andl	$1, %r9d
	shrdq	$2, %r13, %r11
	andl	$1, %ebx
	subq	$8, %r15
	addl	$48, %r9d
	shrdq	$3, %r13, %rsi
	andl	$1, %r11d
	addl	$48, %ebx
	movb	%r9b, 10(%r15)
	movq	%r12, %r9
	shrdq	$5, %r13, %r8
	andl	$1, %esi
	shrdq	$4, %r13, %r9
	andl	$1, %r8d
	addl	$48, %r11d
	addl	$48, %esi
	shrdq	$6, %r13, %r10
	andl	$1, %r9d
	addl	$48, %r8d
	movb	%bl, 11(%r15)
	shrdq	$7, %r13, %rcx
	andl	$1, %r10d
	addl	$48, %r9d
	shrdq	$8, %r13, %r12
	andl	$1, %ecx
	addl	$48, %r10d
	movb	%r11b, 9(%r15)
	shrq	$8, %r13
	addl	$48, %ecx
	movb	%sil, 8(%r15)
	movb	%r9b, 7(%r15)
	movb	%r8b, 6(%r15)
	movb	%r10b, 5(%r15)
	movb	%cl, 4(%r15)
	cmpq	%rdx, %r15
	jne	.L3548
	movq	%rdi, %rbx
.L3547:
	leaq	83(%rsp), %r13
	cltq
	movl	$49, %r10d
	leaq	0(%r13,%rax), %r15
	jmp	.L3544
	.p2align 4,,10
	.p2align 3
.L3754:
	movq	48(%rsp), %rbx
	movq	56(%rsp), %r11
	movl	$127, %eax
	movl	$0, %r10d
	cmpq	%rbx, %rax
	sbbq	%r11, %r10
	jl	.L3528
	movq	%rcx, %rdx
	leaq	79(%rsp), %rsi
	movq	%rdi, %rcx
	movl	$1, %edi
	movb	%bl, 79(%rsp)
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
.L3529:
	movq	440(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L3753
	addq	$456, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L3538:
	.cfi_restore_state
	cmpb	$40, %al
	je	.L3757
	orq	%rsi, %rbx
	jne	.L3599
	movb	$48, 83(%rsp)
	movzbl	(%rdi), %r12d
	cmpb	$48, %al
	je	.L3600
	leaq	84(%rsp), %r15
	leaq	.LC38(%rip), %rbx
	leaq	83(%rsp), %r13
	jmp	.L3572
	.p2align 4,,10
	.p2align 3
.L3533:
	cmpb	$40, %al
	leaq	.LC39(%rip), %rbx
	leaq	.LC38(%rip), %r15
	cmovne	%r15, %rbx
.L3536:
	testq	%r13, %r13
	jne	.L3758
	bsrq	%r12, %r10
	vmovdqa	.LC25(%rip), %xmm0
	movl	$255, %ecx
	addl	$4, %r10d
	vmovdqa	%xmm0, 224(%rsp)
	shrl	$2, %r10d
	leal	-1(%r10), %edx
	cmpq	%r12, %rcx
	jnb	.L3759
.L3575:
	leaq	224(%rsp), %r15
	movl	$255, %r9d
	xorl	%r8d, %r8d
.L3577:
	movq	%r12, %rcx
	movq	%r12, %rsi
	movl	%edx, %edi
	shrdq	$8, %r13, %r12
	shrdq	$4, %r13, %rcx
	andl	$15, %esi
	shrq	$8, %r13
	andl	$15, %ecx
	addq	%r15, %rsi
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rcx
	movzbl	(%rsi), %r11d
	xorl	%esi, %esi
	addq	%r15, %rcx
	cmpq	%r12, %r9
	movq	%rsi, 24(%rsp)
	leal	-2(%rdx), %esi
	movb	%r11b, 83(%rsp,%rdi)
	movzbl	(%rcx), %edi
	leal	-1(%rdx), %r11d
	movb	%dil, 83(%rsp,%r11)
	movq	%r8, %r11
	sbbq	%r13, %r11
	jnc	.L3576
	movq	%r12, %rdi
	movq	%r12, %r11
	movl	%esi, %ecx
	shrdq	$8, %r13, %r12
	andl	$15, %edi
	shrdq	$4, %r13, %r11
	addq	%r15, %rdi
	andl	$15, %r11d
	shrq	$8, %r13
	movzbl	(%rdi), %esi
	movq	%r11, 16(%rsp)
	xorl	%edi, %edi
	leal	-3(%rdx), %r11d
	movq	%rdi, 24(%rsp)
	movq	%r8, %rdi
	movb	%sil, 83(%rsp,%rcx)
	movq	16(%rsp), %rsi
	addq	%r15, %rsi
	cmpq	%r12, %r9
	movzbl	(%rsi), %ecx
	sbbq	%r13, %rdi
	movb	%cl, 83(%rsp,%r11)
	leal	-4(%rdx), %r11d
	jnc	.L3576
	movq	%r12, %rsi
	movl	%r11d, %ecx
	andl	$15, %esi
	addq	%r15, %rsi
	movzbl	(%rsi), %r11d
	xorl	%esi, %esi
	movq	%rsi, 24(%rsp)
	leal	-6(%rdx), %esi
	movb	%r11b, 83(%rsp,%rcx)
	movq	%r12, %rcx
	leal	-5(%rdx), %r11d
	shrdq	$8, %r13, %r12
	shrdq	$4, %r13, %rcx
	shrq	$8, %r13
	andl	$15, %ecx
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rcx
	addq	%r15, %rcx
	cmpq	%r12, %r9
	movzbl	(%rcx), %edi
	movb	%dil, 83(%rsp,%r11)
	movq	%r8, %r11
	sbbq	%r13, %r11
	jnc	.L3576
	movq	%r12, %rdi
	movq	%r12, %r11
	movl	%esi, %ecx
	shrdq	$8, %r13, %r12
	andl	$15, %edi
	shrdq	$4, %r13, %r11
	addq	%r15, %rdi
	andl	$15, %r11d
	shrq	$8, %r13
	movzbl	(%rdi), %esi
	movq	%r11, 16(%rsp)
	xorl	%edi, %edi
	leal	-7(%rdx), %r11d
	movq	%rdi, 24(%rsp)
	subl	$8, %edx
	movq	%r8, %rdi
	movb	%sil, 83(%rsp,%rcx)
	movq	16(%rsp), %rsi
	addq	%r15, %rsi
	cmpq	%r12, %r9
	movzbl	(%rsi), %ecx
	sbbq	%r13, %rdi
	movb	%cl, 83(%rsp,%r11)
	jc	.L3577
	.p2align 4,,10
	.p2align 3
.L3576:
	movl	$15, %edx
	movl	$0, %r9d
	cmpq	%r12, %rdx
	sbbq	%r13, %r9
	jc	.L3760
	addq	%r12, %r15
	movzbl	(%r15), %r13d
.L3579:
	movb	%r13b, 83(%rsp)
	movl	%r10d, %r15d
	leaq	83(%rsp), %r13
	movzbl	(%r14), %r12d
	addq	%r13, %r15
	cmpb	$48, %al
	je	.L3761
.L3572:
	testb	$16, %r12b
	jne	.L3602
	.p2align 4,,10
	.p2align 3
.L3603:
	movq	%r13, %rdx
	jmp	.L3550
	.p2align 4,,10
	.p2align 3
.L3539:
	orq	%rsi, %rbx
	jne	.L3534
	movzbl	(%rdi), %r12d
	movb	$48, 83(%rsp)
	leaq	83(%rsp), %r13
	leaq	84(%rsp), %r15
	movq	%r13, %rdx
	leaq	82(%rsp), %rsi
	shrb	$2, %r12b
	andl	$3, %r12d
.L3551:
	movzbl	%r12b, %ecx
	cmpl	$1, %ecx
	je	.L3762
	cmpl	$3, %ecx
	jne	.L3583
	movb	$32, -1(%rdx)
.L3586:
	movq	%rsi, %rdx
	jmp	.L3583
	.p2align 4,,10
	.p2align 3
.L3762:
	movb	$43, -1(%rdx)
	jmp	.L3586
	.p2align 4,,10
	.p2align 3
.L3537:
	orq	%rsi, %rbx
	jne	.L3532
	xorl	%r10d, %r10d
	xorl	%eax, %eax
	xorl	%ebx, %ebx
	movl	$48, %ecx
	leaq	84(%rsp), %r15
	leaq	83(%rsp), %r13
.L3564:
	movb	%cl, 83(%rsp)
	movzbl	(%r14), %r12d
.L3571:
	testb	$16, %r12b
	je	.L3603
	testb	%r10b, %r10b
	je	.L3603
	movq	%rax, %rdx
	negq	%rdx
	jmp	.L3549
	.p2align 4,,10
	.p2align 3
.L3534:
	xorl	%r8d, %r8d
	movl	$9, %edx
	cmpq	%r12, %rdx
	movq	%r8, %rbx
	sbbq	%r13, %rbx
	jnc	.L3593
	movl	$99, %r9d
	movq	%r8, %r15
	cmpq	%r12, %r9
	sbbq	%r13, %r15
	jnc	.L3763
	movl	$999, %esi
	movq	%r8, %rcx
	cmpq	%r12, %rsi
	sbbq	%r13, %rcx
	jnc	.L3594
	movl	$9999, %eax
	cmpq	%r12, %rax
	sbbq	%r13, %r8
	jnc	.L3595
	movl	$1, %r11d
	movq	%r13, 24(%rsp)
	movq	%r13, %r8
	xorl	%ebx, %ebx
	movq	%r12, 16(%rsp)
	movq	%r12, %r10
	movl	%r11d, %r13d
	movq	%r14, 32(%rsp)
	jmp	.L3556
	.p2align 4,,10
	.p2align 3
.L3560:
	movl	$999999, %r15d
	movq	%rbx, %rsi
	cmpq	%r12, %r15
	sbbq	%r14, %rsi
	jnc	.L3764
	movl	$9999999, %ecx
	movq	%rbx, %rax
	cmpq	%r12, %rcx
	sbbq	%r14, %rax
	jnc	.L3765
	movl	$99999999, %r11d
	cmpq	%r12, %r11
	movq	%rbx, %r12
	sbbq	%r14, %r12
	jnc	.L3766
.L3556:
	movq	%r10, %rdi
	xorl	%ecx, %ecx
	movq	%r8, %rsi
	movl	$10000, %edx
	movq	%r10, %r12
	movq	%r8, %r14
	call	__udivti3@PLT
	movl	$99999, %edi
	movq	%rbx, %r9
	movq	%rdx, %r8
	movl	%r13d, %edx
	addl	$4, %r13d
	cmpq	%r12, %rdi
	sbbq	%r14, %r9
	movq	%rax, %r10
	jc	.L3560
	movl	%r13d, %r10d
	movq	16(%rsp), %r12
	movq	24(%rsp), %r13
	movq	32(%rsp), %r14
.L3558:
	cmpl	$128, %r10d
	ja	.L3596
	leal	-1(%r10), %r8d
	movl	%r10d, %edx
.L3555:
	movabsq	$-8116567392432202711, %rdi
	vmovdqa	.LC26(%rip), %ymm2
	vmovdqa	.LC27(%rip), %ymm3
	movq	%rdx, 32(%rsp)
	vmovdqa	.LC28(%rip), %ymm4
	vmovdqa	.LC29(%rip), %ymm5
	movq	%r14, 8(%rsp)
	leaq	224(%rsp), %r10
	vmovdqa	.LC30(%rip), %ymm6
	vmovdqa	.LC31(%rip), %ymm7
	movq	%rdi, 16(%rsp)
	xorl	%r9d, %r9d
	vmovdqa	.LC32(%rip), %xmm8
	vmovdqu	%ymm2, 224(%rsp)
	movabsq	$1152921504606846975, %r11
	vmovdqu	%ymm7, 384(%rsp)
	vmovdqu	%ymm3, 256(%rsp)
	vmovdqu	%ymm4, 288(%rsp)
	vmovdqu	%ymm5, 320(%rsp)
	vmovdqu	%ymm6, 352(%rsp)
	vmovdqu	%xmm8, 409(%rsp)
	.p2align 4,,10
	.p2align 3
.L3562:
	movq	%r12, %rcx
	movq	%r12, %rax
	movq	16(%rsp), %rdi
	xorl	%r15d, %r15d
	shrdq	$60, %r13, %rcx
	andq	%r11, %rax
	movl	$25, %ebx
	andq	%r11, %rcx
	addq	%rax, %rcx
	movq	%r13, %rax
	shrq	$56, %rax
	addq	%rax, %rcx
	movabsq	$5165088340638674453, %rax
	mulq	%rcx
	movq	%rcx, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$4, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r13, %rdx
	leaq	(%rax,%rax,4), %rax
	subq	%rax, %rcx
	movq	%r12, %rax
	subq	%rcx, %rax
	movq	%rcx, %r14
	movq	%rdi, %rcx
	sbbq	%r15, %rdx
	imulq	%rdx, %rcx
	movabsq	$2951479051793528258, %rdx
	imulq	%rax, %rdx
	addq	%rdx, %rcx
	mulq	%rdi
	movq	%rax, %rsi
	andl	$3, %eax
	movq	%rdx, %rdi
	mulq	%rbx
	addq	%rcx, %rdi
	movq	%r13, %rdx
	movq	%r12, %rcx
	addq	%r14, %rax
	shrdq	$2, %rdi, %rsi
	addq	%rax, %rax
	shrq	$2, %rdi
	movq	%rsi, %r12
	movl	%r8d, %esi
	movq	%rdi, %r13
	leaq	(%r10,%rax), %rdi
	leaq	(%rax,%r10), %rax
	movzbl	1(%rdi), %edi
	movzbl	(%rax), %eax
	movb	%dil, 83(%rsp,%rsi)
	leal	-1(%r8), %esi
	subl	$2, %r8d
	movb	%al, 83(%rsp,%rsi)
	movl	$9999, %eax
	cmpq	%rcx, %rax
	movq	%r9, %rax
	sbbq	%rdx, %rax
	jc	.L3562
	movl	$999, %ebx
	movq	32(%rsp), %r8
	movq	8(%rsp), %r14
	cmpq	%rcx, %rbx
	sbbq	%rdx, %r9
	jnc	.L3751
.L3554:
	addq	%r12, %r12
	leaq	(%r10,%r12), %r9
	addq	%r12, %r10
	movzbl	1(%r9), %r13d
	movzbl	(%r10), %r15d
	movb	%r13b, 84(%rsp)
	vzeroupper
.L3563:
	leaq	83(%rsp), %r13
	movb	%r15b, 83(%rsp)
	leaq	0(%r13,%r8), %r15
.L3561:
	movzbl	(%r14), %r12d
	movq	%r13, %rdx
	jmp	.L3550
	.p2align 4,,10
	.p2align 3
.L3532:
	testq	%r13, %r13
	jne	.L3767
	bsrq	%r12, %r15
	movl	$2863311531, %eax
	movl	$63, %ecx
	addl	$3, %r15d
	imulq	%rax, %r15
	shrq	$33, %r15
	leal	-1(%r15), %r9d
	cmpq	%r12, %rcx
	jnb	.L3567
.L3566:
	movl	$63, %edx
	xorl	%r10d, %r10d
.L3568:
	movq	%r12, %rsi
	movq	%r12, %rbx
	shrdq	$6, %r13, %r12
	movq	%r10, %rdi
	shrdq	$3, %r13, %rsi
	andl	$7, %ebx
	shrq	$6, %r13
	movl	%r9d, %r8d
	andl	$7, %esi
	leal	-1(%r9), %eax
	leal	-2(%r9), %ecx
	addl	$48, %ebx
	addl	$48, %esi
	cmpq	%r12, %rdx
	movb	%bl, 83(%rsp,%r8)
	sbbq	%r13, %rdi
	movb	%sil, 83(%rsp,%rax)
	jnc	.L3567
	movq	%r12, %r8
	leal	-3(%r9), %r11d
	leal	-4(%r9), %eax
	movq	%r12, %rsi
	shrdq	$3, %r13, %rsi
	andl	$7, %r8d
	shrdq	$6, %r13, %r12
	addl	$48, %r8d
	andl	$7, %esi
	shrq	$6, %r13
	movb	%r8b, 83(%rsp,%rcx)
	addl	$48, %esi
	movq	%r10, %rcx
	cmpq	%r12, %rdx
	sbbq	%r13, %rcx
	movb	%sil, 83(%rsp,%r11)
	jnc	.L3567
	movq	%r12, %rdi
	leal	-5(%r9), %ebx
	leal	-6(%r9), %r11d
	movq	%r12, %rsi
	shrdq	$3, %r13, %rsi
	andl	$7, %edi
	shrdq	$6, %r13, %r12
	addl	$48, %edi
	andl	$7, %esi
	shrq	$6, %r13
	movb	%dil, 83(%rsp,%rax)
	addl	$48, %esi
	movq	%r10, %rax
	cmpq	%r12, %rdx
	sbbq	%r13, %rax
	movb	%sil, 83(%rsp,%rbx)
	jnc	.L3567
	movq	%r12, %rdi
	movq	%r12, %rcx
	shrdq	$6, %r13, %r12
	leal	-7(%r9), %r8d
	shrdq	$3, %r13, %rdi
	andl	$7, %ecx
	shrq	$6, %r13
	subl	$8, %r9d
	andl	$7, %edi
	addl	$48, %ecx
	movq	%r10, %rbx
	addl	$48, %edi
	cmpq	%r12, %rdx
	movb	%cl, 83(%rsp,%r11)
	sbbq	%r13, %rbx
	movb	%dil, 83(%rsp,%r8)
	jc	.L3568
	.p2align 4,,10
	.p2align 3
.L3567:
	movl	$7, %r9d
	movl	$0, %edx
	leal	48(%r12), %ecx
	cmpq	%r12, %r9
	sbbq	%r13, %rdx
	jnc	.L3570
	movq	%r12, %r10
	shrdq	$3, %r13, %r12
	andl	$7, %r10d
	leal	48(%r12), %ecx
	addl	$48, %r10d
	movb	%r10b, 84(%rsp)
.L3570:
	leaq	83(%rsp), %r13
	movl	%r15d, %r12d
	movl	$1, %r10d
	movl	$1, %eax
	leaq	0(%r13,%r12), %r15
	leaq	.LC37(%rip), %rbx
	jmp	.L3564
	.p2align 4,,10
	.p2align 3
.L3760:
	movq	%r12, %r8
	shrdq	$4, %r13, %r12
	andl	$15, %r8d
	leaq	(%r15,%r12), %r12
	addq	%r15, %r8
	movzbl	(%r12), %r13d
	movzbl	(%r8), %r11d
	movb	%r11b, 84(%rsp)
	jmp	.L3579
	.p2align 4,,10
	.p2align 3
.L3600:
	leaq	.LC38(%rip), %rbx
	leaq	84(%rsp), %r15
	leaq	83(%rsp), %r13
.L3573:
	movq	%r15, %rax
	movq	%r13, %rsi
	subq	%r13, %rax
	andl	$7, %eax
	je	.L3746
	cmpq	$1, %rax
	je	.L3696
	cmpq	$2, %rax
	je	.L3697
	cmpq	$3, %rax
	je	.L3698
	cmpq	$4, %rax
	je	.L3699
	cmpq	$5, %rax
	je	.L3700
	cmpq	$6, %rax
	je	.L3701
	movsbl	0(%r13), %edi
	call	toupper@PLT
	leaq	84(%rsp), %rsi
	movb	%al, 0(%r13)
.L3701:
	movsbl	(%rsi), %edi
	movq	%rsi, 16(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %rsi
	movb	%al, (%rsi)
	addq	$1, %rsi
.L3700:
	movsbl	(%rsi), %edi
	movq	%rsi, 16(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %rsi
	movb	%al, (%rsi)
	addq	$1, %rsi
.L3699:
	movsbl	(%rsi), %edi
	movq	%rsi, 16(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %rsi
	movb	%al, (%rsi)
	addq	$1, %rsi
.L3698:
	movsbl	(%rsi), %edi
	movq	%rsi, 16(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %rsi
	movb	%al, (%rsi)
	addq	$1, %rsi
.L3697:
	movsbl	(%rsi), %edi
	movq	%rsi, 16(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %rsi
	movb	%al, (%rsi)
	addq	$1, %rsi
.L3696:
	movsbl	(%rsi), %edi
	movq	%rsi, 16(%rsp)
	call	toupper@PLT
	movq	16(%rsp), %rsi
	movb	%al, (%rsi)
	addq	$1, %rsi
	cmpq	%r15, %rsi
	je	.L3740
.L3746:
	movl	%r12d, %ecx
	movq	%rbx, 16(%rsp)
	movq	%r15, %r12
	movq	%rsi, %rbx
	movl	%ecx, %r15d
.L3580:
	movsbl	(%rbx), %edi
	addq	$8, %rbx
	call	toupper@PLT
	movsbl	-7(%rbx), %edi
	movb	%al, -8(%rbx)
	call	toupper@PLT
	movsbl	-6(%rbx), %edi
	movb	%al, -7(%rbx)
	call	toupper@PLT
	movsbl	-5(%rbx), %edi
	movb	%al, -6(%rbx)
	call	toupper@PLT
	movsbl	-4(%rbx), %edi
	movb	%al, -5(%rbx)
	call	toupper@PLT
	movsbl	-3(%rbx), %edi
	movb	%al, -4(%rbx)
	call	toupper@PLT
	movsbl	-2(%rbx), %edi
	movb	%al, -3(%rbx)
	call	toupper@PLT
	movsbl	-1(%rbx), %edi
	movb	%al, -2(%rbx)
	call	toupper@PLT
	movb	%al, -1(%rbx)
	cmpq	%r12, %rbx
	jne	.L3580
	movl	%r15d, %edi
	movq	16(%rsp), %rbx
	movq	%r12, %r15
	movl	%edi, %r12d
.L3740:
	movl	$1, %r10d
	movl	$2, %eax
	jmp	.L3571
	.p2align 4,,10
	.p2align 3
.L3767:
	bsrq	%r13, %r15
	movl	$2863311531, %edi
	addl	$67, %r15d
	imulq	%rdi, %r15
	shrq	$33, %r15
	leal	-1(%r15), %r9d
	jmp	.L3566
	.p2align 4,,10
	.p2align 3
.L3758:
	bsrq	%r13, %r10
	vmovdqa	.LC25(%rip), %xmm1
	addl	$68, %r10d
	vmovdqa	%xmm1, 224(%rsp)
	shrl	$2, %r10d
	leal	-1(%r10), %edx
	jmp	.L3575
	.p2align 4,,10
	.p2align 3
.L3756:
	bsrq	%r13, %r10
	movl	$128, %eax
	movl	$127, %r11d
	xorq	$63, %r10
	subl	%r10d, %eax
	subl	%r10d, %r11d
	jmp	.L3546
.L3591:
	movl	$1, %eax
	jmp	.L3547
	.p2align 4,,10
	.p2align 3
.L3751:
	vzeroupper
.L3552:
	leal	48(%r12), %r15d
	jmp	.L3563
	.p2align 4,,10
	.p2align 3
.L3757:
	orq	%rsi, %rbx
	jne	.L3598
	movb	$48, 83(%rsp)
	movzbl	(%rdi), %r12d
	leaq	84(%rsp), %r15
	leaq	.LC39(%rip), %rbx
	leaq	83(%rsp), %r13
	jmp	.L3572
	.p2align 4,,10
	.p2align 3
.L3761:
	testl	%r10d, %r10d
	jne	.L3573
	movl	$1, %r10d
	movl	$2, %eax
	movq	%r13, %r15
	jmp	.L3571
	.p2align 4,,10
	.p2align 3
.L3764:
	movq	16(%rsp), %r12
	movq	24(%rsp), %r13
	leal	5(%rdx), %r10d
	movq	32(%rsp), %r14
	jmp	.L3558
	.p2align 4,,10
	.p2align 3
.L3765:
	movq	16(%rsp), %r12
	movq	24(%rsp), %r13
	leal	6(%rdx), %r10d
	movq	32(%rsp), %r14
	jmp	.L3558
	.p2align 4,,10
	.p2align 3
.L3766:
	movq	16(%rsp), %r12
	movq	24(%rsp), %r13
	leal	7(%rdx), %r10d
	movq	32(%rsp), %r14
	jmp	.L3558
	.p2align 4,,10
	.p2align 3
.L3599:
	leaq	.LC38(%rip), %rbx
	jmp	.L3536
	.p2align 4,,10
	.p2align 3
.L3759:
	leaq	224(%rsp), %r15
	jmp	.L3576
.L3596:
	leaq	211(%rsp), %r15
	leaq	83(%rsp), %r13
	jmp	.L3561
.L3598:
	leaq	.LC39(%rip), %rbx
	jmp	.L3536
.L3593:
	movl	$1, %r8d
	jmp	.L3552
.L3763:
	vmovdqa	.LC26(%rip), %ymm9
	vmovdqa	.LC27(%rip), %ymm10
	movl	$2, %r8d
	leaq	224(%rsp), %r10
	vmovdqa	.LC28(%rip), %ymm11
	vmovdqa	.LC29(%rip), %ymm12
	vmovdqa	.LC30(%rip), %ymm13
	vmovdqa	.LC31(%rip), %ymm14
	vmovdqu	%ymm9, 224(%rsp)
	vmovdqa	.LC32(%rip), %xmm15
	vmovdqu	%ymm10, 256(%rsp)
	vmovdqu	%ymm14, 384(%rsp)
	vmovdqu	%ymm11, 288(%rsp)
	vmovdqu	%ymm12, 320(%rsp)
	vmovdqu	%ymm13, 352(%rsp)
	vmovdqu	%xmm15, 409(%rsp)
	jmp	.L3554
.L3595:
	movl	$4, %edx
	movl	$3, %r8d
	jmp	.L3555
.L3594:
	movl	$3, %edx
	movl	$2, %r8d
	jmp	.L3555
.L3528:
	movq	440(%rsp), %rax
	subq	%fs:40, %rax
	je	.L3530
.L3753:
	call	__stack_chk_fail@PLT
.L3530:
	leaq	.LC40(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE13688:
	.size	_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.text._ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13690:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %r11
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movq	%rdx, %r13
	pushq	%r12
	.cfi_offset 12, -48
	movq	%rsi, %r12
	movq	%rdx, %rsi
	pushq	%rbx
	subq	$424, %rsp
	.cfi_offset 3, -56
	movzbl	1(%rdi), %edx
	movq	%rcx, 24(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 408(%rsp)
	xorl	%eax, %eax
	movl	%edx, %eax
	andl	$120, %eax
	cmpb	$56, %al
	je	.L3988
	shrb	$3, %dl
	andl	$15, %edx
	cmpb	$4, %dl
	je	.L3773
	ja	.L3774
	cmpb	$1, %dl
	ja	.L3989
	movq	%r12, %r14
	orq	%rsi, %r14
	jne	.L3787
	movb	$48, 51(%rsp)
	leaq	52(%rsp), %r14
	leaq	51(%rsp), %r13
.L3788:
	movzbl	(%r11), %r12d
	movq	%r13, %rdx
.L3786:
	shrb	$2, %r12b
	movl	$43, %r8d
	andl	$3, %r12d
	cmpl	$1, %r12d
	je	.L3823
	cmpl	$3, %r12d
	je	.L3839
.L3822:
	movq	%r14, %rsi
	movq	%r13, %rcx
	movq	24(%rsp), %r8
	movq	%r11, %rdi
	subq	%rdx, %rsi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L3771
	.p2align 4,,10
	.p2align 3
.L3988:
	movl	$127, %ebx
	movl	$0, %eax
	cmpq	%r12, %rbx
	sbbq	%rsi, %rax
	jc	.L3770
	movq	%rcx, %rdx
	leaq	47(%rsp), %rsi
	movq	%rdi, %rcx
	movl	$1, %edi
	movb	%r12b, 47(%rsp)
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
.L3771:
	movq	408(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L3987
	addq	$424, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L3989:
	.cfi_restore_state
	cmpb	$16, %al
	leaq	.LC35(%rip), %rbx
	leaq	.LC36(%rip), %rdi
	movq	%r12, %r9
	cmovne	%rdi, %rbx
	orq	%rsi, %r9
	je	.L3825
	testq	%rsi, %rsi
	jne	.L3990
	bsrq	%r12, %rsi
	movl	$128, %eax
	movl	$127, %r8d
	xorq	$63, %rsi
	addl	$64, %esi
	subl	%esi, %eax
	subl	%esi, %r8d
	je	.L3826
.L3782:
	movl	%r8d, %r10d
	subl	$1, %r8d
	leaq	48(%rsp,%r10), %r15
	leaq	47(%rsp,%r10), %r14
	subq	%r8, %r14
	movq	%r15, %rdi
	subq	%r14, %rdi
	andl	$7, %edi
	je	.L3979
	cmpq	$1, %rdi
	je	.L3922
	cmpq	$2, %rdi
	je	.L3923
	cmpq	$3, %rdi
	je	.L3924
	cmpq	$4, %rdi
	je	.L3925
	cmpq	$5, %rdi
	je	.L3926
	cmpq	$6, %rdi
	je	.L3927
	movl	%r12d, %r9d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r9d
	shrq	%r13
	addl	$48, %r9d
	movb	%r9b, 4(%r15)
.L3927:
	movl	%r12d, %edx
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %edx
	shrq	%r13
	addl	$48, %edx
	movb	%dl, 4(%r15)
.L3926:
	movl	%r12d, %esi
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %esi
	shrq	%r13
	addl	$48, %esi
	movb	%sil, 4(%r15)
.L3925:
	movl	%r12d, %r8d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r8d
	shrq	%r13
	addl	$48, %r8d
	movb	%r8b, 4(%r15)
.L3924:
	movl	%r12d, %ecx
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %ecx
	shrq	%r13
	addl	$48, %ecx
	movb	%cl, 4(%r15)
.L3923:
	movl	%r12d, %r10d
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %r10d
	shrq	%r13
	addl	$48, %r10d
	movb	%r10b, 4(%r15)
.L3922:
	movl	%r12d, %edi
	subq	$1, %r15
	shrdq	$1, %r13, %r12
	andl	$1, %edi
	shrq	%r13
	addl	$48, %edi
	movb	%dil, 4(%r15)
	cmpq	%r14, %r15
	je	.L3783
.L3979:
	movq	%rbx, %r9
.L3784:
	movq	%r12, %rdx
	movq	%r12, %r8
	movq	%r12, %r10
	movq	%r12, %rsi
	shrdq	$1, %r13, %rdx
	movq	%r12, %rcx
	movq	%r12, %rdi
	movl	%r12d, %ebx
	andl	$1, %edx
	shrdq	$2, %r13, %r8
	andl	$1, %ebx
	subq	$8, %r15
	addl	$48, %edx
	shrdq	$3, %r13, %r10
	andl	$1, %r8d
	addl	$48, %ebx
	movb	%dl, 10(%r15)
	movq	%r12, %rdx
	andl	$1, %r10d
	addl	$48, %r8d
	shrdq	$4, %r13, %rdx
	shrdq	$5, %r13, %rsi
	addl	$48, %r10d
	movb	%bl, 11(%r15)
	shrdq	$6, %r13, %rcx
	shrdq	$7, %r13, %rdi
	andl	$1, %edx
	andl	$1, %esi
	andl	$1, %ecx
	andl	$1, %edi
	addl	$48, %edx
	addl	$48, %esi
	addl	$48, %ecx
	addl	$48, %edi
	shrdq	$8, %r13, %r12
	movb	%r8b, 9(%r15)
	movb	%r10b, 8(%r15)
	shrq	$8, %r13
	movb	%dl, 7(%r15)
	movb	%sil, 6(%r15)
	movb	%cl, 5(%r15)
	movb	%dil, 4(%r15)
	cmpq	%r14, %r15
	jne	.L3784
	movq	%r9, %rbx
.L3783:
	leaq	51(%rsp), %r13
	cltq
	movl	$49, %r12d
	leaq	0(%r13,%rax), %r14
	jmp	.L3780
	.p2align 4,,10
	.p2align 3
.L3773:
	movq	%r12, %rcx
	orq	%rsi, %rcx
	je	.L3832
	testq	%rsi, %rsi
	jne	.L3991
	bsrq	%r12, %r9
	movl	$2863311531, %r8d
	movl	$63, %r14d
	addl	$3, %r9d
	imulq	%r8, %r9
	shrq	$33, %r9
	leal	-1(%r9), %r10d
	cmpq	%r12, %r14
	jnb	.L3804
.L3803:
	movl	$63, %edx
	xorl	%eax, %eax
.L3805:
	movq	%r12, %rdi
	movq	%r12, %rcx
	shrdq	$6, %r13, %r12
	movq	%rax, %rsi
	shrdq	$3, %r13, %rdi
	andl	$7, %ecx
	shrq	$6, %r13
	movl	%r10d, %ebx
	andl	$7, %edi
	leal	-1(%r10), %r14d
	leal	-2(%r10), %r15d
	addl	$48, %ecx
	addl	$48, %edi
	cmpq	%r12, %rdx
	movb	%cl, 51(%rsp,%rbx)
	sbbq	%r13, %rsi
	movb	%dil, 51(%rsp,%r14)
	jnc	.L3804
	movq	%r12, %rbx
	leal	-3(%r10), %r8d
	leal	-4(%r10), %r14d
	movq	%r12, %rcx
	shrdq	$3, %r13, %rcx
	andl	$7, %ebx
	shrdq	$6, %r13, %r12
	addl	$48, %ebx
	andl	$7, %ecx
	shrq	$6, %r13
	movb	%bl, 51(%rsp,%r15)
	addl	$48, %ecx
	movq	%rax, %r15
	cmpq	%r12, %rdx
	sbbq	%r13, %r15
	movb	%cl, 51(%rsp,%r8)
	jnc	.L3804
	movq	%r12, %rcx
	movq	%r12, %rsi
	shrdq	$6, %r13, %r12
	movq	%rax, %r8
	shrdq	$3, %r13, %rcx
	andl	$7, %esi
	shrq	$6, %r13
	leal	-5(%r10), %edi
	andl	$7, %ecx
	addl	$48, %esi
	addl	$48, %ecx
	cmpq	%r12, %rdx
	movb	%sil, 51(%rsp,%r14)
	leal	-6(%r10), %r14d
	sbbq	%r13, %r8
	movb	%cl, 51(%rsp,%rdi)
	jnc	.L3804
	movq	%r12, %rsi
	movq	%r12, %r15
	shrdq	$6, %r13, %r12
	leal	-7(%r10), %ebx
	shrdq	$3, %r13, %rsi
	andl	$7, %r15d
	shrq	$6, %r13
	movq	%rax, %rdi
	andl	$7, %esi
	addl	$48, %r15d
	subl	$8, %r10d
	addl	$48, %esi
	cmpq	%r12, %rdx
	movb	%r15b, 51(%rsp,%r14)
	sbbq	%r13, %rdi
	movb	%sil, 51(%rsp,%rbx)
	jc	.L3805
	.p2align 4,,10
	.p2align 3
.L3804:
	movl	$7, %r10d
	movl	$0, %edx
	leal	48(%r12), %edi
	cmpq	%r12, %r10
	sbbq	%r13, %rdx
	jc	.L3992
.L3807:
	leaq	51(%rsp), %r13
	movl	%r9d, %r12d
	movl	$1, %r10d
	movl	$1, %eax
	leaq	0(%r13,%r12), %r14
	leaq	.LC37(%rip), %rbx
.L3801:
	movb	%dil, 51(%rsp)
	movzbl	(%r11), %r12d
.L3808:
	testb	$16, %r12b
	je	.L3838
	testb	%r10b, %r10b
	jne	.L3993
.L3838:
	movq	%r13, %rdx
	jmp	.L3786
	.p2align 4,,10
	.p2align 3
.L3839:
	movl	$32, %r8d
.L3823:
	movb	%r8b, -1(%rdx)
	subq	$1, %rdx
	jmp	.L3822
	.p2align 4,,10
	.p2align 3
.L3774:
	cmpb	$40, %al
	je	.L3994
	movq	%r12, %r9
	orq	%rsi, %r9
	jne	.L3834
	movb	$48, 51(%rsp)
	movzbl	(%rdi), %r12d
	cmpb	$48, %al
	je	.L3835
	leaq	52(%rsp), %r14
	leaq	.LC38(%rip), %rbx
	leaq	51(%rsp), %r13
	jmp	.L3810
	.p2align 4,,10
	.p2align 3
.L3825:
	movl	$48, %r12d
	leaq	52(%rsp), %r14
	leaq	51(%rsp), %r13
.L3780:
	movb	%r12b, 51(%rsp)
	movzbl	(%r11), %r12d
	testb	$16, %r12b
	je	.L3838
.L3837:
	movq	$-2, %rdx
	movl	$2, %eax
.L3785:
	addq	%r13, %rdx
	movl	%eax, %r15d
	testl	%eax, %eax
	je	.L3786
	xorl	%esi, %esi
	leal	-1(%rax), %r9d
	movl	$1, %eax
	movzbl	(%rbx,%rsi), %r8d
	andl	$7, %r9d
	movb	%r8b, (%rdx,%rsi)
	cmpl	%r15d, %eax
	jnb	.L3786
	testl	%r9d, %r9d
	je	.L3819
	cmpl	$1, %r9d
	je	.L3937
	cmpl	$2, %r9d
	je	.L3938
	cmpl	$3, %r9d
	je	.L3939
	cmpl	$4, %r9d
	je	.L3940
	cmpl	$5, %r9d
	je	.L3941
	cmpl	$6, %r9d
	je	.L3942
	movl	$1, %r10d
	movl	$2, %eax
	movzbl	(%rbx,%r10), %ecx
	movb	%cl, (%rdx,%r10)
.L3942:
	movl	%eax, %r9d
	addl	$1, %eax
	movzbl	(%rbx,%r9), %edi
	movb	%dil, (%rdx,%r9)
.L3941:
	movl	%eax, %esi
	addl	$1, %eax
	movzbl	(%rbx,%rsi), %r8d
	movb	%r8b, (%rdx,%rsi)
.L3940:
	movl	%eax, %r10d
	addl	$1, %eax
	movzbl	(%rbx,%r10), %ecx
	movb	%cl, (%rdx,%r10)
.L3939:
	movl	%eax, %r9d
	addl	$1, %eax
	movzbl	(%rbx,%r9), %edi
	movb	%dil, (%rdx,%r9)
.L3938:
	movl	%eax, %esi
	addl	$1, %eax
	movzbl	(%rbx,%rsi), %r8d
	movb	%r8b, (%rdx,%rsi)
.L3937:
	movl	%eax, %r10d
	addl	$1, %eax
	movzbl	(%rbx,%r10), %ecx
	movb	%cl, (%rdx,%r10)
	cmpl	%r15d, %eax
	jnb	.L3786
.L3819:
	movl	%eax, %r9d
	leal	1(%rax), %esi
	leal	2(%rax), %r10d
	movzbl	(%rbx,%r9), %edi
	movzbl	(%rbx,%rsi), %r8d
	movzbl	(%rbx,%r10), %ecx
	movb	%dil, (%rdx,%r9)
	leal	3(%rax), %r9d
	movb	%r8b, (%rdx,%rsi)
	leal	4(%rax), %esi
	movzbl	(%rbx,%r9), %edi
	movzbl	(%rbx,%rsi), %r8d
	movb	%cl, (%rdx,%r10)
	leal	5(%rax), %r10d
	movb	%dil, (%rdx,%r9)
	leal	6(%rax), %r9d
	movzbl	(%rbx,%r10), %ecx
	movb	%r8b, (%rdx,%rsi)
	leal	7(%rax), %esi
	movzbl	(%rbx,%r9), %edi
	addl	$8, %eax
	movzbl	(%rbx,%rsi), %r8d
	movb	%cl, (%rdx,%r10)
	movb	%dil, (%rdx,%r9)
	movb	%r8b, (%rdx,%rsi)
	cmpl	%r15d, %eax
	jb	.L3819
	jmp	.L3786
	.p2align 4,,10
	.p2align 3
.L3832:
	movl	$48, %edi
	xorl	%r10d, %r10d
	leaq	52(%rsp), %r14
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	leaq	51(%rsp), %r13
	jmp	.L3801
	.p2align 4,,10
	.p2align 3
.L3787:
	xorl	%r15d, %r15d
	movl	$9, %edx
	cmpq	%r12, %rdx
	movq	%r15, %rax
	sbbq	%rsi, %rax
	jnc	.L3828
	movl	$99, %r10d
	movq	%r15, %rcx
	cmpq	%r12, %r10
	sbbq	%rsi, %rcx
	jnc	.L3995
	movl	$999, %r9d
	movq	%r15, %rdi
	cmpq	%r12, %r9
	sbbq	%rsi, %rdi
	jnc	.L3829
	movl	$9999, %esi
	cmpq	%r12, %rsi
	sbbq	%r13, %r15
	jnc	.L3830
	movl	$1, %r8d
	movq	%r13, %rax
	movq	%r12, (%rsp)
	movq	%r12, %r15
	movq	%r13, 8(%rsp)
	xorl	%r14d, %r14d
	movl	%r8d, %r13d
	movq	%r11, 16(%rsp)
	jmp	.L3793
	.p2align 4,,10
	.p2align 3
.L3797:
	movl	$999999, %ecx
	movq	%r14, %r9
	cmpq	%r12, %rcx
	sbbq	%rbx, %r9
	jnc	.L3996
	movl	$9999999, %edi
	movq	%r14, %rsi
	cmpq	%r12, %rdi
	sbbq	%rbx, %rsi
	jnc	.L3997
	movl	$99999999, %r8d
	cmpq	%r12, %r8
	movq	%r14, %r12
	sbbq	%rbx, %r12
	jnc	.L3998
.L3793:
	xorl	%ecx, %ecx
	movq	%r15, %rdi
	movl	$10000, %edx
	movq	%rax, %rsi
	movq	%r15, %r12
	movq	%rax, %rbx
	call	__udivti3@PLT
	movl	$99999, %r11d
	movq	%r14, %r10
	movq	%rax, %r15
	movq	%rdx, %rax
	movl	%r13d, %edx
	addl	$4, %r13d
	cmpq	%r12, %r11
	sbbq	%rbx, %r10
	jc	.L3797
	movl	%r13d, %ebx
	movq	16(%rsp), %r11
	movq	(%rsp), %r12
	movq	8(%rsp), %r13
.L3795:
	cmpl	$128, %ebx
	ja	.L3831
	leal	-1(%rbx), %edx
	movl	%ebx, %eax
.L3792:
	vmovdqa	.LC26(%rip), %ymm2
	vmovdqa	.LC27(%rip), %ymm3
	movq	%rax, 16(%rsp)
	leaq	192(%rsp), %r10
	vmovdqa	.LC28(%rip), %ymm4
	vmovdqa	.LC29(%rip), %ymm5
	movq	%r11, (%rsp)
	movabsq	$1152921504606846975, %r14
	vmovdqa	.LC30(%rip), %ymm6
	vmovdqa	.LC31(%rip), %ymm7
	movq	%r10, %r11
	movl	%edx, %r10d
	vmovdqa	.LC32(%rip), %xmm8
	vmovdqu	%ymm2, 192(%rsp)
	movabsq	$-8116567392432202711, %r15
	vmovdqu	%ymm7, 352(%rsp)
	vmovdqu	%ymm3, 224(%rsp)
	vmovdqu	%ymm4, 256(%rsp)
	vmovdqu	%ymm5, 288(%rsp)
	vmovdqu	%ymm6, 320(%rsp)
	vmovdqu	%xmm8, 377(%rsp)
	.p2align 4,,10
	.p2align 3
.L3799:
	movq	%r12, %rax
	movq	%r12, %rcx
	xorl	%ebx, %ebx
	movl	$25, %r9d
	shrdq	$60, %r13, %rax
	andq	%r14, %rcx
	andq	%r14, %rax
	addq	%rax, %rcx
	movq	%r13, %rax
	shrq	$56, %rax
	addq	%rax, %rcx
	movabsq	$5165088340638674453, %rax
	mulq	%rcx
	movq	%rcx, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$4, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r13, %rdx
	leaq	(%rax,%rax,4), %rax
	subq	%rax, %rcx
	movq	%r12, %rax
	subq	%rcx, %rax
	sbbq	%rbx, %rdx
	movq	%rdx, %r8
	movabsq	$2951479051793528258, %rdx
	imulq	%rax, %rdx
	imulq	%r15, %r8
	addq	%rdx, %r8
	mulq	%r15
	movq	%rax, %rsi
	andl	$3, %eax
	movq	%rdx, %rdi
	mulq	%r9
	addq	%r8, %rdi
	movq	%r13, %rdx
	addq	%rcx, %rax
	shrdq	$2, %rdi, %rsi
	movq	%r12, %rcx
	xorl	%r9d, %r9d
	addq	%rax, %rax
	shrq	$2, %rdi
	movq	%rsi, %r12
	movl	%r10d, %esi
	movq	%rdi, %r13
	leaq	(%r11,%rax), %rdi
	leaq	(%rax,%r11), %rax
	movzbl	1(%rdi), %edi
	movzbl	(%rax), %eax
	movb	%dil, 51(%rsp,%rsi)
	leal	-1(%r10), %esi
	subl	$2, %r10d
	movb	%al, 51(%rsp,%rsi)
	movl	$9999, %eax
	cmpq	%rcx, %rax
	sbbq	%rdx, %r9
	jc	.L3799
	movl	$999, %r14d
	xorl	%r9d, %r9d
	movq	%r11, %r13
	movq	16(%rsp), %r15
	cmpq	%rcx, %r14
	movq	(%rsp), %r11
	sbbq	%rdx, %r9
	jnc	.L3985
.L3791:
	addq	%r12, %r12
	leaq	0(%r13,%r12), %rcx
	addq	%r12, %r13
	movzbl	1(%rcx), %edi
	movzbl	0(%r13), %esi
	movb	%dil, 52(%rsp)
	vzeroupper
.L3800:
	leaq	51(%rsp), %r13
	movb	%sil, 51(%rsp)
	leaq	0(%r13,%r15), %r14
	jmp	.L3788
	.p2align 4,,10
	.p2align 3
.L3994:
	movq	%r12, %r8
	orq	%rsi, %r8
	jne	.L3833
	movb	$48, 51(%rsp)
	movzbl	(%rdi), %r12d
	leaq	52(%rsp), %r14
	leaq	.LC39(%rip), %rbx
	leaq	51(%rsp), %r13
	jmp	.L3810
	.p2align 4,,10
	.p2align 3
.L3834:
	leaq	.LC38(%rip), %rbx
.L3809:
	testq	%r13, %r13
	jne	.L3999
	bsrq	%r12, %r10
	vmovdqa	.LC25(%rip), %xmm0
	movl	$255, %ecx
	addl	$4, %r10d
	vmovdqa	%xmm0, 192(%rsp)
	shrl	$2, %r10d
	leal	-1(%r10), %edx
	cmpq	%r12, %rcx
	jnb	.L4000
.L3813:
	leaq	192(%rsp), %r15
	movl	$255, %r9d
	xorl	%r8d, %r8d
.L3815:
	movq	%r12, %r14
	movl	%edx, %edi
	andl	$15, %r14d
	addq	%r15, %r14
	movzbl	(%r14), %esi
	movq	%r12, %r14
	shrdq	$8, %r13, %r12
	shrdq	$4, %r13, %r14
	shrq	$8, %r13
	andl	$15, %r14d
	movb	%sil, 51(%rsp,%rdi)
	leal	-1(%rdx), %edi
	cmpq	%r12, %r9
	leaq	(%r14,%r15), %rsi
	leal	-2(%rdx), %r14d
	movzbl	(%rsi), %ecx
	movb	%cl, 51(%rsp,%rdi)
	movq	%r8, %rdi
	sbbq	%r13, %rdi
	jnc	.L3814
	movq	%r12, %rsi
	movl	%r14d, %ecx
	andl	$15, %esi
	addq	%r15, %rsi
	movzbl	(%rsi), %r14d
	movb	%r14b, 51(%rsp,%rcx)
	movq	%r12, %r14
	leal	-3(%rdx), %ecx
	shrdq	$8, %r13, %r12
	shrdq	$4, %r13, %r14
	shrq	$8, %r13
	andl	$15, %r14d
	cmpq	%r12, %r9
	leaq	(%r14,%r15), %rsi
	leal	-4(%rdx), %r14d
	movzbl	(%rsi), %edi
	movb	%dil, 51(%rsp,%rcx)
	movq	%r8, %rcx
	sbbq	%r13, %rcx
	jnc	.L3814
	movq	%r12, %rsi
	movl	%r14d, %edi
	andl	$15, %esi
	addq	%r15, %rsi
	movzbl	(%rsi), %r14d
	movb	%r14b, 51(%rsp,%rdi)
	movq	%r12, %r14
	leal	-5(%rdx), %edi
	shrdq	$8, %r13, %r12
	shrdq	$4, %r13, %r14
	shrq	$8, %r13
	andl	$15, %r14d
	cmpq	%r12, %r9
	leaq	(%r14,%r15), %rsi
	leal	-6(%rdx), %r14d
	movzbl	(%rsi), %ecx
	movb	%cl, 51(%rsp,%rdi)
	movq	%r8, %rdi
	sbbq	%r13, %rdi
	jnc	.L3814
	movq	%r12, %rsi
	movl	%r14d, %ecx
	andl	$15, %esi
	addq	%r15, %rsi
	movzbl	(%rsi), %r14d
	movb	%r14b, 51(%rsp,%rcx)
	movq	%r12, %r14
	shrdq	$8, %r13, %r12
	leal	-7(%rdx), %ecx
	shrdq	$4, %r13, %r14
	subl	$8, %edx
	shrq	$8, %r13
	andl	$15, %r14d
	cmpq	%r12, %r9
	leaq	(%r14,%r15), %rsi
	movq	%r8, %r14
	movzbl	(%rsi), %edi
	sbbq	%r13, %r14
	movb	%dil, 51(%rsp,%rcx)
	jc	.L3815
	.p2align 4,,10
	.p2align 3
.L3814:
	movl	$15, %edx
	movl	$0, %r9d
	cmpq	%r12, %rdx
	sbbq	%r13, %r9
	jc	.L4001
	addq	%r12, %r15
	movzbl	(%r15), %r13d
.L3817:
	movb	%r13b, 51(%rsp)
	movl	%r10d, %r14d
	leaq	51(%rsp), %r13
	movzbl	(%r11), %r12d
	addq	%r13, %r14
	cmpb	$48, %al
	je	.L4002
.L3810:
	testb	$16, %r12b
	jne	.L3837
	jmp	.L3838
	.p2align 4,,10
	.p2align 3
.L3992:
	movq	%r12, %rax
	shrdq	$3, %r13, %r12
	andl	$7, %eax
	leal	48(%r12), %edi
	addl	$48, %eax
	movb	%al, 52(%rsp)
	jmp	.L3807
	.p2align 4,,10
	.p2align 3
.L4001:
	movq	%r12, %r8
	shrdq	$4, %r13, %r12
	andl	$15, %r8d
	leaq	(%r15,%r12), %r12
	addq	%r15, %r8
	movzbl	(%r12), %r13d
	movzbl	(%r8), %ecx
	movb	%cl, 52(%rsp)
	jmp	.L3817
	.p2align 4,,10
	.p2align 3
.L3833:
	leaq	.LC39(%rip), %rbx
	jmp	.L3809
	.p2align 4,,10
	.p2align 3
.L3835:
	leaq	.LC38(%rip), %rbx
	leaq	52(%rsp), %r14
	leaq	51(%rsp), %r13
.L3811:
	movq	%r14, %rax
	movq	%r13, %r15
	subq	%r13, %rax
	andl	$7, %eax
	je	.L3980
	cmpq	$1, %rax
	je	.L3930
	cmpq	$2, %rax
	je	.L3931
	cmpq	$3, %rax
	je	.L3932
	cmpq	$4, %rax
	je	.L3933
	cmpq	$5, %rax
	je	.L3934
	cmpq	$6, %rax
	je	.L3935
	movsbl	0(%r13), %edi
	movq	%r11, 16(%rsp)
	leaq	52(%rsp), %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, 0(%r13)
.L3935:
	movsbl	(%r15), %edi
	movq	%r11, 16(%rsp)
	addq	$1, %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, -1(%r15)
.L3934:
	movsbl	(%r15), %edi
	movq	%r11, 16(%rsp)
	addq	$1, %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, -1(%r15)
.L3933:
	movsbl	(%r15), %edi
	movq	%r11, 16(%rsp)
	addq	$1, %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, -1(%r15)
.L3932:
	movsbl	(%r15), %edi
	movq	%r11, 16(%rsp)
	addq	$1, %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, -1(%r15)
.L3931:
	movsbl	(%r15), %edi
	movq	%r11, 16(%rsp)
	addq	$1, %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, -1(%r15)
.L3930:
	movsbl	(%r15), %edi
	movq	%r11, 16(%rsp)
	addq	$1, %r15
	call	toupper@PLT
	movq	16(%rsp), %r11
	movb	%al, -1(%r15)
	cmpq	%r14, %r15
	je	.L3974
.L3980:
	movq	%rbx, 16(%rsp)
	movq	%r14, %rbx
	movq	%r13, %r14
	movl	%r12d, %r13d
	movq	%r11, %r12
.L3818:
	movsbl	(%r15), %edi
	addq	$8, %r15
	call	toupper@PLT
	movsbl	-7(%r15), %edi
	movb	%al, -8(%r15)
	call	toupper@PLT
	movsbl	-6(%r15), %edi
	movb	%al, -7(%r15)
	call	toupper@PLT
	movsbl	-5(%r15), %edi
	movb	%al, -6(%r15)
	call	toupper@PLT
	movsbl	-4(%r15), %edi
	movb	%al, -5(%r15)
	call	toupper@PLT
	movsbl	-3(%r15), %edi
	movb	%al, -4(%r15)
	call	toupper@PLT
	movsbl	-2(%r15), %edi
	movb	%al, -3(%r15)
	call	toupper@PLT
	movsbl	-1(%r15), %edi
	movb	%al, -2(%r15)
	call	toupper@PLT
	movb	%al, -1(%r15)
	cmpq	%rbx, %r15
	jne	.L3818
	movq	%r12, %r11
	movl	%r13d, %r12d
	movq	%r14, %r13
	movq	%rbx, %r14
	movq	16(%rsp), %rbx
.L3974:
	movl	$1, %r10d
	movl	$2, %eax
	jmp	.L3808
	.p2align 4,,10
	.p2align 3
.L3999:
	bsrq	%r13, %r10
	vmovdqa	.LC25(%rip), %xmm1
	addl	$68, %r10d
	vmovdqa	%xmm1, 192(%rsp)
	shrl	$2, %r10d
	leal	-1(%r10), %edx
	jmp	.L3813
	.p2align 4,,10
	.p2align 3
.L3990:
	bsrq	%rsi, %rcx
	movl	$128, %eax
	movl	$127, %r8d
	xorq	$63, %rcx
	subl	%ecx, %eax
	subl	%ecx, %r8d
	jmp	.L3782
.L3826:
	movl	$1, %eax
	jmp	.L3783
	.p2align 4,,10
	.p2align 3
.L3985:
	vzeroupper
.L3789:
	leal	48(%r12), %esi
	jmp	.L3800
	.p2align 4,,10
	.p2align 3
.L4002:
	testl	%r10d, %r10d
	jne	.L3811
	movl	$1, %r10d
	movl	$2, %eax
	movq	%r13, %r14
	jmp	.L3808
	.p2align 4,,10
	.p2align 3
.L3991:
	bsrq	%rsi, %r15
	movl	$2863311531, %esi
	leal	67(%r15), %r9d
	imulq	%rsi, %r9
	shrq	$33, %r9
	leal	-1(%r9), %r10d
	jmp	.L3803
	.p2align 4,,10
	.p2align 3
.L3996:
	movq	16(%rsp), %r11
	movq	(%rsp), %r12
	leal	5(%rdx), %ebx
	movq	8(%rsp), %r13
	jmp	.L3795
	.p2align 4,,10
	.p2align 3
.L3997:
	movq	16(%rsp), %r11
	movq	(%rsp), %r12
	leal	6(%rdx), %ebx
	movq	8(%rsp), %r13
	jmp	.L3795
	.p2align 4,,10
	.p2align 3
.L3998:
	movq	16(%rsp), %r11
	movq	(%rsp), %r12
	leal	7(%rdx), %ebx
	movq	8(%rsp), %r13
	jmp	.L3795
	.p2align 4,,10
	.p2align 3
.L4000:
	leaq	192(%rsp), %r15
	jmp	.L3814
	.p2align 4,,10
	.p2align 3
.L3993:
	movq	%rax, %rdx
	negq	%rdx
	jmp	.L3785
.L3831:
	leaq	179(%rsp), %r14
	leaq	51(%rsp), %r13
	jmp	.L3788
.L3828:
	movl	$1, %r15d
	jmp	.L3789
.L3995:
	vmovdqa	.LC26(%rip), %ymm9
	vmovdqa	.LC27(%rip), %ymm10
	movl	$2, %r15d
	leaq	192(%rsp), %r13
	vmovdqa	.LC28(%rip), %ymm11
	vmovdqa	.LC29(%rip), %ymm12
	vmovdqa	.LC30(%rip), %ymm13
	vmovdqa	.LC31(%rip), %ymm14
	vmovdqu	%ymm9, 192(%rsp)
	vmovdqa	.LC32(%rip), %xmm15
	vmovdqu	%ymm10, 224(%rsp)
	vmovdqu	%ymm14, 352(%rsp)
	vmovdqu	%ymm11, 256(%rsp)
	vmovdqu	%ymm12, 288(%rsp)
	vmovdqu	%ymm13, 320(%rsp)
	vmovdqu	%xmm15, 377(%rsp)
	jmp	.L3791
.L3830:
	movl	$4, %eax
	movl	$3, %edx
	jmp	.L3792
.L3829:
	movl	$3, %eax
	movl	$2, %edx
	jmp	.L3792
.L3770:
	movq	408(%rsp), %rax
	subq	%fs:40, %rax
	je	.L3772
.L3987:
	call	__stack_chk_fail@PLT
.L3772:
	leaq	.LC40(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
	.cfi_endproc
.LFE13690:
	.size	_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.rodata.str1.1
.LC41:
	.string	"basic_string_view::copy"
	.section	.text.unlikely
	.align 2
.LCOLDB42:
	.text
.LHOTB42:
	.align 2
	.p2align 4
	.type	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0, @function
_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0:
.LFB14067:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA14067
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movl	%ecx, %r14d
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movq	%r8, %r13
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$104, %rsp
	.cfi_def_cfa_offset 160
	movq	%fs:40, %rax
	movq	%rax, 88(%rsp)
	xorl	%eax, %eax
	leaq	16(%rdi), %rax
	movq	$0, 8(%rdi)
	movq	%rax, (%rdi)
	movb	$0, 16(%rdi)
.LEHB8:
	call	_ZNSt6locale7classicEv@PLT
	movq	%rax, %rsi
	movq	%r13, %rdi
	call	_ZNKSt6localeeqERKS_@PLT
	testb	%al, %al
	je	.L4062
.L4003:
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4061
	addq	$104, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L4062:
	.cfi_restore_state
	leaq	_ZNSt7__cxx118numpunctIcE2idE(%rip), %rdi
	call	_ZNKSt6locale2id5_M_idEv@PLT
	movq	0(%r13), %rcx
	movq	8(%rcx), %rsi
	movq	(%rsi,%rax,8), %r13
	testq	%r13, %r13
	je	.L4005
	movq	0(%r13), %r15
	movq	%r13, %rdi
	call	*16(%r15)
	movq	0(%r13), %r8
	leaq	48(%rsp), %rdi
	movb	%al, 23(%rsp)
	movq	%r13, %rsi
	movq	%rdi, 40(%rsp)
	call	*32(%r8)
.LEHE8:
	cmpb	$46, 23(%rsp)
	movq	56(%rsp), %r15
	jne	.L4008
	testq	%r15, %r15
	jne	.L4008
	.p2align 4,,10
	.p2align 3
.L4009:
	movq	48(%rsp), %rdi
	leaq	64(%rsp), %r9
	cmpq	%r9, %rdi
	je	.L4003
	movq	64(%rsp), %rbp
	leaq	1(%rbp), %rsi
	call	_ZdlPvm@PLT
	jmp	.L4003
	.p2align 4,,10
	.p2align 3
.L4008:
	testq	%r12, %r12
	je	.L4063
	movq	%r12, %rdx
	movl	$46, %esi
	movq	%rbp, %rdi
	call	memchr@PLT
	movsbl	%r14b, %esi
	movq	%r12, %rdx
	movq	%rbp, %rdi
	movq	%rax, (%rsp)
	call	memchr@PLT
	movq	(%rsp), %r9
	movq	%rax, %r14
	testq	%r9, %r9
	je	.L4012
	movq	%r9, %rcx
	subq	%rbp, %rcx
	movq	%rcx, 32(%rsp)
	testq	%rax, %rax
	je	.L4013
	subq	%rbp, %r14
	cmpq	%rcx, %r14
	jnb	.L4013
.L4058:
	movq	%r12, %r10
	subq	%r14, %r10
	movq	%r10, (%rsp)
.L4014:
	movq	8(%rbx), %rsi
	leaq	(%r10,%r14,2), %rcx
	cmpq	%rcx, %rsi
	jnb	.L4011
	subq	%rsi, %rcx
	xorl	%r8d, %r8d
	xorl	%edx, %edx
	movq	%rbx, %rdi
.LEHB9:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
.L4059:
	movq	56(%rsp), %r15
.L4015:
	movq	(%rbx), %rax
	movq	48(%rsp), %rdx
	movq	%r13, %rdi
	movq	%rax, 8(%rsp)
	movq	0(%r13), %rax
	movq	%rdx, 24(%rsp)
	call	*24(%rax)
	movq	24(%rsp), %rdx
	movq	8(%rsp), %rdi
	movq	%r15, %rcx
	movsbl	%al, %esi
	leaq	0(%rbp,%r14), %r9
	movq	%rbp, %r8
	call	_ZSt14__add_groupingIcEPT_S1_S0_PKcmPKS0_S5_
	cmpq	$0, (%rsp)
	movq	%rax, %rcx
	je	.L4016
	cmpq	$-1, 32(%rsp)
	je	.L4017
	movzbl	23(%rsp), %esi
	addq	$1, %rcx
	addq	$1, %r14
	movb	%sil, (%rax)
.L4017:
	cmpq	$1, (%rsp)
	jne	.L4064
.L4016:
	movq	8(%rsp), %r8
	movq	8(%rbx), %rsi
	subq	%r8, %rcx
	cmpq	%rcx, %rsi
	jb	.L4065
	cmpq	%rsi, %rcx
	jnb	.L4009
	movq	(%rbx), %rdi
	movq	%rcx, 8(%rbx)
	movb	$0, (%rdi,%rcx)
	jmp	.L4009
	.p2align 4,,10
	.p2align 3
.L4063:
	movq	$0, (%rsp)
	movq	8(%rbx), %rsi
	xorl	%r14d, %r14d
	xorl	%ecx, %ecx
	movq	$-1, 32(%rsp)
.L4011:
	cmpq	%rsi, %rcx
	jnb	.L4015
	movq	(%rbx), %r11
	movq	%rcx, 8(%rbx)
	movb	$0, (%r11,%rcx)
	jmp	.L4059
	.p2align 4,,10
	.p2align 3
.L4065:
	subq	%rsi, %rcx
	xorl	%r8d, %r8d
	xorl	%edx, %edx
	movq	%rbx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
.LEHE9:
	jmp	.L4009
	.p2align 4,,10
	.p2align 3
.L4013:
	movq	32(%rsp), %r14
	cmpq	$-1, %r14
	jne	.L4058
.L4057:
	movq	$0, (%rsp)
	movq	%r12, %r14
	movq	(%rsp), %r10
	jmp	.L4014
	.p2align 4,,10
	.p2align 3
.L4064:
	cmpq	%r14, %r12
	jb	.L4066
	subq	%r14, %r12
	je	.L4016
	movq	%rcx, %rdi
	leaq	0(%rbp,%r14), %rsi
	movq	%r12, %rdx
	call	memcpy@PLT
	movq	%rax, %rcx
	addq	%r12, %rcx
	jmp	.L4016
	.p2align 4,,10
	.p2align 3
.L4012:
	testq	%rax, %rax
	je	.L4030
	movq	$-1, 32(%rsp)
	subq	%rbp, %r14
	cmpq	$-1, %r14
	jne	.L4058
	jmp	.L4057
.L4030:
	movq	$0, (%rsp)
	movq	%r12, %r14
	movq	(%rsp), %r10
	movq	$-1, 32(%rsp)
	jmp	.L4014
.L4061:
	call	__stack_chk_fail@PLT
.L4005:
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4061
.LEHB10:
	call	_ZSt16__throw_bad_castv@PLT
.LEHE10:
.L4066:
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4061
	movq	%r12, %rcx
	movq	%r14, %rdx
	leaq	.LC41(%rip), %rsi
	xorl	%eax, %eax
	leaq	.LC10(%rip), %rdi
.LEHB11:
	call	_ZSt24__throw_out_of_range_fmtPKcz@PLT
.LEHE11:
.L4032:
	endbr64
	movq	%rax, %r12
	vzeroupper
	jmp	.L4025
.L4033:
	endbr64
	movq	%rax, %r12
	jmp	.L4024
	.section	.gcc_except_table,"a",@progbits
.LLSDA14067:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE14067-.LLSDACSB14067
.LLSDACSB14067:
	.uleb128 .LEHB8-.LFB14067
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L4032-.LFB14067
	.uleb128 0
	.uleb128 .LEHB9-.LFB14067
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L4033-.LFB14067
	.uleb128 0
	.uleb128 .LEHB10-.LFB14067
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L4032-.LFB14067
	.uleb128 0
	.uleb128 .LEHB11-.LFB14067
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L4033-.LFB14067
	.uleb128 0
.LLSDACSE14067:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC14067
	.type	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0.cold, @function
_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0.cold:
.LFSB14067:
.L4024:
	.cfi_def_cfa_offset 160
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
	movq	40(%rsp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
.L4025:
	movq	%rbx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4067
	movq	%r12, %rdi
.LEHB12:
	call	_Unwind_Resume@PLT
.LEHE12:
.L4067:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14067:
	.section	.gcc_except_table
.LLSDAC14067:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC14067-.LLSDACSBC14067
.LLSDACSBC14067:
	.uleb128 .LEHB12-.LCOLDB42
	.uleb128 .LEHE12-.LEHB12
	.uleb128 0
	.uleb128 0
.LLSDACSEC14067:
	.section	.text.unlikely
	.text
	.size	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0, .-_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0
	.section	.text.unlikely
	.size	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0.cold, .-_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0.cold
.LCOLDE42:
	.text
.LHOTE42:
	.globl	__getf2
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_.str1.1,"aMS",@progbits,1
.LC44:
	.string	"basic_string::insert"
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_.str1.8,"aMS",@progbits,1
	.align 8
.LC45:
	.string	"%s: __pos (which is %zu) > this->size() (which is %zu)"
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13691:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13691
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	pushq	%rbx
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 10, -56
	.cfi_offset 3, -64
	movq	%rdi, %rbx
	subq	$400, %rsp
	movq	%rsi, -400(%rbp)
	vmovdqa	%xmm0, -384(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	leaq	-272(%rbp), %rax
	movb	$0, -272(%rbp)
	movq	%rax, -408(%rbp)
	movq	%rax, -288(%rbp)
	movzbl	1(%rdi), %eax
	movq	$0, -280(%rbp)
	movl	%eax, %edx
	andl	$6, %edx
	je	.L4069
	cmpb	$2, %dl
	je	.L4391
	movq	$-1, -392(%rbp)
	cmpb	$4, %dl
	je	.L4392
.L4071:
	movl	%eax, %edx
	shrb	$3, %dl
	andl	$15, %edx
	cmpb	$8, %dl
	ja	.L4075
	leaq	.L4112(%rip), %r11
	movzbl	%dl, %r10d
	movslq	(%r11,%r10,4), %rcx
	addq	%r11, %rcx
	notrack jmp	*%rcx
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4112:
	.long	.L4120-.L4112
	.long	.L4119-.L4112
	.long	.L4118-.L4112
	.long	.L4117-.L4112
	.long	.L4116-.L4112
	.long	.L4115-.L4112
	.long	.L4114-.L4112
	.long	.L4113-.L4112
	.long	.L4111-.L4112
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L4069:
	movl	%eax, %ecx
	shrb	$3, %cl
	andl	$15, %ecx
	cmpb	$8, %cl
	ja	.L4075
	leaq	.L4126(%rip), %rdi
	movzbl	%cl, %esi
	movslq	(%rdi,%rsi,4), %r8
	addq	%rdi, %r8
	notrack jmp	*%r8
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4126:
	.long	.L4131-.L4126
	.long	.L4130-.L4126
	.long	.L4129-.L4126
	.long	.L4217-.L4126
	.long	.L4218-.L4126
	.long	.L4219-.L4126
	.long	.L4220-.L4126
	.long	.L4221-.L4126
	.long	.L4222-.L4126
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L4131:
	vmovdqa	-384(%rbp), %xmm0
	leaq	-64(%rbp), %r14
	leaq	-191(%rbp), %r15
	xorl	%r12d, %r12d
	movq	%r14, %rsi
	movq	%r15, %rdi
	call	_ZSt8to_charsPcS_DF128_@PLT
	movb	$101, -417(%rbp)
	movl	$0, -416(%rbp)
.L4135:
	movb	$0, -424(%rbp)
	movq	$6, -392(%rbp)
	movb	$0, -418(%rbp)
.L4134:
	movq	%r14, -432(%rbp)
	movq	%rax, %r13
	movq	%r15, %r14
	cmpl	$75, %edx
	je	.L4393
.L4136:
	testb	%r12b, %r12b
	je	.L4151
	cmpq	%r13, %r14
	je	.L4151
	movq	%r13, %r9
	movq	%r14, %r12
	subq	%r14, %r9
	andl	$7, %r9d
	je	.L4152
	cmpq	$1, %r9
	je	.L4320
	cmpq	$2, %r9
	je	.L4321
	cmpq	$3, %r9
	je	.L4322
	cmpq	$4, %r9
	je	.L4323
	cmpq	$5, %r9
	je	.L4324
	cmpq	$6, %r9
	je	.L4325
	movsbl	(%r14), %edi
	leaq	1(%r14), %r12
	call	toupper@PLT
	movb	%al, (%r14)
.L4325:
	movsbl	(%r12), %edi
	addq	$1, %r12
	call	toupper@PLT
	movb	%al, -1(%r12)
.L4324:
	movsbl	(%r12), %edi
	addq	$1, %r12
	call	toupper@PLT
	movb	%al, -1(%r12)
.L4323:
	movsbl	(%r12), %edi
	addq	$1, %r12
	call	toupper@PLT
	movb	%al, -1(%r12)
.L4322:
	movsbl	(%r12), %edi
	addq	$1, %r12
	call	toupper@PLT
	movb	%al, -1(%r12)
.L4321:
	movsbl	(%r12), %edi
	addq	$1, %r12
	call	toupper@PLT
	movb	%al, -1(%r12)
.L4320:
	movsbl	(%r12), %edi
	addq	$1, %r12
	call	toupper@PLT
	movb	%al, -1(%r12)
	cmpq	%r12, %r13
	je	.L4151
.L4152:
	movsbl	(%r12), %edi
	addq	$8, %r12
	call	toupper@PLT
	movsbl	-7(%r12), %edi
	movb	%al, -8(%r12)
	call	toupper@PLT
	movsbl	-6(%r12), %edi
	movb	%al, -7(%r12)
	call	toupper@PLT
	movsbl	-5(%r12), %edi
	movb	%al, -6(%r12)
	call	toupper@PLT
	movsbl	-4(%r12), %edi
	movb	%al, -5(%r12)
	call	toupper@PLT
	movsbl	-3(%r12), %edi
	movb	%al, -4(%r12)
	call	toupper@PLT
	movsbl	-2(%r12), %edi
	movb	%al, -3(%r12)
	call	toupper@PLT
	movsbl	-1(%r12), %edi
	movb	%al, -2(%r12)
	call	toupper@PLT
	movb	%al, -1(%r12)
	cmpq	%r12, %r13
	jne	.L4152
	.p2align 4,,10
	.p2align 3
.L4151:
	movzbl	(%rbx), %edx
	vmovdqa	-384(%rbp), %xmm0
	vpxor	%xmm1, %xmm1, %xmm1
	movb	%dl, -416(%rbp)
	call	__getf2@PLT
	movzbl	-416(%rbp), %ecx
	testq	%rax, %rax
	js	.L4386
	movl	%ecx, %eax
	andl	$12, %eax
	cmpb	$4, %al
	je	.L4394
	xorl	%r10d, %r10d
	cmpb	$12, %al
	je	.L4395
.L4150:
	movq	%r13, %r15
	subq	%r14, %r15
	testb	$16, %cl
	je	.L4154
	testq	%r15, %r15
	je	.L4232
	movq	%r15, %rdx
	movl	$46, %esi
	movq	%r14, %rdi
	movb	%cl, -384(%rbp)
	movq	%r10, -416(%rbp)
	call	memchr@PLT
	movzbl	-384(%rbp), %ecx
	movq	-416(%rbp), %r9
	testq	%rax, %rax
	movq	%rax, %r12
	je	.L4156
	subq	%r14, %r12
	cmpq	$-1, %r12
	je	.L4156
	leaq	1(%r12), %r8
	movq	%r15, -384(%rbp)
	cmpq	%r15, %r8
	jnb	.L4157
	movsbl	-417(%rbp), %esi
	movq	%r15, %rdx
	leaq	(%r14,%r8), %rdi
	movq	%r9, -440(%rbp)
	subq	%r8, %rdx
	movb	%cl, -424(%rbp)
	movq	%r8, -416(%rbp)
	call	memchr@PLT
	movq	-416(%rbp), %r8
	movzbl	-424(%rbp), %ecx
	testq	%rax, %rax
	movq	-440(%rbp), %r9
	je	.L4157
	subq	%r14, %rax
	cmpq	$-1, %rax
	cmove	%r15, %rax
	movq	%rax, -384(%rbp)
.L4157:
	movq	-384(%rbp), %rdi
	cmpq	%r12, %rdi
	sete	%dl
	sete	-416(%rbp)
	cmpb	$0, -418(%rbp)
	movzbl	%dl, %r12d
	jne	.L4396
	movq	$0, -392(%rbp)
.L4158:
	testq	%r12, %r12
	je	.L4154
.L4164:
	cmpq	$0, -280(%rbp)
	jne	.L4165
	movq	-432(%rbp), %rcx
	subq	%r13, %rcx
	cmpq	%r12, %rcx
	jnb	.L4397
.L4165:
	leaq	-288(%rbp), %rdi
	leaq	(%r15,%r12), %rsi
	movq	%rdi, -432(%rbp)
.LEHB13:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	-280(%rbp), %rcx
	movq	-384(%rbp), %r13
	testq	%rcx, %rcx
	jne	.L4169
	cmpq	%r13, %r15
	movq	%r13, %r8
	movq	-432(%rbp), %rdi
	movq	%r14, %rcx
	cmovbe	%r15, %r8
	xorl	%edx, %edx
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	cmpb	$0, -416(%rbp)
	jne	.L4398
.L4170:
	movq	-392(%rbp), %rsi
	testq	%rsi, %rsi
	jne	.L4399
.L4171:
	movq	-384(%rbp), %rdx
	movq	$-1, %rcx
	movq	%r14, %rsi
	movq	%r15, %rdi
	call	_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0
.LEHE13:
	movq	-280(%rbp), %rsi
	movq	%rax, %r8
	movq	%rdx, %rcx
	movabsq	$9223372036854775807, %r12
	subq	%rsi, %r12
	cmpq	%rax, %r12
	jb	.L4400
	leaq	(%rax,%rsi), %r13
	movq	-408(%rbp), %r9
	movq	-288(%rbp), %rax
	cmpq	%r9, %rax
	je	.L4239
	movq	-272(%rbp), %rdx
.L4174:
	cmpq	%r13, %rdx
	jb	.L4175
	testq	%r8, %r8
	je	.L4176
	leaq	(%rax,%rsi), %rdi
	cmpq	$1, %r8
	je	.L4401
	movq	%r8, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
	movq	-288(%rbp), %rax
.L4176:
	movq	%r13, -280(%rbp)
	movb	$0, (%rax,%r13)
.L4178:
	movq	-280(%rbp), %r15
	movq	-288(%rbp), %r14
	movzbl	(%rbx), %ecx
	.p2align 4,,10
	.p2align 3
.L4154:
	leaq	-240(%rbp), %r13
	andl	$32, %ecx
	movq	$0, -368(%rbp)
	movb	$0, -360(%rbp)
	movq	%r13, -256(%rbp)
	movq	$0, -248(%rbp)
	movb	$0, -240(%rbp)
	jne	.L4402
.L4181:
	movq	%r14, -384(%rbp)
.L4192:
	movzwl	(%rbx), %r11d
	andw	$384, %r11w
	cmpw	$128, %r11w
	je	.L4403
	cmpw	$256, %r11w
	je	.L4195
	movq	-400(%rbp), %rbx
	movq	16(%rbx), %r12
.L4198:
	testq	%r15, %r15
	jne	.L4404
.L4200:
	movq	-256(%rbp), %rdi
	cmpq	%r13, %rdi
	je	.L4204
	movq	-240(%rbp), %r13
	leaq	1(%r13), %rsi
	call	_ZdlPvm@PLT
.L4204:
	cmpb	$0, -360(%rbp)
	jne	.L4405
.L4205:
	movq	-288(%rbp), %rdi
	movq	-408(%rbp), %r15
	cmpq	%r15, %rdi
	je	.L4206
	movq	-272(%rbp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L4206:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4387
	addq	$400, %rsp
	movq	%r12, %rax
	popq	%rbx
	popq	%r10
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L4395:
	.cfi_restore_state
	movb	$32, -1(%r14)
	movzbl	(%rbx), %ecx
	subq	$1, %r14
.L4386:
	movl	$1, %r10d
	jmp	.L4150
	.p2align 4,,10
	.p2align 3
.L4403:
	movzwl	2(%rbx), %eax
.L4194:
	movq	-400(%rbp), %rdi
	movq	16(%rdi), %r12
	cmpq	%rax, %r15
	jnb	.L4198
	movzbl	(%rbx), %esi
	subq	%r15, %rax
	movsbl	6(%rbx), %r9d
	movq	%rax, %rbx
	movl	%esi, %ecx
	andl	$3, %ecx
	jne	.L4202
	andl	$64, %esi
	je	.L4240
	movzbl	(%r14), %r14d
	movl	$48, %r9d
	movl	$2, %ecx
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %rdx
	cmpb	$15, (%rdx,%r14)
	jbe	.L4202
	movq	-384(%rbp), %r9
	movq	24(%r12), %rax
	movzbl	(%r9), %r10d
	leaq	1(%rax), %rcx
	movq	%rcx, 24(%r12)
	movb	%r10b, (%rax)
	movq	24(%r12), %r8
	subq	8(%r12), %r8
	cmpq	16(%r12), %r8
	je	.L4406
.L4203:
	subq	$1, %r15
	movl	$48, %r9d
	movl	$2, %ecx
	addq	$1, -384(%rbp)
	.p2align 4,,10
	.p2align 3
.L4202:
	movq	-384(%rbp), %rdx
	movq	%rbx, %r8
	movq	%r15, %rsi
	movq	%r12, %rdi
.LEHB14:
	call	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
.LEHE14:
	movq	%rax, %r12
	jmp	.L4200
	.p2align 4,,10
	.p2align 3
.L4405:
	leaq	-368(%rbp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L4205
	.p2align 4,,10
	.p2align 3
.L4402:
	movq	-400(%rbp), %rdi
	cmpb	$0, 32(%rdi)
	leaq	24(%rdi), %rsi
	je	.L4407
.L4182:
	leaq	-320(%rbp), %r12
	movq	%r12, %rdi
	call	_ZNSt6localeC1ERKS_@PLT
	movq	%r12, %r8
	movq	%r15, %rsi
	movq	%r14, %rdx
	movsbl	-417(%rbp), %ecx
	leaq	-224(%rbp), %rdi
	movq	%rdi, -384(%rbp)
.LEHB15:
	call	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0
.LEHE15:
	movq	-224(%rbp), %rsi
	leaq	-208(%rbp), %r9
	movq	-256(%rbp), %rdi
	vmovq	-216(%rbp), %xmm1
	cmpq	%r9, %rsi
	je	.L4408
	vpinsrq	$1, -208(%rbp), %xmm1, %xmm0
	cmpq	%r13, %rdi
	je	.L4409
	movq	-240(%rbp), %rdx
	movq	%rsi, -256(%rbp)
	vmovdqu	%xmm0, -248(%rbp)
	testq	%rdi, %rdi
	je	.L4190
	movq	%rdi, -224(%rbp)
	movq	%rdx, -208(%rbp)
.L4189:
	movq	$0, -216(%rbp)
	movb	$0, (%rdi)
	movq	-384(%rbp), %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	%r12, %rdi
	call	_ZNSt6localeD1Ev@PLT
	movq	-248(%rbp), %r12
	testq	%r12, %r12
	je	.L4181
	movq	-256(%rbp), %r15
	movq	%r15, -384(%rbp)
	movq	%r12, %r15
	jmp	.L4192
	.p2align 4,,10
	.p2align 3
.L4394:
	movb	$43, -1(%r14)
	movl	$1, %r10d
	movzbl	(%rbx), %ecx
	subq	$1, %r14
	jmp	.L4150
	.p2align 4,,10
	.p2align 3
.L4391:
	movzwl	4(%rdi), %r9d
	movq	%r9, -392(%rbp)
	jmp	.L4071
	.p2align 4,,10
	.p2align 3
.L4240:
	movl	$32, %r9d
	movl	$2, %ecx
	jmp	.L4202
	.p2align 4,,10
	.p2align 3
.L4396:
	subq	%r9, %rdi
	subq	$1, %rdi
	cmpb	$48, (%r14,%r9)
	je	.L4410
.L4162:
	cmpq	$0, -392(%rbp)
	je	.L4158
.L4212:
	subq	%rdi, -392(%rbp)
	movq	-392(%rbp), %r11
	addq	%r11, %r12
	jmp	.L4158
	.p2align 4,,10
	.p2align 3
.L4113:
	movb	$101, -417(%rbp)
	movl	-392(%rbp), %ecx
	xorl	%r12d, %r12d
.L4122:
	movl	$3, -416(%rbp)
	movb	$1, -418(%rbp)
.L4213:
	movl	-416(%rbp), %edx
	vmovdqa	-384(%rbp), %xmm0
	leaq	-64(%rbp), %r14
	leaq	-191(%rbp), %r15
	movq	%r14, %rsi
	movq	%r15, %rdi
	call	_ZSt8to_charsPcS_DF128_St12chars_formati@PLT
	movb	$1, -424(%rbp)
	jmp	.L4134
	.p2align 4,,10
	.p2align 3
.L4118:
	andl	$120, %eax
	movl	$80, %r14d
	movl	$112, %r15d
	movl	$1, %r12d
	cmpb	$16, %al
	cmove	%r14d, %r15d
	movb	%r15b, -417(%rbp)
.L4121:
	movl	$4, -416(%rbp)
	movl	-392(%rbp), %ecx
	movb	$0, -418(%rbp)
	jmp	.L4213
	.p2align 4,,10
	.p2align 3
.L4119:
	andl	$120, %eax
	movl	$112, %r12d
	movl	$101, %r13d
	cmpb	$16, %al
	cmovne	%r12d, %r13d
	xorl	%r12d, %r12d
	movb	%r13b, -417(%rbp)
	jmp	.L4121
	.p2align 4,,10
	.p2align 3
.L4120:
	movb	$101, -417(%rbp)
	movl	-392(%rbp), %ecx
	xorl	%r12d, %r12d
	movl	$3, -416(%rbp)
	movb	$0, -418(%rbp)
	jmp	.L4213
	.p2align 4,,10
	.p2align 3
.L4111:
	movl	-392(%rbp), %ecx
.L4125:
	movb	$69, -417(%rbp)
	movl	$1, %r12d
	jmp	.L4122
	.p2align 4,,10
	.p2align 3
.L4114:
	movl	-392(%rbp), %ecx
.L4127:
	movl	$1, %r12d
.L4123:
	movl	$2, -416(%rbp)
	movb	$101, -417(%rbp)
	movb	$0, -418(%rbp)
	jmp	.L4213
	.p2align 4,,10
	.p2align 3
.L4115:
	movl	-392(%rbp), %ecx
	xorl	%r12d, %r12d
	jmp	.L4123
	.p2align 4,,10
	.p2align 3
.L4116:
	movl	-392(%rbp), %ecx
.L4128:
	movb	$69, -417(%rbp)
	movl	$1, %r12d
.L4124:
	movl	$1, -416(%rbp)
	movb	$0, -418(%rbp)
	jmp	.L4213
	.p2align 4,,10
	.p2align 3
.L4117:
	movb	$101, -417(%rbp)
	movl	-392(%rbp), %ecx
	xorl	%r12d, %r12d
	jmp	.L4124
	.p2align 4,,10
	.p2align 3
.L4222:
	movq	$6, -392(%rbp)
	movl	$6, %ecx
	jmp	.L4125
	.p2align 4,,10
	.p2align 3
.L4220:
	movq	$6, -392(%rbp)
	movl	$6, %ecx
	jmp	.L4127
	.p2align 4,,10
	.p2align 3
.L4221:
	movb	$101, -417(%rbp)
	movl	$6, %ecx
	xorl	%r12d, %r12d
	movq	$6, -392(%rbp)
	jmp	.L4122
	.p2align 4,,10
	.p2align 3
.L4130:
	andl	$120, %eax
	movl	$101, %r11d
	movl	$112, %r12d
	cmpb	$16, %al
	cmove	%r11d, %r12d
	movb	%r12b, -417(%rbp)
	xorl	%r12d, %r12d
.L4132:
	leaq	-64(%rbp), %r14
	leaq	-191(%rbp), %r15
	movl	$4, %edx
	movq	%r14, %rsi
	movq	%r15, %rdi
	call	_ZSt8to_charsPcS_DF128_St12chars_format@PLT
	movl	$4, -416(%rbp)
	jmp	.L4135
	.p2align 4,,10
	.p2align 3
.L4129:
	andl	$120, %eax
	movl	$80, %r9d
	movl	$112, %r10d
	movl	$1, %r12d
	cmpb	$16, %al
	cmove	%r9d, %r10d
	movb	%r10b, -417(%rbp)
	jmp	.L4132
	.p2align 4,,10
	.p2align 3
.L4217:
	movb	$101, -417(%rbp)
	movl	$6, %ecx
	xorl	%r12d, %r12d
	movq	$6, -392(%rbp)
	jmp	.L4124
	.p2align 4,,10
	.p2align 3
.L4218:
	movq	$6, -392(%rbp)
	movl	$6, %ecx
	jmp	.L4128
	.p2align 4,,10
	.p2align 3
.L4219:
	movq	$6, -392(%rbp)
	movl	$6, %ecx
	xorl	%r12d, %r12d
	jmp	.L4123
	.p2align 4,,10
	.p2align 3
.L4409:
	movq	%rsi, -256(%rbp)
	vmovdqu	%xmm0, -248(%rbp)
.L4190:
	movq	%r9, -224(%rbp)
	leaq	-208(%rbp), %rdi
	jmp	.L4189
	.p2align 4,,10
	.p2align 3
.L4232:
	movq	$0, -384(%rbp)
.L4155:
	cmpb	$0, -418(%rbp)
	jne	.L4163
	movb	$1, -416(%rbp)
	movl	$1, %r12d
	movq	$0, -392(%rbp)
	jmp	.L4164
	.p2align 4,,10
	.p2align 3
.L4404:
	movq	-384(%rbp), %rdx
	movq	%r15, %rsi
	movq	%r12, %rdi
.LEHB16:
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L4200
	.p2align 4,,10
	.p2align 3
.L4195:
	movzwl	2(%rbx), %edi
	movq	-400(%rbp), %rsi
	call	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
.LEHE16:
	jmp	.L4194
	.p2align 4,,10
	.p2align 3
.L4156:
	movsbl	-417(%rbp), %esi
	movq	%r15, %rdx
	movq	%r14, %rdi
	movb	%cl, -384(%rbp)
	movq	%r9, -416(%rbp)
	call	memchr@PLT
	movzbl	-384(%rbp), %ecx
	movq	-416(%rbp), %r10
	testq	%rax, %rax
	je	.L4237
	subq	%r14, %rax
	cmpq	$-1, %rax
	cmove	%r15, %rax
	movq	%rax, -384(%rbp)
	jmp	.L4155
	.p2align 4,,10
	.p2align 3
.L4407:
	movq	%rsi, %rdi
	movq	%rsi, -384(%rbp)
	call	_ZNSt6localeC1Ev@PLT
	movq	-400(%rbp), %rsi
	movb	$1, 32(%rsi)
	movq	-384(%rbp), %rsi
	jmp	.L4182
	.p2align 4,,10
	.p2align 3
.L4392:
	movq	%rsi, %r13
	movzwl	4(%rdi), %esi
	movzbl	0(%r13), %edx
	movl	%edx, %ecx
	andl	$15, %edx
	andl	$15, %ecx
	cmpq	%rdx, %rsi
	jb	.L4411
	testb	%cl, %cl
	jne	.L4074
	movq	-400(%rbp), %rdi
	movq	(%rdi), %r8
	movq	%r8, -392(%rbp)
	shrq	$4, %r8
	cmpq	%r8, %rsi
	jb	.L4412
.L4074:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4387
	leaq	-288(%rbp), %rdi
	movq	%rdi, -432(%rbp)
.LEHB17:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L4393:
	movq	-392(%rbp), %rax
	movl	$256, %edi
	leaq	8(%rax), %rsi
	cmpq	$128, %rsi
	cmovbe	%rdi, %rsi
	leaq	-288(%rbp), %rdi
	movq	%rdi, -432(%rbp)
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	cmpb	$0, -424(%rbp)
	movq	-288(%rbp), %r10
	jne	.L4413
	movl	-416(%rbp), %ecx
	testl	%ecx, %ecx
	je	.L4143
.L4139:
	movq	-408(%rbp), %rax
	cmpq	%rax, %r10
	je	.L4229
	movq	-272(%rbp), %rdi
	leaq	(%rdi,%rdi), %r13
.L4144:
	movq	-432(%rbp), %rdi
	movq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	-288(%rbp), %r14
	movl	-416(%rbp), %edx
	vmovdqa	-384(%rbp), %xmm0
	leaq	-1(%r14,%r13), %rsi
	leaq	1(%r14), %rdi
	call	_ZSt8to_charsPcS_DF128_St12chars_format@PLT
	movq	%rax, %r13
	movl	%edx, %r15d
	testl	%edx, %edx
	je	.L4390
	movq	-432(%rbp), %rdi
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	-288(%rbp), %r10
	movq	-280(%rbp), %r11
	cmpl	$75, %r15d
	je	.L4139
	.p2align 4,,10
	.p2align 3
.L4142:
	leaq	1(%r10), %r14
	addq	%r11, %r10
	movq	%r10, -432(%rbp)
	jmp	.L4136
	.p2align 4,,10
	.p2align 3
.L4169:
	cmpq	%r13, %rcx
	jb	.L4414
	movq	-432(%rbp), %rdi
	movq	%r13, %rsi
	movq	%r12, %rcx
	xorl	%edx, %edx
	movl	$48, %r8d
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	cmpb	$0, -416(%rbp)
	je	.L4178
	movq	-288(%rbp), %r15
	movq	-384(%rbp), %r14
	movb	$46, (%r15,%r14)
	jmp	.L4178
	.p2align 4,,10
	.p2align 3
.L4163:
	cmpq	$0, -392(%rbp)
	je	.L4242
	movq	-384(%rbp), %rdi
	movzbl	-418(%rbp), %esi
	movl	$1, %r12d
	movb	%sil, -416(%rbp)
	subq	%r10, %rdi
	jmp	.L4212
	.p2align 4,,10
	.p2align 3
.L4408:
	vmovq	%xmm1, %rax
	testq	%rax, %rax
	je	.L4185
	cmpq	$1, %rax
	je	.L4415
	vmovq	%xmm1, %rdx
	call	memcpy@PLT
	vmovq	-216(%rbp), %xmm1
	movq	-256(%rbp), %rdi
.L4185:
	vmovq	%xmm1, %r8
	vmovq	%xmm1, -248(%rbp)
	movb	$0, (%rdi,%r8)
	movq	-224(%rbp), %rdi
	jmp	.L4189
	.p2align 4,,10
	.p2align 3
.L4410:
	cmpq	%r15, %r8
	jnb	.L4371
	movq	%r8, %rax
	notq	%rax
	addq	%r15, %rax
	andl	$7, %eax
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
	cmpq	%r15, %r8
	jnb	.L4371
	testq	%rax, %rax
	je	.L4160
	cmpq	$1, %rax
	je	.L4328
	cmpq	$2, %rax
	je	.L4329
	cmpq	$3, %rax
	je	.L4330
	cmpq	$4, %rax
	je	.L4331
	cmpq	$5, %rax
	je	.L4332
	cmpq	$6, %rax
	je	.L4333
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
.L4333:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
.L4332:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
.L4331:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
.L4330:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
.L4329:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
.L4328:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	addq	$1, %r8
	cmpq	%r15, %r8
	jnb	.L4371
.L4160:
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	leaq	1(%r8), %r10
	cmpb	$48, (%r14,%r10)
	movq	%r10, %r8
	jne	.L4161
	addq	$1, %r8
	cmpb	$48, (%r14,%r8)
	jne	.L4161
	cmpb	$48, 2(%r14,%r10)
	leaq	2(%r10), %r8
	jne	.L4161
	cmpb	$48, 3(%r14,%r10)
	leaq	3(%r10), %r8
	jne	.L4161
	cmpb	$48, 4(%r14,%r10)
	leaq	4(%r10), %r8
	jne	.L4161
	cmpb	$48, 5(%r14,%r10)
	leaq	5(%r10), %r8
	jne	.L4161
	cmpb	$48, 6(%r14,%r10)
	leaq	6(%r10), %r8
	jne	.L4161
	leaq	7(%r10), %r8
	cmpq	%r15, %r8
	jb	.L4160
.L4371:
	movq	$-1, %r8
.L4161:
	movq	-384(%rbp), %rdi
	subq	%r8, %rdi
	jmp	.L4162
	.p2align 4,,10
	.p2align 3
.L4397:
	movq	-384(%rbp), %r8
	movq	%r15, %rdx
	leaq	(%r14,%r8), %r13
	leaq	(%r8,%r12), %rdi
	subq	%r8, %rdx
	addq	%r14, %rdi
	movq	%r13, %rsi
	call	memmove@PLT
	cmpb	$0, -416(%rbp)
	jne	.L4416
.L4167:
	movq	-392(%rbp), %rdx
	movl	$48, %esi
	movq	%r13, %rdi
	addq	%r12, %r15
	call	memset@PLT
	movzbl	(%rbx), %ecx
	jmp	.L4154
	.p2align 4,,10
	.p2align 3
.L4399:
	movq	%rsi, %rcx
	movq	-432(%rbp), %rdi
	movq	-280(%rbp), %rsi
	xorl	%edx, %edx
	movl	$48, %r8d
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	jmp	.L4171
.L4242:
	movzbl	-418(%rbp), %r11d
	movl	$1, %r12d
	movb	%r11b, -416(%rbp)
	jmp	.L4164
.L4141:
	movq	-432(%rbp), %rdi
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	-288(%rbp), %r10
	movq	-280(%rbp), %r11
	cmpl	$75, %r14d
	jne	.L4142
.L4143:
	movq	-408(%rbp), %rsi
	cmpq	%rsi, %r10
	je	.L4228
	movq	-272(%rbp), %r8
	leaq	(%r8,%r8), %r14
.L4140:
	movq	-432(%rbp), %rdi
	movq	%r14, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	-288(%rbp), %r15
	vmovdqa	-384(%rbp), %xmm0
	leaq	-1(%r15,%r14), %rsi
	leaq	1(%r15), %rdi
	call	_ZSt8to_charsPcS_DF128_@PLT
	movq	%rax, %r13
	movl	%edx, %r14d
	testl	%edx, %edx
	jne	.L4141
	movq	%rax, %rsi
	movq	-432(%rbp), %rdi
	subq	%r15, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE17:
.L4385:
	movq	-288(%rbp), %r10
	movq	-280(%rbp), %r11
	jmp	.L4142
.L4416:
	movq	-384(%rbp), %r11
	movb	$46, 0(%r13)
	leaq	1(%r14,%r11), %r13
	jmp	.L4167
.L4406:
	movq	(%r12), %r11
	movq	%r12, %rdi
.LEHB18:
	call	*(%r11)
.LEHE18:
	jmp	.L4203
.L4390:
	movq	%r13, %rsi
	movq	-432(%rbp), %rdi
	subq	%r14, %rsi
.LEHB19:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	jmp	.L4385
.L4411:
	movq	-400(%rbp), %r11
	leaq	(%rsi,%rsi,4), %rcx
	salq	$4, %rsi
	movq	(%r11), %rdx
	addq	8(%r11), %rsi
	vmovdqa	(%rsi), %xmm3
	movq	%rdx, %r10
	movq	%rdx, -392(%rbp)
	shrq	$4, %r10
	vmovdqa	%xmm3, -320(%rbp)
	shrq	%cl, %r10
	andl	$31, %r10d
.L4073:
	leaq	.L4077(%rip), %r15
	movzbl	%r10b, %r14d
	movb	%r10b, -304(%rbp)
	vmovdqu	-320(%rbp), %ymm2
	movslq	(%r15,%r14,4), %r12
	vmovdqu	%ymm2, -352(%rbp)
	addq	%r15, %r12
	notrack jmp	*%r12
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4077:
	.long	.L4383-.L4077
	.long	.L4092-.L4077
	.long	.L4091-.L4077
	.long	.L4090-.L4077
	.long	.L4089-.L4077
	.long	.L4088-.L4077
	.long	.L4087-.L4077
	.long	.L4086-.L4077
	.long	.L4085-.L4077
	.long	.L4084-.L4077
	.long	.L4083-.L4077
	.long	.L4082-.L4077
	.long	.L4081-.L4077
	.long	.L4080-.L4077
	.long	.L4079-.L4077
	.long	.L4078-.L4077
	.long	.L4076-.L4077
	.long	.L4076-.L4077
	.long	.L4076-.L4077
	.long	.L4076-.L4077
	.long	.L4076-.L4077
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
.L4087:
	movq	-352(%rbp), %rdx
	movq	%rdx, -392(%rbp)
	vzeroupper
	jmp	.L4071
.L4088:
	movq	-352(%rbp), %rcx
	testq	%rcx, %rcx
	js	.L4099
	movq	%rcx, -392(%rbp)
	vzeroupper
	jmp	.L4071
.L4089:
	movl	-352(%rbp), %r15d
	movq	%r15, -392(%rbp)
	vzeroupper
	jmp	.L4071
.L4090:
	movl	-352(%rbp), %r12d
	testl	%r12d, %r12d
	js	.L4096
	movslq	%r12d, %r13
	movq	%r13, -392(%rbp)
	vzeroupper
	jmp	.L4071
.L4383:
	vzeroupper
	jmp	.L4074
.L4413:
	movl	-392(%rbp), %esi
	movl	%esi, -424(%rbp)
.L4138:
	movq	-408(%rbp), %r8
	cmpq	%r8, %r10
	je	.L4230
	movq	-272(%rbp), %r9
	leaq	(%r9,%r9), %r15
.L4146:
	movq	-432(%rbp), %rdi
	movq	%r15, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	-288(%rbp), %r14
	movl	-424(%rbp), %ecx
	movl	-416(%rbp), %edx
	vmovdqa	-384(%rbp), %xmm0
	leaq	-1(%r14,%r15), %rsi
	leaq	1(%r14), %rdi
	call	_ZSt8to_charsPcS_DF128_St12chars_formati@PLT
	movq	%rax, %r13
	movl	%edx, %r15d
	testl	%edx, %edx
	je	.L4390
	movq	-432(%rbp), %rdi
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	-288(%rbp), %r10
	movq	-280(%rbp), %r11
	cmpl	$75, %r15d
	jne	.L4142
	jmp	.L4138
.L4175:
	movq	-432(%rbp), %rdi
	xorl	%edx, %edx
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	movq	-288(%rbp), %rax
	jmp	.L4176
.L4237:
	movq	%r15, -384(%rbp)
	jmp	.L4155
.L4398:
	movq	-432(%rbp), %rdi
	movl	$46, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
	jmp	.L4170
.L4412:
	salq	$5, %rsi
	addq	8(%rdi), %rsi
	vmovdqu	(%rsi), %xmm4
	vmovdqa	%xmm4, -320(%rbp)
	movzbl	16(%rsi), %r9d
	movb	%r9b, -304(%rbp)
	movzbl	16(%rsi), %r10d
	jmp	.L4073
.L4239:
	movl	$15, %edx
	jmp	.L4174
.L4401:
	movzbl	(%rcx), %r10d
	movb	%r10b, (%rdi)
	movq	-288(%rbp), %rax
	jmp	.L4176
.L4415:
	movzbl	-208(%rbp), %ecx
	movb	%cl, (%rdi)
	vmovq	-216(%rbp), %xmm1
	movq	-256(%rbp), %rdi
	jmp	.L4185
.L4229:
	movl	$30, %r13d
	jmp	.L4144
.L4230:
	movl	$30, %r15d
	jmp	.L4146
.L4228:
	movl	$30, %r14d
	jmp	.L4140
.L4092:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	je	.L4094
.L4388:
	vzeroupper
.L4387:
	call	__stack_chk_fail@PLT
.L4414:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4387
	movq	-384(%rbp), %rdx
	leaq	.LC44(%rip), %rsi
	leaq	.LC45(%rip), %rdi
	xorl	%eax, %eax
	call	_ZSt24__throw_out_of_range_fmtPKcz@PLT
.L4400:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4387
	leaq	.LC33(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L4099:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r14
	leaq	.LC18(%rip), %rdi
	movq	%r14, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4078:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %rbx
	leaq	.LC18(%rip), %rdi
	movq	%rbx, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.LEHE19:
.L4243:
	endbr64
	movq	%rax, %rbx
	jmp	.L4210
.L4207:
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt6localeD1Ev@PLT
.L4208:
	leaq	-256(%rbp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	cmpb	$0, -360(%rbp)
	jne	.L4417
.L4209:
	leaq	-288(%rbp), %rdi
	movq	%rdi, -432(%rbp)
.L4210:
	movq	-432(%rbp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4387
	movq	%rbx, %rdi
.LEHB20:
	call	_Unwind_Resume@PLT
.LEHE20:
.L4091:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %rax
	leaq	.LC18(%rip), %rdi
	movq	%rax, -432(%rbp)
	vzeroupper
.LEHB21:
	call	_ZSt20__throw_format_errorPKc
.L4094:
	leaq	-288(%rbp), %rsi
	leaq	.LC18(%rip), %rdi
	movq	%rsi, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4076:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r8
	leaq	.LC18(%rip), %rdi
	movq	%r8, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4096:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %rbx
	leaq	.LC18(%rip), %rdi
	movq	%rbx, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4075:
.L4245:
	endbr64
	movq	%rax, %rbx
	jmp	.L4208
.L4244:
	endbr64
	movq	%rax, %rbx
	jmp	.L4207
.L4417:
	leaq	-368(%rbp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L4209
.L4079:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %rax
	leaq	.LC18(%rip), %rdi
	movq	%rax, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4080:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r13
	leaq	.LC18(%rip), %rdi
	movq	%r13, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4081:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %rsi
	leaq	.LC18(%rip), %rdi
	movq	%rsi, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4082:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %rdi
	movq	%rdi, -432(%rbp)
	leaq	.LC18(%rip), %rdi
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4083:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r8
	leaq	.LC18(%rip), %rdi
	movq	%r8, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4084:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r9
	leaq	.LC18(%rip), %rdi
	movq	%r9, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4085:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r10
	leaq	.LC18(%rip), %rdi
	movq	%r10, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.L4086:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L4388
	leaq	-288(%rbp), %r11
	leaq	.LC18(%rip), %rdi
	movq	%r11, -432(%rbp)
	vzeroupper
	call	_ZSt20__throw_format_errorPKc
.LEHE21:
	.cfi_endproc
.LFE13691:
	.section	.gcc_except_table
.LLSDA13691:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13691-.LLSDACSB13691
.LLSDACSB13691:
	.uleb128 .LEHB13-.LFB13691
	.uleb128 .LEHE13-.LEHB13
	.uleb128 .L4243-.LFB13691
	.uleb128 0
	.uleb128 .LEHB14-.LFB13691
	.uleb128 .LEHE14-.LEHB14
	.uleb128 .L4245-.LFB13691
	.uleb128 0
	.uleb128 .LEHB15-.LFB13691
	.uleb128 .LEHE15-.LEHB15
	.uleb128 .L4244-.LFB13691
	.uleb128 0
	.uleb128 .LEHB16-.LFB13691
	.uleb128 .LEHE16-.LEHB16
	.uleb128 .L4245-.LFB13691
	.uleb128 0
	.uleb128 .LEHB17-.LFB13691
	.uleb128 .LEHE17-.LEHB17
	.uleb128 .L4243-.LFB13691
	.uleb128 0
	.uleb128 .LEHB18-.LFB13691
	.uleb128 .LEHE18-.LEHB18
	.uleb128 .L4245-.LFB13691
	.uleb128 0
	.uleb128 .LEHB19-.LFB13691
	.uleb128 .LEHE19-.LEHB19
	.uleb128 .L4243-.LFB13691
	.uleb128 0
	.uleb128 .LEHB20-.LFB13691
	.uleb128 .LEHE20-.LEHB20
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB21-.LFB13691
	.uleb128 .LEHE21-.LEHB21
	.uleb128 .L4243-.LFB13691
	.uleb128 0
.LLSDACSE13691:
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.size	_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13682:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13682
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$344, %rsp
	.cfi_def_cfa_offset 400
	movq	%rsi, 16(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 328(%rsp)
	xorl	%eax, %eax
	leaq	112(%rsp), %rax
	movb	$0, 112(%rsp)
	movq	%rax, 24(%rsp)
	movq	%rax, 96(%rsp)
	movzbl	1(%rdi), %eax
	movq	$0, 104(%rsp)
	movl	%eax, %ebx
	andl	$6, %ebx
	je	.L4419
	cmpb	$2, %bl
	je	.L4703
	movq	$-1, 8(%rsp)
	cmpb	$4, %bl
	je	.L4704
.L4421:
	movl	%eax, %edx
	leaq	.L4424(%rip), %r8
	shrb	$3, %dl
	andl	$15, %edx
	movslq	(%r8,%rdx,4), %r9
	addq	%r8, %r9
	notrack jmp	*%r9
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4424:
	.long	.L4523-.L4424
	.long	.L4431-.L4424
	.long	.L4696-.L4424
	.long	.L4429-.L4424
	.long	.L4697-.L4424
	.long	.L4427-.L4424
	.long	.L4698-.L4424
	.long	.L4425-.L4424
	.long	.L4699-.L4424
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L4419:
	movl	%eax, %edx
	shrb	$3, %dl
	andl	$15, %edx
	cmpb	$8, %dl
	ja	.L4422
	leaq	.L4519(%rip), %rcx
	movzbl	%dl, %esi
	movslq	(%rcx,%rsi,4), %rdi
	addq	%rcx, %rdi
	notrack jmp	*%rdi
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4519:
	.long	.L4439-.L4519
	.long	.L4546-.L4519
	.long	.L4547-.L4519
	.long	.L4548-.L4519
	.long	.L4549-.L4519
	.long	.L4550-.L4519
	.long	.L4551-.L4519
	.long	.L4552-.L4519
	.long	.L4553-.L4519
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L4439:
	leaq	320(%rsp), %r14
	leaq	193(%rsp), %rbx
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	movq	%r14, %rsi
	movq	%rbx, %rdi
	call	_ZSt8to_charsPcS_e@PLT
	popq	%r13
	.cfi_def_cfa_offset 408
	popq	%r15
	.cfi_def_cfa_offset 400
	xorl	%r13d, %r13d
	movq	$6, 8(%rsp)
	movq	%rax, %rbp
	xorl	%r15d, %r15d
	movb	$101, 53(%rsp)
.L4438:
	cmpl	$75, %edx
	je	.L4440
	movb	$0, 54(%rsp)
.L4437:
	testb	%r15b, %r15b
	je	.L4457
	cmpq	%rbp, %rbx
	je	.L4457
	movq	%rbp, %r15
	movq	%rbx, %r13
	subq	%rbx, %r15
	andl	$7, %r15d
	je	.L4458
	cmpq	$1, %r15
	je	.L4631
	cmpq	$2, %r15
	je	.L4632
	cmpq	$3, %r15
	je	.L4633
	cmpq	$4, %r15
	je	.L4634
	cmpq	$5, %r15
	je	.L4635
	cmpq	$6, %r15
	je	.L4636
	movsbl	(%rbx), %edi
	leaq	1(%rbx), %r13
	call	toupper@PLT
	movb	%al, (%rbx)
.L4636:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4635:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4634:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4633:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4632:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4631:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
	cmpq	%r13, %rbp
	je	.L4457
.L4458:
	movsbl	0(%r13), %edi
	addq	$8, %r13
	call	toupper@PLT
	movsbl	-7(%r13), %edi
	movb	%al, -8(%r13)
	call	toupper@PLT
	movsbl	-6(%r13), %edi
	movb	%al, -7(%r13)
	call	toupper@PLT
	movsbl	-5(%r13), %edi
	movb	%al, -6(%r13)
	call	toupper@PLT
	movsbl	-4(%r13), %edi
	movb	%al, -5(%r13)
	call	toupper@PLT
	movsbl	-3(%r13), %edi
	movb	%al, -4(%r13)
	call	toupper@PLT
	movsbl	-2(%r13), %edi
	movb	%al, -3(%r13)
	call	toupper@PLT
	movsbl	-1(%r13), %edi
	movb	%al, -2(%r13)
	call	toupper@PLT
	movb	%al, -1(%r13)
	cmpq	%r13, %rbp
	jne	.L4458
	.p2align 4,,10
	.p2align 3
.L4457:
	fldz
	movzbl	(%r12), %ecx
	fldt	400(%rsp)
	fcomip	%st(1), %st
	fstp	%st(0)
	jb	.L4701
	movl	%ecx, %edx
	andl	$12, %edx
	cmpb	$4, %dl
	je	.L4705
	xorl	%esi, %esi
	cmpb	$12, %dl
	je	.L4706
.L4456:
	movq	%rbp, %r13
	subq	%rbx, %r13
	testb	$16, %cl
	je	.L4460
	testq	%r13, %r13
	je	.L4535
	movq	%rsi, 40(%rsp)
	movq	%r13, %rdx
	movl	$46, %esi
	movq	%rbx, %rdi
	movb	%cl, 32(%rsp)
	call	memchr@PLT
	movzbl	32(%rsp), %ecx
	movq	40(%rsp), %r9
	testq	%rax, %rax
	movq	%rax, %r15
	je	.L4462
	subq	%rbx, %r15
	cmpq	$-1, %r15
	je	.L4462
	leaq	1(%r15), %r8
	movq	%r13, 32(%rsp)
	cmpq	%r13, %r8
	jnb	.L4463
	movsbl	53(%rsp), %esi
	movq	%r13, %rdx
	leaq	(%rbx,%r8), %rdi
	movb	%cl, 55(%rsp)
	subq	%r8, %rdx
	movq	%r9, 56(%rsp)
	movq	%r8, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r8
	movzbl	55(%rsp), %ecx
	testq	%rax, %rax
	movq	56(%rsp), %r9
	je	.L4463
	subq	%rbx, %rax
	cmpq	$-1, %rax
	cmove	%r13, %rax
	movq	%rax, 32(%rsp)
.L4463:
	movq	32(%rsp), %rax
	cmpq	%r15, %rax
	sete	%r10b
	sete	40(%rsp)
	cmpb	$0, 54(%rsp)
	movzbl	%r10b, %r15d
	jne	.L4707
	movq	$0, 8(%rsp)
.L4464:
	testq	%r15, %r15
	je	.L4460
.L4470:
	cmpq	$0, 104(%rsp)
	jne	.L4471
	subq	%rbp, %r14
	cmpq	%r15, %r14
	jnb	.L4708
.L4471:
	leaq	96(%rsp), %r14
	leaq	0(%r13,%r15), %rsi
	movq	%r14, %rdi
.LEHB22:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	104(%rsp), %rcx
	movq	32(%rsp), %rbp
	testq	%rcx, %rcx
	jne	.L4475
	cmpq	%rbp, %r13
	movq	%rbp, %r8
	movq	%rbx, %rcx
	movq	%r14, %rdi
	cmovbe	%r13, %r8
	xorl	%edx, %edx
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	cmpb	$0, 40(%rsp)
	jne	.L4709
.L4476:
	movq	8(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.L4710
.L4477:
	movq	32(%rsp), %rdx
	movq	$-1, %rcx
	movq	%rbx, %rsi
	movq	%r13, %rdi
	call	_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0
.LEHE22:
	movq	104(%rsp), %rsi
	movq	%rax, %r8
	movq	%rdx, %rcx
	movabsq	$9223372036854775807, %r13
	subq	%rsi, %r13
	cmpq	%rax, %r13
	jb	.L4711
	leaq	(%rax,%rsi), %r15
	movq	24(%rsp), %r9
	movq	96(%rsp), %rax
	cmpq	%r9, %rax
	je	.L4542
	movq	112(%rsp), %r10
.L4480:
	cmpq	%r15, %r10
	jb	.L4481
	testq	%r8, %r8
	je	.L4482
	leaq	(%rax,%rsi), %rdi
	cmpq	$1, %r8
	je	.L4712
	movq	%r8, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
	movq	96(%rsp), %rax
.L4482:
	movq	%r15, 104(%rsp)
	movb	$0, (%rax,%r15)
.L4484:
	movq	104(%rsp), %r13
	movq	96(%rsp), %rbx
	movzbl	(%r12), %ecx
	.p2align 4,,10
	.p2align 3
.L4460:
	leaq	144(%rsp), %r14
	andl	$32, %ecx
	movq	$0, 80(%rsp)
	movb	$0, 88(%rsp)
	movq	%r14, 128(%rsp)
	movq	$0, 136(%rsp)
	movb	$0, 144(%rsp)
	jne	.L4713
.L4487:
	movq	%rbx, %r15
.L4498:
	movzwl	(%r12), %edx
	andw	$384, %dx
	cmpw	$128, %dx
	je	.L4714
	cmpw	$256, %dx
	je	.L4501
	movq	16(%rsp), %r12
	movq	16(%r12), %rbp
.L4504:
	testq	%r13, %r13
	jne	.L4715
.L4506:
	movq	128(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.L4510
	movq	144(%rsp), %r14
	leaq	1(%r14), %rsi
	call	_ZdlPvm@PLT
.L4510:
	cmpb	$0, 88(%rsp)
	jne	.L4716
.L4511:
	movq	96(%rsp), %rdi
	movq	24(%rsp), %r15
	cmpq	%r15, %rdi
	je	.L4512
	movq	112(%rsp), %r13
	leaq	1(%r13), %rsi
	call	_ZdlPvm@PLT
.L4512:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4682
	addq	$344, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%rbp, %rax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L4706:
	.cfi_restore_state
	movb	$32, -1(%rbx)
	movzbl	(%r12), %ecx
	subq	$1, %rbx
.L4701:
	movl	$1, %esi
	jmp	.L4456
	.p2align 4,,10
	.p2align 3
.L4714:
	movzwl	2(%r12), %eax
.L4500:
	movq	16(%rsp), %rsi
	movq	16(%rsi), %rbp
	cmpq	%rax, %r13
	jnb	.L4504
	movzbl	(%r12), %r10d
	subq	%r13, %rax
	movsbl	6(%r12), %r9d
	movq	%rax, %r12
	movl	%r10d, %ecx
	andl	$3, %ecx
	jne	.L4508
	andl	$64, %r10d
	je	.L4543
	movzbl	(%rbx), %ebx
	movl	$48, %r9d
	movl	$2, %ecx
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %rax
	cmpb	$15, (%rax,%rbx)
	jbe	.L4508
	movq	24(%rbp), %r11
	movzbl	(%r15), %r9d
	leaq	1(%r11), %rcx
	movq	%rcx, 24(%rbp)
	movb	%r9b, (%r11)
	movq	24(%rbp), %r8
	subq	8(%rbp), %r8
	cmpq	16(%rbp), %r8
	je	.L4717
.L4509:
	addq	$1, %r15
	subq	$1, %r13
	movl	$48, %r9d
	movl	$2, %ecx
	.p2align 4,,10
	.p2align 3
.L4508:
	movq	%r12, %r8
	movq	%r13, %rsi
	movq	%r15, %rdx
	movq	%rbp, %rdi
.LEHB23:
	call	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
.LEHE23:
	movq	%rax, %rbp
	jmp	.L4506
	.p2align 4,,10
	.p2align 3
.L4716:
	leaq	80(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L4511
	.p2align 4,,10
	.p2align 3
.L4713:
	movq	16(%rsp), %rsi
	cmpb	$0, 32(%rsi)
	leaq	24(%rsi), %r15
	je	.L4718
.L4488:
	leaq	72(%rsp), %rbp
	movq	%r15, %rsi
	leaq	160(%rsp), %r15
	movq	%rbp, %rdi
	call	_ZNSt6localeC1ERKS_@PLT
	movsbl	53(%rsp), %ecx
	movq	%rbp, %r8
	movq	%r13, %rsi
	movq	%rbx, %rdx
	movq	%r15, %rdi
.LEHB24:
	call	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0
.LEHE24:
	movq	160(%rsp), %rsi
	leaq	176(%rsp), %r9
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm1
	cmpq	%r9, %rsi
	je	.L4719
	vpinsrq	$1, 176(%rsp), %xmm1, %xmm0
	cmpq	%r14, %rdi
	je	.L4720
	movq	144(%rsp), %r10
	movq	%rsi, 128(%rsp)
	vmovdqu	%xmm0, 136(%rsp)
	testq	%rdi, %rdi
	je	.L4496
	movq	%rdi, 160(%rsp)
	movq	%r10, 176(%rsp)
.L4495:
	movq	$0, 168(%rsp)
	movb	$0, (%rdi)
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	%rbp, %rdi
	call	_ZNSt6localeD1Ev@PLT
	movq	136(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L4487
	movq	128(%rsp), %r15
	movq	%rdi, %r13
	jmp	.L4498
	.p2align 4,,10
	.p2align 3
.L4705:
	movb	$43, -1(%rbx)
	movl	$1, %esi
	movzbl	(%r12), %ecx
	subq	$1, %rbx
	jmp	.L4456
	.p2align 4,,10
	.p2align 3
.L4703:
	movzwl	4(%rdi), %ebp
	movq	%rbp, 8(%rsp)
	jmp	.L4421
	.p2align 4,,10
	.p2align 3
.L4543:
	movl	$32, %r9d
	movl	$2, %ecx
	jmp	.L4508
	.p2align 4,,10
	.p2align 3
.L4707:
	subq	%r9, %rax
	subq	$1, %rax
	cmpb	$48, (%rbx,%r9)
	je	.L4721
.L4468:
	cmpq	$0, 8(%rsp)
	je	.L4464
.L4518:
	subq	%rax, 8(%rsp)
	movq	8(%rsp), %rsi
	addq	%rsi, %r15
	jmp	.L4464
	.p2align 4,,10
	.p2align 3
.L4552:
	movq	$6, 8(%rsp)
.L4425:
	movb	$101, 53(%rsp)
	xorl	%r15d, %r15d
.L4423:
	movb	$1, 54(%rsp)
	movl	$3, %r13d
.L4434:
	leaq	320(%rsp), %r14
	leaq	193(%rsp), %rbx
	movl	%r13d, %edx
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	movq	%r14, %rsi
	movq	%rbx, %rdi
	movl	24(%rsp), %ecx
	call	_ZSt8to_charsPcS_eSt12chars_formati@PLT
	popq	%rsi
	.cfi_def_cfa_offset 408
	popq	%rdi
	.cfi_def_cfa_offset 400
	movq	%rax, %rbp
	cmpl	$75, %edx
	jne	.L4437
	movq	8(%rsp), %rsi
	movl	$1, %ebx
	leaq	8(%rsi), %rbp
	cmpl	$2, %r13d
	je	.L4722
.L4441:
	cmpq	$128, %rbp
	movl	$256, %esi
	leaq	96(%rsp), %r14
	cmova	%rbp, %rsi
	movq	%r14, %rdi
.LEHB25:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	96(%rsp), %rdi
	testb	%bl, %bl
	jne	.L4723
	testl	%r13d, %r13d
	je	.L4449
.L4445:
	movq	24(%rsp), %rdx
	cmpq	%rdx, %rdi
	je	.L4532
	movq	112(%rsp), %rax
	leaq	(%rax,%rax), %rbp
.L4450:
	movq	%rbp, %rsi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %r8
	movl	%r13d, %edx
	movq	%r8, 32(%rsp)
	leaq	-1(%r8,%rbp), %rsi
	leaq	1(%r8), %rdi
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	call	_ZSt8to_charsPcS_eSt12chars_format@PLT
	popq	%rsi
	.cfi_def_cfa_offset 408
	popq	%rdi
	.cfi_def_cfa_offset 400
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L4451
	movq	32(%rsp), %r9
	movq	%rax, %rsi
	movq	%r14, %rdi
	subq	%r9, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE25:
.L4700:
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
.L4448:
	leaq	1(%rdi), %rbx
	leaq	(%rdi,%rcx), %r14
	jmp	.L4437
	.p2align 4,,10
	.p2align 3
.L4547:
	movq	$6, 8(%rsp)
.L4696:
	movl	$80, %r10d
	movl	$1, %r15d
.L4430:
	andl	$120, %eax
	movl	$112, %r11d
	cmpb	$16, %al
	cmove	%r10d, %r11d
	movb	%r11b, 53(%rsp)
	testb	%bl, %bl
	jne	.L4724
	leaq	320(%rsp), %r14
	leaq	193(%rsp), %rbx
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	movl	$4, %edx
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	movq	%r14, %rsi
	movq	%rbx, %rdi
	movl	$4, %r13d
	call	_ZSt8to_charsPcS_eSt12chars_format@PLT
	movq	%rax, %rbp
	popq	%rax
	.cfi_def_cfa_offset 408
	popq	%rcx
	.cfi_def_cfa_offset 400
	jmp	.L4438
	.p2align 4,,10
	.p2align 3
.L4551:
	movq	$6, 8(%rsp)
.L4698:
	movl	$1, %r15d
.L4426:
	movb	$101, 53(%rsp)
	movl	$2, %r13d
	movb	$0, 54(%rsp)
	jmp	.L4434
	.p2align 4,,10
	.p2align 3
.L4550:
	movq	$6, 8(%rsp)
.L4427:
	xorl	%r15d, %r15d
	jmp	.L4426
	.p2align 4,,10
	.p2align 3
.L4549:
	movq	$6, 8(%rsp)
.L4697:
	movb	$69, 53(%rsp)
	movl	$1, %r15d
.L4428:
	movb	$0, 54(%rsp)
	movl	$1, %r13d
	jmp	.L4434
	.p2align 4,,10
	.p2align 3
.L4548:
	movq	$6, 8(%rsp)
.L4429:
	movb	$101, 53(%rsp)
	xorl	%r15d, %r15d
	jmp	.L4428
	.p2align 4,,10
	.p2align 3
.L4553:
	movq	$6, 8(%rsp)
.L4699:
	movb	$69, 53(%rsp)
	movl	$1, %r15d
	jmp	.L4423
	.p2align 4,,10
	.p2align 3
.L4546:
	movq	$6, 8(%rsp)
.L4431:
	movl	$101, %r10d
	xorl	%r15d, %r15d
	jmp	.L4430
	.p2align 4,,10
	.p2align 3
.L4523:
	movb	$101, 53(%rsp)
	movl	$3, %r13d
	xorl	%r15d, %r15d
	movb	$0, 54(%rsp)
	jmp	.L4434
	.p2align 4,,10
	.p2align 3
.L4720:
	movq	%rsi, 128(%rsp)
	vmovdqu	%xmm0, 136(%rsp)
.L4496:
	movq	%r9, 160(%rsp)
	leaq	176(%rsp), %rdi
	jmp	.L4495
	.p2align 4,,10
	.p2align 3
.L4535:
	movq	$0, 32(%rsp)
.L4461:
	cmpb	$0, 54(%rsp)
	jne	.L4469
	movq	$0, 8(%rsp)
	movl	$1, %r15d
	movb	$1, 40(%rsp)
	jmp	.L4470
	.p2align 4,,10
	.p2align 3
.L4715:
	movq	%r13, %rsi
	movq	%r15, %rdx
	movq	%rbp, %rdi
.LEHB26:
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L4506
	.p2align 4,,10
	.p2align 3
.L4501:
	movzwl	2(%r12), %edi
	movq	16(%rsp), %rsi
	call	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
.LEHE26:
	jmp	.L4500
	.p2align 4,,10
	.p2align 3
.L4462:
	movsbl	53(%rsp), %esi
	movq	%r13, %rdx
	movq	%rbx, %rdi
	movb	%cl, 32(%rsp)
	movq	%r9, 40(%rsp)
	call	memchr@PLT
	movzbl	32(%rsp), %ecx
	movq	40(%rsp), %rsi
	testq	%rax, %rax
	je	.L4540
	subq	%rbx, %rax
	cmpq	$-1, %rax
	cmove	%r13, %rax
	movq	%rax, 32(%rsp)
	jmp	.L4461
	.p2align 4,,10
	.p2align 3
.L4718:
	movq	%r15, %rdi
	call	_ZNSt6localeC1Ev@PLT
	movq	16(%rsp), %rbp
	movb	$1, 32(%rbp)
	jmp	.L4488
	.p2align 4,,10
	.p2align 3
.L4704:
	movzwl	4(%rdi), %edi
	leaq	96(%rsp), %r14
.LEHB27:
	call	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
	movq	%rax, 8(%rsp)
	movzbl	1(%r12), %eax
	jmp	.L4421
	.p2align 4,,10
	.p2align 3
.L4475:
	cmpq	%rbp, %rcx
	jb	.L4725
	movq	%rbp, %rsi
	movl	$48, %r8d
	movq	%r15, %rcx
	xorl	%edx, %edx
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	cmpb	$0, 40(%rsp)
	je	.L4484
	movq	96(%rsp), %rbx
	movq	32(%rsp), %rcx
	movb	$46, (%rbx,%rcx)
	jmp	.L4484
	.p2align 4,,10
	.p2align 3
.L4469:
	cmpq	$0, 8(%rsp)
	je	.L4545
	movq	32(%rsp), %rax
	movzbl	54(%rsp), %r8d
	movl	$1, %r15d
	movb	%r8b, 40(%rsp)
	subq	%rsi, %rax
	jmp	.L4518
	.p2align 4,,10
	.p2align 3
.L4719:
	vmovq	%xmm1, %rax
	testq	%rax, %rax
	je	.L4491
	cmpq	$1, %rax
	je	.L4726
	vmovq	%xmm1, %rdx
	call	memcpy@PLT
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm1
.L4491:
	vmovq	%xmm1, 136(%rsp)
	vmovq	%xmm1, %r8
	movb	$0, (%rdi,%r8)
	movq	160(%rsp), %rdi
	jmp	.L4495
	.p2align 4,,10
	.p2align 3
.L4721:
	cmpq	%r13, %r8
	jnb	.L4683
	movq	%r8, %r11
	notq	%r11
	addq	%r13, %r11
	andl	$7, %r11d
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
	cmpq	%r13, %r8
	jnb	.L4683
	testq	%r11, %r11
	je	.L4466
	cmpq	$1, %r11
	je	.L4639
	cmpq	$2, %r11
	je	.L4640
	cmpq	$3, %r11
	je	.L4641
	cmpq	$4, %r11
	je	.L4642
	cmpq	$5, %r11
	je	.L4643
	cmpq	$6, %r11
	je	.L4644
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
.L4644:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
.L4643:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
.L4642:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
.L4641:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
.L4640:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
.L4639:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	addq	$1, %r8
	cmpq	%r13, %r8
	jnb	.L4683
.L4466:
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	leaq	1(%r8), %rdx
	cmpb	$48, (%rbx,%rdx)
	movq	%rdx, %r8
	jne	.L4467
	addq	$1, %r8
	cmpb	$48, (%rbx,%r8)
	jne	.L4467
	cmpb	$48, 2(%rbx,%rdx)
	leaq	2(%rdx), %r8
	jne	.L4467
	cmpb	$48, 3(%rbx,%rdx)
	leaq	3(%rdx), %r8
	jne	.L4467
	cmpb	$48, 4(%rbx,%rdx)
	leaq	4(%rdx), %r8
	jne	.L4467
	cmpb	$48, 5(%rbx,%rdx)
	leaq	5(%rdx), %r8
	jne	.L4467
	cmpb	$48, 6(%rbx,%rdx)
	leaq	6(%rdx), %r8
	jne	.L4467
	leaq	7(%rdx), %r8
	cmpq	%r13, %r8
	jb	.L4466
.L4683:
	movq	$-1, %r8
.L4467:
	movq	32(%rsp), %rax
	subq	%r8, %rax
	jmp	.L4468
	.p2align 4,,10
	.p2align 3
.L4708:
	movq	32(%rsp), %r8
	movq	%r13, %rdx
	leaq	(%rbx,%r8), %r14
	leaq	(%r15,%r8), %rdi
	subq	%r8, %rdx
	addq	%rbx, %rdi
	movq	%r14, %rsi
	call	memmove@PLT
	cmpb	$0, 40(%rsp)
	jne	.L4727
.L4473:
	movq	8(%rsp), %rdx
	movl	$48, %esi
	movq	%r14, %rdi
	addq	%r15, %r13
	call	memset@PLT
	movzbl	(%r12), %ecx
	jmp	.L4460
	.p2align 4,,10
	.p2align 3
.L4710:
	movq	104(%rsp), %rsi
	movq	%rdi, %rcx
	movl	$48, %r8d
	xorl	%edx, %edx
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	jmp	.L4477
.L4545:
	movzbl	54(%rsp), %edi
	movl	$1, %r15d
	movb	%dil, 40(%rsp)
	jmp	.L4470
.L4447:
	xorl	%esi, %esi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L4448
.L4449:
	movq	24(%rsp), %r10
	cmpq	%r10, %rdi
	je	.L4531
	movq	112(%rsp), %r11
	leaq	(%r11,%r11), %rbp
.L4446:
	movq	%rbp, %rsi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %r13
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	leaq	-1(%r13,%rbp), %rsi
	leaq	1(%r13), %rdi
	call	_ZSt8to_charsPcS_e@PLT
	popq	%r8
	.cfi_def_cfa_offset 408
	popq	%r9
	.cfi_def_cfa_offset 400
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L4447
	movq	%rax, %rsi
	movq	%r14, %rdi
	subq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE27:
	jmp	.L4700
.L4722:
	leaq	80(%rsp), %rdi
	movl	$0, 80(%rsp)
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	call	frexpl@PLT
	fstp	%st(0)
	movl	96(%rsp), %edi
	popq	%r10
	.cfi_def_cfa_offset 408
	popq	%r11
	.cfi_def_cfa_offset 400
	testl	%edi, %edi
	jle	.L4442
	imull	$4004, %edi, %edx
	imulq	$995517945, %rdx, %r8
	movq	%rdx, %rax
	shrq	$32, %r8
	subl	%r8d, %eax
	shrl	%eax
	addl	%r8d, %eax
	shrl	$13, %eax
	addl	$1, %eax
	addq	%rax, %rbp
.L4442:
	movl	$1, %ebx
	jmp	.L4441
.L4727:
	movq	32(%rsp), %rdx
	movb	$46, (%r14)
	leaq	1(%rbx,%rdx), %r14
	jmp	.L4473
.L4717:
	movq	0(%rbp), %rdx
	movq	%rbp, %rdi
.LEHB28:
	call	*(%rdx)
.LEHE28:
	jmp	.L4509
.L4440:
	movq	8(%rsp), %r14
	movb	$0, 54(%rsp)
	xorl	%ebx, %ebx
	leaq	8(%r14), %rbp
	jmp	.L4441
.L4723:
	movl	8(%rsp), %r9d
	movl	%r9d, 32(%rsp)
.L4444:
	movq	24(%rsp), %r10
	cmpq	%r10, %rdi
	je	.L4533
	movq	112(%rsp), %r11
	leaq	(%r11,%r11), %rbp
.L4452:
	movq	%rbp, %rsi
	movq	%r14, %rdi
.LEHB29:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rbx
	movl	%r13d, %edx
	movq	%rbx, 40(%rsp)
	leaq	-1(%rbx,%rbp), %rsi
	leaq	1(%rbx), %rdi
	pushq	408(%rsp)
	.cfi_def_cfa_offset 408
	pushq	408(%rsp)
	.cfi_def_cfa_offset 416
	movl	48(%rsp), %ecx
	call	_ZSt8to_charsPcS_eSt12chars_formati@PLT
	movq	%rdx, %rsi
	movl	%edx, %ebx
	movq	%rax, %rbp
	popq	%rdx
	.cfi_def_cfa_offset 408
	popq	%rcx
	.cfi_def_cfa_offset 400
	testl	%esi, %esi
	jne	.L4453
	movq	40(%rsp), %r13
	movq	%rbp, %rsi
	movq	%r14, %rdi
	subq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	jmp	.L4700
.L4481:
	xorl	%edx, %edx
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	movq	96(%rsp), %rax
	jmp	.L4482
.L4540:
	movq	%r13, 32(%rsp)
	jmp	.L4461
.L4709:
	movl	$46, %esi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
	jmp	.L4476
.L4451:
	xorl	%esi, %esi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L4448
	jmp	.L4445
.L4453:
	xorl	%esi, %esi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L4448
	jmp	.L4444
.L4542:
	movl	$15, %r10d
	jmp	.L4480
.L4712:
	movzbl	(%rcx), %r11d
	movb	%r11b, (%rdi)
	movq	96(%rsp), %rax
	jmp	.L4482
.L4726:
	movzbl	176(%rsp), %ecx
	movb	%cl, (%rdi)
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm1
	jmp	.L4491
.L4532:
	movl	$30, %ebp
	jmp	.L4450
.L4533:
	movl	$30, %ebp
	jmp	.L4452
.L4531:
	movl	$30, %ebp
	jmp	.L4446
.L4513:
	movq	%rbp, %rdi
	vzeroupper
	call	_ZNSt6localeD1Ev@PLT
.L4514:
	leaq	128(%rsp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	cmpb	$0, 88(%rsp)
	jne	.L4728
.L4515:
	leaq	96(%rsp), %r14
.L4516:
	movq	%r14, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	je	.L4517
.L4682:
	call	__stack_chk_fail@PLT
.L4724:
	movb	$0, 54(%rsp)
	movl	$4, %r13d
	jmp	.L4434
.L4711:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4682
	leaq	.LC33(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L4725:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4682
	movq	32(%rsp), %rdx
	leaq	.LC44(%rip), %rsi
	leaq	.LC45(%rip), %rdi
	xorl	%eax, %eax
	call	_ZSt24__throw_out_of_range_fmtPKcz@PLT
.LEHE29:
.L4422:
.L4556:
	endbr64
	movq	%rax, %r12
	jmp	.L4514
.L4555:
	endbr64
	movq	%rax, %r12
	jmp	.L4513
.L4554:
	endbr64
	movq	%rax, %r12
	jmp	.L4516
.L4728:
	leaq	80(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L4515
.L4517:
	movq	%r12, %rdi
.LEHB30:
	call	_Unwind_Resume@PLT
.LEHE30:
	.cfi_endproc
.LFE13682:
	.section	.gcc_except_table
.LLSDA13682:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13682-.LLSDACSB13682
.LLSDACSB13682:
	.uleb128 .LEHB22-.LFB13682
	.uleb128 .LEHE22-.LEHB22
	.uleb128 .L4554-.LFB13682
	.uleb128 0
	.uleb128 .LEHB23-.LFB13682
	.uleb128 .LEHE23-.LEHB23
	.uleb128 .L4556-.LFB13682
	.uleb128 0
	.uleb128 .LEHB24-.LFB13682
	.uleb128 .LEHE24-.LEHB24
	.uleb128 .L4555-.LFB13682
	.uleb128 0
	.uleb128 .LEHB25-.LFB13682
	.uleb128 .LEHE25-.LEHB25
	.uleb128 .L4554-.LFB13682
	.uleb128 0
	.uleb128 .LEHB26-.LFB13682
	.uleb128 .LEHE26-.LEHB26
	.uleb128 .L4556-.LFB13682
	.uleb128 0
	.uleb128 .LEHB27-.LFB13682
	.uleb128 .LEHE27-.LEHB27
	.uleb128 .L4554-.LFB13682
	.uleb128 0
	.uleb128 .LEHB28-.LFB13682
	.uleb128 .LEHE28-.LEHB28
	.uleb128 .L4556-.LFB13682
	.uleb128 0
	.uleb128 .LEHB29-.LFB13682
	.uleb128 .LEHE29-.LEHB29
	.uleb128 .L4554-.LFB13682
	.uleb128 0
	.uleb128 .LEHB30-.LFB13682
	.uleb128 .LEHE30-.LEHB30
	.uleb128 0
	.uleb128 0
.LLSDACSE13682:
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.size	_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13679:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13679
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$344, %rsp
	.cfi_def_cfa_offset 400
	movq	%rsi, 16(%rsp)
	vmovsd	%xmm0, (%rsp)
	movq	%fs:40, %rax
	movq	%rax, 328(%rsp)
	xorl	%eax, %eax
	leaq	112(%rsp), %rax
	movb	$0, 112(%rsp)
	movq	%rax, 24(%rsp)
	movq	%rax, 96(%rsp)
	movzbl	1(%rdi), %eax
	movq	$0, 104(%rsp)
	movl	%eax, %ebx
	andl	$6, %ebx
	je	.L4730
	cmpb	$2, %bl
	je	.L5014
	movq	$-1, 8(%rsp)
	cmpb	$4, %bl
	je	.L5015
.L4732:
	movl	%eax, %edx
	leaq	.L4735(%rip), %r8
	shrb	$3, %dl
	andl	$15, %edx
	movslq	(%r8,%rdx,4), %r9
	addq	%r8, %r9
	notrack jmp	*%r9
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4735:
	.long	.L4834-.L4735
	.long	.L4742-.L4735
	.long	.L5007-.L4735
	.long	.L4740-.L4735
	.long	.L5008-.L4735
	.long	.L4738-.L4735
	.long	.L5009-.L4735
	.long	.L4736-.L4735
	.long	.L5010-.L4735
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L4730:
	movl	%eax, %edx
	shrb	$3, %dl
	andl	$15, %edx
	cmpb	$8, %dl
	ja	.L4733
	leaq	.L4830(%rip), %rcx
	movzbl	%dl, %esi
	movslq	(%rcx,%rsi,4), %rdi
	addq	%rcx, %rdi
	notrack jmp	*%rdi
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L4830:
	.long	.L4750-.L4830
	.long	.L4857-.L4830
	.long	.L4858-.L4830
	.long	.L4859-.L4830
	.long	.L4860-.L4830
	.long	.L4861-.L4830
	.long	.L4862-.L4830
	.long	.L4863-.L4830
	.long	.L4864-.L4830
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L4750:
	vmovsd	(%rsp), %xmm0
	leaq	320(%rsp), %r15
	xorl	%r13d, %r13d
	xorl	%r14d, %r14d
	leaq	193(%rsp), %rbx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	_ZSt8to_charsPcS_d@PLT
	movb	$101, 38(%rsp)
	movq	$6, 8(%rsp)
	movq	%rax, %rbp
.L4749:
	cmpl	$75, %edx
	je	.L4751
	movb	$0, 39(%rsp)
.L4748:
	testb	%r14b, %r14b
	je	.L4768
	cmpq	%rbp, %rbx
	je	.L4768
	movq	%rbp, %r14
	movq	%rbx, %r13
	subq	%rbx, %r14
	andl	$7, %r14d
	je	.L4769
	cmpq	$1, %r14
	je	.L4942
	cmpq	$2, %r14
	je	.L4943
	cmpq	$3, %r14
	je	.L4944
	cmpq	$4, %r14
	je	.L4945
	cmpq	$5, %r14
	je	.L4946
	cmpq	$6, %r14
	je	.L4947
	movsbl	(%rbx), %edi
	leaq	1(%rbx), %r13
	call	toupper@PLT
	movb	%al, (%rbx)
.L4947:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4946:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4945:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4944:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4943:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L4942:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
	cmpq	%r13, %rbp
	je	.L4768
.L4769:
	movsbl	0(%r13), %edi
	addq	$8, %r13
	call	toupper@PLT
	movsbl	-7(%r13), %edi
	movb	%al, -8(%r13)
	call	toupper@PLT
	movsbl	-6(%r13), %edi
	movb	%al, -7(%r13)
	call	toupper@PLT
	movsbl	-5(%r13), %edi
	movb	%al, -6(%r13)
	call	toupper@PLT
	movsbl	-4(%r13), %edi
	movb	%al, -5(%r13)
	call	toupper@PLT
	movsbl	-3(%r13), %edi
	movb	%al, -4(%r13)
	call	toupper@PLT
	movsbl	-2(%r13), %edi
	movb	%al, -3(%r13)
	call	toupper@PLT
	movsbl	-1(%r13), %edi
	movb	%al, -2(%r13)
	call	toupper@PLT
	movb	%al, -1(%r13)
	cmpq	%r13, %rbp
	jne	.L4769
	.p2align 4,,10
	.p2align 3
.L4768:
	vmovsd	(%rsp), %xmm1
	vxorpd	%xmm0, %xmm0, %xmm0
	movzbl	(%r12), %ecx
	vcomisd	%xmm0, %xmm1
	jb	.L5012
	movl	%ecx, %edx
	andl	$12, %edx
	cmpb	$4, %dl
	je	.L5016
	xorl	%esi, %esi
	cmpb	$12, %dl
	je	.L5017
.L4767:
	movq	%rbp, %r13
	subq	%rbx, %r13
	testb	$16, %cl
	je	.L4771
	testq	%r13, %r13
	je	.L4846
	movq	%rsi, 40(%rsp)
	movq	%r13, %rdx
	movl	$46, %esi
	movq	%rbx, %rdi
	movb	%cl, (%rsp)
	call	memchr@PLT
	movzbl	(%rsp), %ecx
	movq	40(%rsp), %r9
	testq	%rax, %rax
	movq	%rax, %r14
	je	.L4773
	subq	%rbx, %r14
	cmpq	$-1, %r14
	je	.L4773
	leaq	1(%r14), %r8
	movq	%r13, (%rsp)
	cmpq	%r13, %r8
	jnb	.L4774
	movsbl	38(%rsp), %esi
	movq	%r13, %rdx
	leaq	(%rbx,%r8), %rdi
	movb	%cl, 48(%rsp)
	subq	%r8, %rdx
	movq	%r9, 56(%rsp)
	movq	%r8, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r8
	movzbl	48(%rsp), %ecx
	testq	%rax, %rax
	movq	56(%rsp), %r9
	je	.L4774
	subq	%rbx, %rax
	cmpq	$-1, %rax
	cmove	%r13, %rax
	movq	%rax, (%rsp)
.L4774:
	movq	(%rsp), %rax
	cmpq	%r14, %rax
	sete	%r10b
	sete	40(%rsp)
	cmpb	$0, 39(%rsp)
	movzbl	%r10b, %r14d
	jne	.L5018
	movq	$0, 8(%rsp)
.L4775:
	testq	%r14, %r14
	je	.L4771
.L4781:
	cmpq	$0, 104(%rsp)
	jne	.L4782
	subq	%rbp, %r15
	cmpq	%r14, %r15
	jnb	.L5019
.L4782:
	leaq	96(%rsp), %r15
	leaq	0(%r13,%r14), %rsi
	movq	%r15, %rdi
.LEHB31:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	104(%rsp), %rcx
	movq	(%rsp), %rbp
	testq	%rcx, %rcx
	jne	.L4786
	cmpq	%rbp, %r13
	movq	%rbp, %r8
	movq	%rbx, %rcx
	movq	%r15, %rdi
	cmovbe	%r13, %r8
	xorl	%edx, %edx
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	cmpb	$0, 40(%rsp)
	jne	.L5020
.L4787:
	movq	8(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.L5021
.L4788:
	movq	(%rsp), %rdx
	movq	$-1, %rcx
	movq	%rbx, %rsi
	movq	%r13, %rdi
	call	_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0
.LEHE31:
	movq	104(%rsp), %rsi
	movq	%rax, %r8
	movq	%rdx, %rcx
	movabsq	$9223372036854775807, %r13
	subq	%rsi, %r13
	cmpq	%rax, %r13
	jb	.L5022
	leaq	(%rax,%rsi), %r14
	movq	24(%rsp), %r9
	movq	96(%rsp), %rax
	cmpq	%r9, %rax
	je	.L4853
	movq	112(%rsp), %r10
.L4791:
	cmpq	%r14, %r10
	jb	.L4792
	testq	%r8, %r8
	je	.L4793
	leaq	(%rax,%rsi), %rdi
	cmpq	$1, %r8
	je	.L5023
	movq	%r8, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
	movq	96(%rsp), %rax
.L4793:
	movq	%r14, 104(%rsp)
	movb	$0, (%rax,%r14)
.L4795:
	movq	104(%rsp), %r13
	movq	96(%rsp), %rbx
	movzbl	(%r12), %ecx
	.p2align 4,,10
	.p2align 3
.L4771:
	leaq	144(%rsp), %r14
	andl	$32, %ecx
	movq	$0, 80(%rsp)
	movb	$0, 88(%rsp)
	movq	%r14, 128(%rsp)
	movq	$0, 136(%rsp)
	movb	$0, 144(%rsp)
	jne	.L5024
.L4798:
	movq	%rbx, %r15
.L4809:
	movzwl	(%r12), %edx
	andw	$384, %dx
	cmpw	$128, %dx
	je	.L5025
	cmpw	$256, %dx
	je	.L4812
	movq	16(%rsp), %r12
	movq	16(%r12), %rbp
.L4815:
	testq	%r13, %r13
	jne	.L5026
.L4817:
	movq	128(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.L4821
	movq	144(%rsp), %r14
	leaq	1(%r14), %rsi
	call	_ZdlPvm@PLT
.L4821:
	cmpb	$0, 88(%rsp)
	jne	.L5027
.L4822:
	movq	96(%rsp), %rdi
	movq	24(%rsp), %r15
	cmpq	%r15, %rdi
	je	.L4823
	movq	112(%rsp), %r13
	leaq	1(%r13), %rsi
	call	_ZdlPvm@PLT
.L4823:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4993
	addq	$344, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%rbp, %rax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L5017:
	.cfi_restore_state
	movb	$32, -1(%rbx)
	movzbl	(%r12), %ecx
	subq	$1, %rbx
.L5012:
	movl	$1, %esi
	jmp	.L4767
	.p2align 4,,10
	.p2align 3
.L5025:
	movzwl	2(%r12), %eax
.L4811:
	movq	16(%rsp), %rsi
	movq	16(%rsi), %rbp
	cmpq	%rax, %r13
	jnb	.L4815
	movzbl	(%r12), %r10d
	subq	%r13, %rax
	movsbl	6(%r12), %r9d
	movq	%rax, %r12
	movl	%r10d, %ecx
	andl	$3, %ecx
	jne	.L4819
	andl	$64, %r10d
	je	.L4854
	movzbl	(%rbx), %ebx
	movl	$48, %r9d
	movl	$2, %ecx
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %rax
	cmpb	$15, (%rax,%rbx)
	jbe	.L4819
	movq	24(%rbp), %r11
	movzbl	(%r15), %r9d
	leaq	1(%r11), %rcx
	movq	%rcx, 24(%rbp)
	movb	%r9b, (%r11)
	movq	24(%rbp), %r8
	subq	8(%rbp), %r8
	cmpq	16(%rbp), %r8
	je	.L5028
.L4820:
	addq	$1, %r15
	subq	$1, %r13
	movl	$48, %r9d
	movl	$2, %ecx
	.p2align 4,,10
	.p2align 3
.L4819:
	movq	%r12, %r8
	movq	%r13, %rsi
	movq	%r15, %rdx
	movq	%rbp, %rdi
.LEHB32:
	call	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
.LEHE32:
	movq	%rax, %rbp
	jmp	.L4817
	.p2align 4,,10
	.p2align 3
.L5027:
	leaq	80(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L4822
	.p2align 4,,10
	.p2align 3
.L5024:
	movq	16(%rsp), %rsi
	cmpb	$0, 32(%rsi)
	leaq	24(%rsi), %r15
	je	.L5029
.L4799:
	leaq	72(%rsp), %rbp
	movq	%r15, %rsi
	leaq	160(%rsp), %r15
	movq	%rbp, %rdi
	call	_ZNSt6localeC1ERKS_@PLT
	movsbl	38(%rsp), %ecx
	movq	%rbp, %r8
	movq	%r13, %rsi
	movq	%rbx, %rdx
	movq	%r15, %rdi
.LEHB33:
	call	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0
.LEHE33:
	movq	160(%rsp), %rsi
	leaq	176(%rsp), %r9
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm3
	cmpq	%r9, %rsi
	je	.L5030
	vpinsrq	$1, 176(%rsp), %xmm3, %xmm2
	cmpq	%r14, %rdi
	je	.L5031
	movq	144(%rsp), %r10
	movq	%rsi, 128(%rsp)
	vmovdqu	%xmm2, 136(%rsp)
	testq	%rdi, %rdi
	je	.L4807
	movq	%rdi, 160(%rsp)
	movq	%r10, 176(%rsp)
.L4806:
	movq	$0, 168(%rsp)
	movb	$0, (%rdi)
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	%rbp, %rdi
	call	_ZNSt6localeD1Ev@PLT
	movq	136(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L4798
	movq	128(%rsp), %r15
	movq	%rdi, %r13
	jmp	.L4809
	.p2align 4,,10
	.p2align 3
.L5016:
	movb	$43, -1(%rbx)
	movl	$1, %esi
	movzbl	(%r12), %ecx
	subq	$1, %rbx
	jmp	.L4767
	.p2align 4,,10
	.p2align 3
.L5014:
	movzwl	4(%rdi), %ebp
	movq	%rbp, 8(%rsp)
	jmp	.L4732
	.p2align 4,,10
	.p2align 3
.L4854:
	movl	$32, %r9d
	movl	$2, %ecx
	jmp	.L4819
	.p2align 4,,10
	.p2align 3
.L5018:
	subq	%r9, %rax
	subq	$1, %rax
	cmpb	$48, (%rbx,%r9)
	je	.L5032
.L4779:
	cmpq	$0, 8(%rsp)
	je	.L4775
.L4829:
	subq	%rax, 8(%rsp)
	movq	8(%rsp), %rsi
	addq	%rsi, %r14
	jmp	.L4775
	.p2align 4,,10
	.p2align 3
.L4863:
	movq	$6, 8(%rsp)
.L4736:
	movb	$101, 38(%rsp)
	xorl	%r14d, %r14d
.L4734:
	movb	$1, 39(%rsp)
	movl	$3, %r13d
.L4745:
	movl	8(%rsp), %ecx
	vmovsd	(%rsp), %xmm0
	movl	%r13d, %edx
	leaq	320(%rsp), %r15
	leaq	193(%rsp), %rbx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	_ZSt8to_charsPcS_dSt12chars_formati@PLT
	movq	%rax, %rbp
	cmpl	$75, %edx
	jne	.L4748
	movq	8(%rsp), %rcx
	movl	$1, %ebx
	leaq	8(%rcx), %rbp
	cmpl	$2, %r13d
	je	.L5033
.L4752:
	cmpq	$128, %rbp
	movl	$256, %esi
	leaq	96(%rsp), %r15
	cmova	%rbp, %rsi
	movq	%r15, %rdi
.LEHB34:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	96(%rsp), %rdi
	testb	%bl, %bl
	jne	.L5034
	testl	%r13d, %r13d
	je	.L4760
.L4756:
	movq	24(%rsp), %rdx
	cmpq	%rdx, %rdi
	je	.L4843
	movq	112(%rsp), %rax
	leaq	(%rax,%rax), %rbp
.L4761:
	movq	%rbp, %rsi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %r8
	vmovsd	(%rsp), %xmm0
	movl	%r13d, %edx
	leaq	-1(%r8,%rbp), %rsi
	leaq	1(%r8), %rdi
	movq	%r8, 40(%rsp)
	call	_ZSt8to_charsPcS_dSt12chars_format@PLT
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L4762
	movq	40(%rsp), %r9
	movq	%rax, %rsi
	movq	%r15, %rdi
	subq	%r9, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE34:
.L5011:
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
.L4759:
	leaq	1(%rdi), %rbx
	leaq	(%rdi,%rcx), %r15
	jmp	.L4748
	.p2align 4,,10
	.p2align 3
.L4858:
	movq	$6, 8(%rsp)
.L5007:
	movl	$80, %r10d
	movl	$1, %r14d
.L4741:
	andl	$120, %eax
	movl	$112, %r11d
	cmpb	$16, %al
	cmove	%r10d, %r11d
	movb	%r11b, 38(%rsp)
	testb	%bl, %bl
	jne	.L5035
	vmovsd	(%rsp), %xmm0
	movl	$4, %edx
	leaq	320(%rsp), %r15
	leaq	193(%rsp), %rbx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	movl	$4, %r13d
	call	_ZSt8to_charsPcS_dSt12chars_format@PLT
	movq	%rax, %rbp
	jmp	.L4749
	.p2align 4,,10
	.p2align 3
.L4862:
	movq	$6, 8(%rsp)
.L5009:
	movl	$1, %r14d
.L4737:
	movb	$101, 38(%rsp)
	movl	$2, %r13d
	movb	$0, 39(%rsp)
	jmp	.L4745
	.p2align 4,,10
	.p2align 3
.L4861:
	movq	$6, 8(%rsp)
.L4738:
	xorl	%r14d, %r14d
	jmp	.L4737
	.p2align 4,,10
	.p2align 3
.L4860:
	movq	$6, 8(%rsp)
.L5008:
	movb	$69, 38(%rsp)
	movl	$1, %r14d
.L4739:
	movb	$0, 39(%rsp)
	movl	$1, %r13d
	jmp	.L4745
	.p2align 4,,10
	.p2align 3
.L4859:
	movq	$6, 8(%rsp)
.L4740:
	movb	$101, 38(%rsp)
	xorl	%r14d, %r14d
	jmp	.L4739
	.p2align 4,,10
	.p2align 3
.L4864:
	movq	$6, 8(%rsp)
.L5010:
	movb	$69, 38(%rsp)
	movl	$1, %r14d
	jmp	.L4734
	.p2align 4,,10
	.p2align 3
.L4857:
	movq	$6, 8(%rsp)
.L4742:
	movl	$101, %r10d
	xorl	%r14d, %r14d
	jmp	.L4741
	.p2align 4,,10
	.p2align 3
.L4834:
	movb	$101, 38(%rsp)
	movl	$3, %r13d
	xorl	%r14d, %r14d
	movb	$0, 39(%rsp)
	jmp	.L4745
	.p2align 4,,10
	.p2align 3
.L5031:
	movq	%rsi, 128(%rsp)
	vmovdqu	%xmm2, 136(%rsp)
.L4807:
	movq	%r9, 160(%rsp)
	leaq	176(%rsp), %rdi
	jmp	.L4806
	.p2align 4,,10
	.p2align 3
.L4846:
	movq	$0, (%rsp)
.L4772:
	cmpb	$0, 39(%rsp)
	jne	.L4780
	movq	$0, 8(%rsp)
	movl	$1, %r14d
	movb	$1, 40(%rsp)
	jmp	.L4781
	.p2align 4,,10
	.p2align 3
.L5026:
	movq	%r13, %rsi
	movq	%r15, %rdx
	movq	%rbp, %rdi
.LEHB35:
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L4817
	.p2align 4,,10
	.p2align 3
.L4812:
	movzwl	2(%r12), %edi
	movq	16(%rsp), %rsi
	call	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
.LEHE35:
	jmp	.L4811
	.p2align 4,,10
	.p2align 3
.L4773:
	movsbl	38(%rsp), %esi
	movq	%r13, %rdx
	movq	%rbx, %rdi
	movb	%cl, (%rsp)
	movq	%r9, 40(%rsp)
	call	memchr@PLT
	movzbl	(%rsp), %ecx
	movq	40(%rsp), %rsi
	testq	%rax, %rax
	je	.L4851
	subq	%rbx, %rax
	cmpq	$-1, %rax
	cmove	%r13, %rax
	movq	%rax, (%rsp)
	jmp	.L4772
	.p2align 4,,10
	.p2align 3
.L5029:
	movq	%r15, %rdi
	call	_ZNSt6localeC1Ev@PLT
	movq	16(%rsp), %rbp
	movb	$1, 32(%rbp)
	jmp	.L4799
	.p2align 4,,10
	.p2align 3
.L5015:
	movzwl	4(%rdi), %edi
	leaq	96(%rsp), %r15
.LEHB36:
	call	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
	movq	%rax, 8(%rsp)
	movzbl	1(%r12), %eax
	jmp	.L4732
	.p2align 4,,10
	.p2align 3
.L4786:
	cmpq	%rbp, %rcx
	jb	.L5036
	movq	%rbp, %rsi
	movl	$48, %r8d
	movq	%r14, %rcx
	xorl	%edx, %edx
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	cmpb	$0, 40(%rsp)
	je	.L4795
	movq	96(%rsp), %rbx
	movq	(%rsp), %rcx
	movb	$46, (%rbx,%rcx)
	jmp	.L4795
	.p2align 4,,10
	.p2align 3
.L4780:
	cmpq	$0, 8(%rsp)
	je	.L4856
	movq	(%rsp), %rax
	movzbl	39(%rsp), %r8d
	movl	$1, %r14d
	movb	%r8b, 40(%rsp)
	subq	%rsi, %rax
	jmp	.L4829
	.p2align 4,,10
	.p2align 3
.L5030:
	vmovq	%xmm3, %rax
	testq	%rax, %rax
	je	.L4802
	cmpq	$1, %rax
	je	.L5037
	vmovq	%xmm3, %rdx
	call	memcpy@PLT
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm3
.L4802:
	vmovq	%xmm3, 136(%rsp)
	vmovq	%xmm3, %r8
	movb	$0, (%rdi,%r8)
	movq	160(%rsp), %rdi
	jmp	.L4806
	.p2align 4,,10
	.p2align 3
.L5032:
	cmpq	%r13, %r8
	jnb	.L4994
	movq	%r8, %r11
	notq	%r11
	addq	%r13, %r11
	andl	$7, %r11d
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
	cmpq	%r13, %r8
	jnb	.L4994
	testq	%r11, %r11
	je	.L4777
	cmpq	$1, %r11
	je	.L4950
	cmpq	$2, %r11
	je	.L4951
	cmpq	$3, %r11
	je	.L4952
	cmpq	$4, %r11
	je	.L4953
	cmpq	$5, %r11
	je	.L4954
	cmpq	$6, %r11
	je	.L4955
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
.L4955:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
.L4954:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
.L4953:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
.L4952:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
.L4951:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
.L4950:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	addq	$1, %r8
	cmpq	%r13, %r8
	jnb	.L4994
.L4777:
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	leaq	1(%r8), %rdx
	cmpb	$48, (%rbx,%rdx)
	movq	%rdx, %r8
	jne	.L4778
	addq	$1, %r8
	cmpb	$48, (%rbx,%r8)
	jne	.L4778
	cmpb	$48, 2(%rbx,%rdx)
	leaq	2(%rdx), %r8
	jne	.L4778
	cmpb	$48, 3(%rbx,%rdx)
	leaq	3(%rdx), %r8
	jne	.L4778
	cmpb	$48, 4(%rbx,%rdx)
	leaq	4(%rdx), %r8
	jne	.L4778
	cmpb	$48, 5(%rbx,%rdx)
	leaq	5(%rdx), %r8
	jne	.L4778
	cmpb	$48, 6(%rbx,%rdx)
	leaq	6(%rdx), %r8
	jne	.L4778
	leaq	7(%rdx), %r8
	cmpq	%r13, %r8
	jb	.L4777
.L4994:
	movq	$-1, %r8
.L4778:
	movq	(%rsp), %rax
	subq	%r8, %rax
	jmp	.L4779
	.p2align 4,,10
	.p2align 3
.L5019:
	movq	(%rsp), %r8
	movq	%r13, %rdx
	leaq	(%rbx,%r8), %r15
	leaq	(%r14,%r8), %rdi
	subq	%r8, %rdx
	addq	%rbx, %rdi
	movq	%r15, %rsi
	call	memmove@PLT
	cmpb	$0, 40(%rsp)
	jne	.L5038
.L4784:
	movq	8(%rsp), %rdx
	movl	$48, %esi
	movq	%r15, %rdi
	addq	%r14, %r13
	call	memset@PLT
	movzbl	(%r12), %ecx
	jmp	.L4771
	.p2align 4,,10
	.p2align 3
.L5021:
	movq	104(%rsp), %rsi
	movq	%rdi, %rcx
	movl	$48, %r8d
	xorl	%edx, %edx
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	jmp	.L4788
.L4856:
	movzbl	39(%rsp), %edi
	movl	$1, %r14d
	movb	%dil, 40(%rsp)
	jmp	.L4781
.L4758:
	xorl	%esi, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L4759
.L4760:
	movq	24(%rsp), %r10
	cmpq	%r10, %rdi
	je	.L4842
	movq	112(%rsp), %r11
	leaq	(%r11,%r11), %rbp
.L4757:
	movq	%rbp, %rsi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %r13
	vmovsd	(%rsp), %xmm0
	leaq	-1(%r13,%rbp), %rsi
	leaq	1(%r13), %rdi
	call	_ZSt8to_charsPcS_d@PLT
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L4758
	movq	%rax, %rsi
	movq	%r15, %rdi
	subq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE36:
	jmp	.L5011
.L5033:
	vmovsd	(%rsp), %xmm0
	leaq	80(%rsp), %rdi
	movl	$0, 80(%rsp)
	call	frexp@PLT
	movl	80(%rsp), %edi
	testl	%edi, %edi
	jle	.L4753
	imull	$4004, %edi, %edx
	imulq	$995517945, %rdx, %r8
	movq	%rdx, %rax
	shrq	$32, %r8
	subl	%r8d, %eax
	shrl	%eax
	addl	%r8d, %eax
	shrl	$13, %eax
	addl	$1, %eax
	addq	%rax, %rbp
.L4753:
	movl	$1, %ebx
	jmp	.L4752
.L5038:
	movq	(%rsp), %rdx
	movb	$46, (%r15)
	leaq	1(%rbx,%rdx), %r15
	jmp	.L4784
.L5028:
	movq	0(%rbp), %rdx
	movq	%rbp, %rdi
.LEHB37:
	call	*(%rdx)
.LEHE37:
	jmp	.L4820
.L4751:
	movq	8(%rsp), %r15
	movb	$0, 39(%rsp)
	xorl	%ebx, %ebx
	leaq	8(%r15), %rbp
	jmp	.L4752
.L5034:
	movl	8(%rsp), %r9d
	movl	%r9d, 40(%rsp)
.L4755:
	movq	24(%rsp), %r10
	cmpq	%r10, %rdi
	je	.L4844
	movq	112(%rsp), %r11
	leaq	(%r11,%r11), %rbp
.L4763:
	movq	%rbp, %rsi
	movq	%r15, %rdi
.LEHB38:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rbx
	movl	40(%rsp), %ecx
	movl	%r13d, %edx
	vmovsd	(%rsp), %xmm0
	leaq	-1(%rbx,%rbp), %rsi
	leaq	1(%rbx), %rdi
	movq	%rbx, 48(%rsp)
	call	_ZSt8to_charsPcS_dSt12chars_formati@PLT
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L4764
	movq	48(%rsp), %r13
	movq	%rax, %rsi
	movq	%r15, %rdi
	subq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	jmp	.L5011
.L4792:
	xorl	%edx, %edx
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	movq	96(%rsp), %rax
	jmp	.L4793
.L4851:
	movq	%r13, (%rsp)
	jmp	.L4772
.L5020:
	movl	$46, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
	jmp	.L4787
.L4762:
	xorl	%esi, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L4759
	jmp	.L4756
.L4764:
	xorl	%esi, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L4759
	jmp	.L4755
.L4853:
	movl	$15, %r10d
	jmp	.L4791
.L5023:
	movzbl	(%rcx), %r11d
	movb	%r11b, (%rdi)
	movq	96(%rsp), %rax
	jmp	.L4793
.L5037:
	movzbl	176(%rsp), %ecx
	movb	%cl, (%rdi)
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm3
	jmp	.L4802
.L4843:
	movl	$30, %ebp
	jmp	.L4761
.L4844:
	movl	$30, %ebp
	jmp	.L4763
.L4842:
	movl	$30, %ebp
	jmp	.L4757
.L4824:
	movq	%rbp, %rdi
	vzeroupper
	call	_ZNSt6localeD1Ev@PLT
.L4825:
	leaq	128(%rsp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	cmpb	$0, 88(%rsp)
	jne	.L5039
.L4826:
	leaq	96(%rsp), %r15
.L4827:
	movq	%r15, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	je	.L4828
.L4993:
	call	__stack_chk_fail@PLT
.L5035:
	movb	$0, 39(%rsp)
	movl	$4, %r13d
	jmp	.L4745
.L5022:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4993
	leaq	.LC33(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L5036:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L4993
	movq	(%rsp), %rdx
	leaq	.LC44(%rip), %rsi
	leaq	.LC45(%rip), %rdi
	xorl	%eax, %eax
	call	_ZSt24__throw_out_of_range_fmtPKcz@PLT
.LEHE38:
.L4733:
.L4867:
	endbr64
	movq	%rax, %r12
	jmp	.L4825
.L4866:
	endbr64
	movq	%rax, %r12
	jmp	.L4824
.L4865:
	endbr64
	movq	%rax, %r12
	jmp	.L4827
.L5039:
	leaq	80(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L4826
.L4828:
	movq	%r12, %rdi
.LEHB39:
	call	_Unwind_Resume@PLT
.LEHE39:
	.cfi_endproc
.LFE13679:
	.section	.gcc_except_table
.LLSDA13679:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13679-.LLSDACSB13679
.LLSDACSB13679:
	.uleb128 .LEHB31-.LFB13679
	.uleb128 .LEHE31-.LEHB31
	.uleb128 .L4865-.LFB13679
	.uleb128 0
	.uleb128 .LEHB32-.LFB13679
	.uleb128 .LEHE32-.LEHB32
	.uleb128 .L4867-.LFB13679
	.uleb128 0
	.uleb128 .LEHB33-.LFB13679
	.uleb128 .LEHE33-.LEHB33
	.uleb128 .L4866-.LFB13679
	.uleb128 0
	.uleb128 .LEHB34-.LFB13679
	.uleb128 .LEHE34-.LEHB34
	.uleb128 .L4865-.LFB13679
	.uleb128 0
	.uleb128 .LEHB35-.LFB13679
	.uleb128 .LEHE35-.LEHB35
	.uleb128 .L4867-.LFB13679
	.uleb128 0
	.uleb128 .LEHB36-.LFB13679
	.uleb128 .LEHE36-.LEHB36
	.uleb128 .L4865-.LFB13679
	.uleb128 0
	.uleb128 .LEHB37-.LFB13679
	.uleb128 .LEHE37-.LEHB37
	.uleb128 .L4867-.LFB13679
	.uleb128 0
	.uleb128 .LEHB38-.LFB13679
	.uleb128 .LEHE38-.LEHB38
	.uleb128 .L4865-.LFB13679
	.uleb128 0
	.uleb128 .LEHB39-.LFB13679
	.uleb128 .LEHE39-.LEHB39
	.uleb128 0
	.uleb128 0
.LLSDACSE13679:
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.size	_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.type	_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, @function
_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_:
.LFB13675:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13675
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$344, %rsp
	.cfi_def_cfa_offset 400
	movq	%rsi, 16(%rsp)
	vmovss	%xmm0, (%rsp)
	movq	%fs:40, %rax
	movq	%rax, 328(%rsp)
	xorl	%eax, %eax
	leaq	112(%rsp), %rax
	movb	$0, 112(%rsp)
	movq	%rax, 24(%rsp)
	movq	%rax, 96(%rsp)
	movzbl	1(%rdi), %eax
	movq	$0, 104(%rsp)
	movl	%eax, %ebx
	andl	$6, %ebx
	je	.L5041
	cmpb	$2, %bl
	je	.L5325
	movq	$-1, 8(%rsp)
	cmpb	$4, %bl
	je	.L5326
.L5043:
	movl	%eax, %edx
	leaq	.L5046(%rip), %r8
	shrb	$3, %dl
	andl	$15, %edx
	movslq	(%r8,%rdx,4), %r9
	addq	%r8, %r9
	notrack jmp	*%r9
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L5046:
	.long	.L5145-.L5046
	.long	.L5053-.L5046
	.long	.L5318-.L5046
	.long	.L5051-.L5046
	.long	.L5319-.L5046
	.long	.L5049-.L5046
	.long	.L5320-.L5046
	.long	.L5047-.L5046
	.long	.L5321-.L5046
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L5041:
	movl	%eax, %edx
	shrb	$3, %dl
	andl	$15, %edx
	cmpb	$8, %dl
	ja	.L5044
	leaq	.L5141(%rip), %rcx
	movzbl	%dl, %esi
	movslq	(%rcx,%rsi,4), %rdi
	addq	%rcx, %rdi
	notrack jmp	*%rdi
	.section	.rodata._ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"aG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.align 4
	.align 4
.L5141:
	.long	.L5061-.L5141
	.long	.L5168-.L5141
	.long	.L5169-.L5141
	.long	.L5170-.L5141
	.long	.L5171-.L5141
	.long	.L5172-.L5141
	.long	.L5173-.L5141
	.long	.L5174-.L5141
	.long	.L5175-.L5141
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.p2align 4,,10
	.p2align 3
.L5061:
	vmovss	(%rsp), %xmm0
	leaq	320(%rsp), %r15
	xorl	%r13d, %r13d
	xorl	%r14d, %r14d
	leaq	193(%rsp), %rbx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	_ZSt8to_charsPcS_f@PLT
	movb	$101, 38(%rsp)
	movq	$6, 8(%rsp)
	movq	%rax, %rbp
.L5060:
	cmpl	$75, %edx
	je	.L5062
	movb	$0, 39(%rsp)
.L5059:
	testb	%r14b, %r14b
	je	.L5079
	cmpq	%rbp, %rbx
	je	.L5079
	movq	%rbp, %r14
	movq	%rbx, %r13
	subq	%rbx, %r14
	andl	$7, %r14d
	je	.L5080
	cmpq	$1, %r14
	je	.L5253
	cmpq	$2, %r14
	je	.L5254
	cmpq	$3, %r14
	je	.L5255
	cmpq	$4, %r14
	je	.L5256
	cmpq	$5, %r14
	je	.L5257
	cmpq	$6, %r14
	je	.L5258
	movsbl	(%rbx), %edi
	leaq	1(%rbx), %r13
	call	toupper@PLT
	movb	%al, (%rbx)
.L5258:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L5257:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L5256:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L5255:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L5254:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
.L5253:
	movsbl	0(%r13), %edi
	addq	$1, %r13
	call	toupper@PLT
	movb	%al, -1(%r13)
	cmpq	%r13, %rbp
	je	.L5079
.L5080:
	movsbl	0(%r13), %edi
	addq	$8, %r13
	call	toupper@PLT
	movsbl	-7(%r13), %edi
	movb	%al, -8(%r13)
	call	toupper@PLT
	movsbl	-6(%r13), %edi
	movb	%al, -7(%r13)
	call	toupper@PLT
	movsbl	-5(%r13), %edi
	movb	%al, -6(%r13)
	call	toupper@PLT
	movsbl	-4(%r13), %edi
	movb	%al, -5(%r13)
	call	toupper@PLT
	movsbl	-3(%r13), %edi
	movb	%al, -4(%r13)
	call	toupper@PLT
	movsbl	-2(%r13), %edi
	movb	%al, -3(%r13)
	call	toupper@PLT
	movsbl	-1(%r13), %edi
	movb	%al, -2(%r13)
	call	toupper@PLT
	movb	%al, -1(%r13)
	cmpq	%r13, %rbp
	jne	.L5080
	.p2align 4,,10
	.p2align 3
.L5079:
	vmovss	(%rsp), %xmm1
	vxorps	%xmm0, %xmm0, %xmm0
	movzbl	(%r12), %ecx
	vcomiss	%xmm0, %xmm1
	jb	.L5323
	movl	%ecx, %edx
	andl	$12, %edx
	cmpb	$4, %dl
	je	.L5327
	xorl	%esi, %esi
	cmpb	$12, %dl
	je	.L5328
.L5078:
	movq	%rbp, %r13
	subq	%rbx, %r13
	testb	$16, %cl
	je	.L5082
	testq	%r13, %r13
	je	.L5157
	movq	%rsi, 40(%rsp)
	movq	%r13, %rdx
	movl	$46, %esi
	movq	%rbx, %rdi
	movb	%cl, (%rsp)
	call	memchr@PLT
	movzbl	(%rsp), %ecx
	movq	40(%rsp), %r9
	testq	%rax, %rax
	movq	%rax, %r14
	je	.L5084
	subq	%rbx, %r14
	cmpq	$-1, %r14
	je	.L5084
	leaq	1(%r14), %r8
	movq	%r13, (%rsp)
	cmpq	%r13, %r8
	jnb	.L5085
	movsbl	38(%rsp), %esi
	movq	%r13, %rdx
	leaq	(%rbx,%r8), %rdi
	movb	%cl, 48(%rsp)
	subq	%r8, %rdx
	movq	%r9, 56(%rsp)
	movq	%r8, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r8
	movzbl	48(%rsp), %ecx
	testq	%rax, %rax
	movq	56(%rsp), %r9
	je	.L5085
	subq	%rbx, %rax
	cmpq	$-1, %rax
	cmove	%r13, %rax
	movq	%rax, (%rsp)
.L5085:
	movq	(%rsp), %rax
	cmpq	%r14, %rax
	sete	%r10b
	sete	40(%rsp)
	cmpb	$0, 39(%rsp)
	movzbl	%r10b, %r14d
	jne	.L5329
	movq	$0, 8(%rsp)
.L5086:
	testq	%r14, %r14
	je	.L5082
.L5092:
	cmpq	$0, 104(%rsp)
	jne	.L5093
	subq	%rbp, %r15
	cmpq	%r14, %r15
	jnb	.L5330
.L5093:
	leaq	96(%rsp), %r15
	leaq	0(%r13,%r14), %rsi
	movq	%r15, %rdi
.LEHB40:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	104(%rsp), %rcx
	movq	(%rsp), %rbp
	testq	%rcx, %rcx
	jne	.L5097
	cmpq	%rbp, %r13
	movq	%rbp, %r8
	movq	%rbx, %rcx
	movq	%r15, %rdi
	cmovbe	%r13, %r8
	xorl	%edx, %edx
	xorl	%esi, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	cmpb	$0, 40(%rsp)
	jne	.L5331
.L5098:
	movq	8(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.L5332
.L5099:
	movq	(%rsp), %rdx
	movq	$-1, %rcx
	movq	%rbx, %rsi
	movq	%r13, %rdi
	call	_ZNKSt17basic_string_viewIcSt11char_traitsIcEE6substrEmm.isra.0
.LEHE40:
	movq	104(%rsp), %rsi
	movq	%rax, %r8
	movq	%rdx, %rcx
	movabsq	$9223372036854775807, %r13
	subq	%rsi, %r13
	cmpq	%rax, %r13
	jb	.L5333
	leaq	(%rax,%rsi), %r14
	movq	24(%rsp), %r9
	movq	96(%rsp), %rax
	cmpq	%r9, %rax
	je	.L5164
	movq	112(%rsp), %r10
.L5102:
	cmpq	%r14, %r10
	jb	.L5103
	testq	%r8, %r8
	je	.L5104
	leaq	(%rax,%rsi), %rdi
	cmpq	$1, %r8
	je	.L5334
	movq	%r8, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
	movq	96(%rsp), %rax
.L5104:
	movq	%r14, 104(%rsp)
	movb	$0, (%rax,%r14)
.L5106:
	movq	104(%rsp), %r13
	movq	96(%rsp), %rbx
	movzbl	(%r12), %ecx
	.p2align 4,,10
	.p2align 3
.L5082:
	leaq	144(%rsp), %r14
	andl	$32, %ecx
	movq	$0, 80(%rsp)
	movb	$0, 88(%rsp)
	movq	%r14, 128(%rsp)
	movq	$0, 136(%rsp)
	movb	$0, 144(%rsp)
	jne	.L5335
.L5109:
	movq	%rbx, %r15
.L5120:
	movzwl	(%r12), %edx
	andw	$384, %dx
	cmpw	$128, %dx
	je	.L5336
	cmpw	$256, %dx
	je	.L5123
	movq	16(%rsp), %r12
	movq	16(%r12), %rbp
.L5126:
	testq	%r13, %r13
	jne	.L5337
.L5128:
	movq	128(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.L5132
	movq	144(%rsp), %r14
	leaq	1(%r14), %rsi
	call	_ZdlPvm@PLT
.L5132:
	cmpb	$0, 88(%rsp)
	jne	.L5338
.L5133:
	movq	96(%rsp), %rdi
	movq	24(%rsp), %r15
	cmpq	%r15, %rdi
	je	.L5134
	movq	112(%rsp), %r13
	leaq	1(%r13), %rsi
	call	_ZdlPvm@PLT
.L5134:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L5304
	addq	$344, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%rbp, %rax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L5328:
	.cfi_restore_state
	movb	$32, -1(%rbx)
	movzbl	(%r12), %ecx
	subq	$1, %rbx
.L5323:
	movl	$1, %esi
	jmp	.L5078
	.p2align 4,,10
	.p2align 3
.L5336:
	movzwl	2(%r12), %eax
.L5122:
	movq	16(%rsp), %rsi
	movq	16(%rsi), %rbp
	cmpq	%rax, %r13
	jnb	.L5126
	movzbl	(%r12), %r10d
	subq	%r13, %rax
	movsbl	6(%r12), %r9d
	movq	%rax, %r12
	movl	%r10d, %ecx
	andl	$3, %ecx
	jne	.L5130
	andl	$64, %r10d
	je	.L5165
	movzbl	(%rbx), %ebx
	movl	$48, %r9d
	movl	$2, %ecx
	leaq	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE(%rip), %rax
	cmpb	$15, (%rax,%rbx)
	jbe	.L5130
	movq	24(%rbp), %r11
	movzbl	(%r15), %r9d
	leaq	1(%r11), %rcx
	movq	%rcx, 24(%rbp)
	movb	%r9b, (%r11)
	movq	24(%rbp), %r8
	subq	8(%rbp), %r8
	cmpq	16(%rbp), %r8
	je	.L5339
.L5131:
	addq	$1, %r15
	subq	$1, %r13
	movl	$48, %r9d
	movl	$2, %ecx
	.p2align 4,,10
	.p2align 3
.L5130:
	movq	%r12, %r8
	movq	%r13, %rsi
	movq	%r15, %rdx
	movq	%rbp, %rdi
.LEHB41:
	call	_ZNSt8__format14__write_paddedINS_10_Sink_iterIcEEcEET_S3_St17basic_string_viewIT0_St11char_traitsIS5_EENS_6_AlignEmS5_
.LEHE41:
	movq	%rax, %rbp
	jmp	.L5128
	.p2align 4,,10
	.p2align 3
.L5338:
	leaq	80(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L5133
	.p2align 4,,10
	.p2align 3
.L5335:
	movq	16(%rsp), %rsi
	cmpb	$0, 32(%rsi)
	leaq	24(%rsi), %r15
	je	.L5340
.L5110:
	leaq	72(%rsp), %rbp
	movq	%r15, %rsi
	leaq	160(%rsp), %r15
	movq	%rbp, %rdi
	call	_ZNSt6localeC1ERKS_@PLT
	movsbl	38(%rsp), %ecx
	movq	%rbp, %r8
	movq	%r13, %rsi
	movq	%rbx, %rdx
	movq	%r15, %rdi
.LEHB42:
	call	_ZNKSt8__format14__formatter_fpIcE11_M_localizeB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEEcRKSt6locale.isra.0
.LEHE42:
	movq	160(%rsp), %rsi
	leaq	176(%rsp), %r9
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm3
	cmpq	%r9, %rsi
	je	.L5341
	vpinsrq	$1, 176(%rsp), %xmm3, %xmm2
	cmpq	%r14, %rdi
	je	.L5342
	movq	144(%rsp), %r10
	movq	%rsi, 128(%rsp)
	vmovdqu	%xmm2, 136(%rsp)
	testq	%rdi, %rdi
	je	.L5118
	movq	%rdi, 160(%rsp)
	movq	%r10, 176(%rsp)
.L5117:
	movq	$0, 168(%rsp)
	movb	$0, (%rdi)
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	%rbp, %rdi
	call	_ZNSt6localeD1Ev@PLT
	movq	136(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L5109
	movq	128(%rsp), %r15
	movq	%rdi, %r13
	jmp	.L5120
	.p2align 4,,10
	.p2align 3
.L5327:
	movb	$43, -1(%rbx)
	movl	$1, %esi
	movzbl	(%r12), %ecx
	subq	$1, %rbx
	jmp	.L5078
	.p2align 4,,10
	.p2align 3
.L5325:
	movzwl	4(%rdi), %ebp
	movq	%rbp, 8(%rsp)
	jmp	.L5043
	.p2align 4,,10
	.p2align 3
.L5165:
	movl	$32, %r9d
	movl	$2, %ecx
	jmp	.L5130
	.p2align 4,,10
	.p2align 3
.L5329:
	subq	%r9, %rax
	subq	$1, %rax
	cmpb	$48, (%rbx,%r9)
	je	.L5343
.L5090:
	cmpq	$0, 8(%rsp)
	je	.L5086
.L5140:
	subq	%rax, 8(%rsp)
	movq	8(%rsp), %rsi
	addq	%rsi, %r14
	jmp	.L5086
	.p2align 4,,10
	.p2align 3
.L5174:
	movq	$6, 8(%rsp)
.L5047:
	movb	$101, 38(%rsp)
	xorl	%r14d, %r14d
.L5045:
	movb	$1, 39(%rsp)
	movl	$3, %r13d
.L5056:
	movl	8(%rsp), %ecx
	vmovss	(%rsp), %xmm0
	movl	%r13d, %edx
	leaq	320(%rsp), %r15
	leaq	193(%rsp), %rbx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	_ZSt8to_charsPcS_fSt12chars_formati@PLT
	movq	%rax, %rbp
	cmpl	$75, %edx
	jne	.L5059
	movq	8(%rsp), %rcx
	movl	$1, %ebx
	leaq	8(%rcx), %rbp
	cmpl	$2, %r13d
	je	.L5344
.L5063:
	cmpq	$128, %rbp
	movl	$256, %esi
	leaq	96(%rsp), %r15
	cmova	%rbp, %rsi
	movq	%r15, %rdi
.LEHB43:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	movq	96(%rsp), %rdi
	testb	%bl, %bl
	jne	.L5345
	testl	%r13d, %r13d
	je	.L5071
.L5067:
	movq	24(%rsp), %rdx
	cmpq	%rdx, %rdi
	je	.L5154
	movq	112(%rsp), %rax
	leaq	(%rax,%rax), %rbp
.L5072:
	movq	%rbp, %rsi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %r8
	vmovss	(%rsp), %xmm0
	movl	%r13d, %edx
	leaq	-1(%r8,%rbp), %rsi
	leaq	1(%r8), %rdi
	movq	%r8, 40(%rsp)
	call	_ZSt8to_charsPcS_fSt12chars_format@PLT
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L5073
	movq	40(%rsp), %r9
	movq	%rax, %rsi
	movq	%r15, %rdi
	subq	%r9, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE43:
.L5322:
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
.L5070:
	leaq	1(%rdi), %rbx
	leaq	(%rdi,%rcx), %r15
	jmp	.L5059
	.p2align 4,,10
	.p2align 3
.L5169:
	movq	$6, 8(%rsp)
.L5318:
	movl	$80, %r10d
	movl	$1, %r14d
.L5052:
	andl	$120, %eax
	movl	$112, %r11d
	cmpb	$16, %al
	cmove	%r10d, %r11d
	movb	%r11b, 38(%rsp)
	testb	%bl, %bl
	jne	.L5346
	vmovss	(%rsp), %xmm0
	movl	$4, %edx
	leaq	320(%rsp), %r15
	leaq	193(%rsp), %rbx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	movl	$4, %r13d
	call	_ZSt8to_charsPcS_fSt12chars_format@PLT
	movq	%rax, %rbp
	jmp	.L5060
	.p2align 4,,10
	.p2align 3
.L5173:
	movq	$6, 8(%rsp)
.L5320:
	movl	$1, %r14d
.L5048:
	movb	$101, 38(%rsp)
	movl	$2, %r13d
	movb	$0, 39(%rsp)
	jmp	.L5056
	.p2align 4,,10
	.p2align 3
.L5172:
	movq	$6, 8(%rsp)
.L5049:
	xorl	%r14d, %r14d
	jmp	.L5048
	.p2align 4,,10
	.p2align 3
.L5171:
	movq	$6, 8(%rsp)
.L5319:
	movb	$69, 38(%rsp)
	movl	$1, %r14d
.L5050:
	movb	$0, 39(%rsp)
	movl	$1, %r13d
	jmp	.L5056
	.p2align 4,,10
	.p2align 3
.L5170:
	movq	$6, 8(%rsp)
.L5051:
	movb	$101, 38(%rsp)
	xorl	%r14d, %r14d
	jmp	.L5050
	.p2align 4,,10
	.p2align 3
.L5175:
	movq	$6, 8(%rsp)
.L5321:
	movb	$69, 38(%rsp)
	movl	$1, %r14d
	jmp	.L5045
	.p2align 4,,10
	.p2align 3
.L5168:
	movq	$6, 8(%rsp)
.L5053:
	movl	$101, %r10d
	xorl	%r14d, %r14d
	jmp	.L5052
	.p2align 4,,10
	.p2align 3
.L5145:
	movb	$101, 38(%rsp)
	movl	$3, %r13d
	xorl	%r14d, %r14d
	movb	$0, 39(%rsp)
	jmp	.L5056
	.p2align 4,,10
	.p2align 3
.L5342:
	movq	%rsi, 128(%rsp)
	vmovdqu	%xmm2, 136(%rsp)
.L5118:
	movq	%r9, 160(%rsp)
	leaq	176(%rsp), %rdi
	jmp	.L5117
	.p2align 4,,10
	.p2align 3
.L5157:
	movq	$0, (%rsp)
.L5083:
	cmpb	$0, 39(%rsp)
	jne	.L5091
	movq	$0, 8(%rsp)
	movl	$1, %r14d
	movb	$1, 40(%rsp)
	jmp	.L5092
	.p2align 4,,10
	.p2align 3
.L5337:
	movq	%r13, %rsi
	movq	%r15, %rdx
	movq	%rbp, %rdi
.LEHB44:
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	jmp	.L5128
	.p2align 4,,10
	.p2align 3
.L5123:
	movzwl	2(%r12), %edi
	movq	16(%rsp), %rsi
	call	_ZNKSt8__format5_SpecIcE12_M_get_widthISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
.LEHE44:
	jmp	.L5122
	.p2align 4,,10
	.p2align 3
.L5084:
	movsbl	38(%rsp), %esi
	movq	%r13, %rdx
	movq	%rbx, %rdi
	movb	%cl, (%rsp)
	movq	%r9, 40(%rsp)
	call	memchr@PLT
	movzbl	(%rsp), %ecx
	movq	40(%rsp), %rsi
	testq	%rax, %rax
	je	.L5162
	subq	%rbx, %rax
	cmpq	$-1, %rax
	cmove	%r13, %rax
	movq	%rax, (%rsp)
	jmp	.L5083
	.p2align 4,,10
	.p2align 3
.L5340:
	movq	%r15, %rdi
	call	_ZNSt6localeC1Ev@PLT
	movq	16(%rsp), %rbp
	movb	$1, 32(%rbp)
	jmp	.L5110
	.p2align 4,,10
	.p2align 3
.L5326:
	movzwl	4(%rdi), %edi
	leaq	96(%rsp), %r15
.LEHB45:
	call	_ZNKSt8__format5_SpecIcE16_M_get_precisionISt20basic_format_contextINS_10_Sink_iterIcEEcEEEmRT_.part.0.isra.0
	movq	%rax, 8(%rsp)
	movzbl	1(%r12), %eax
	jmp	.L5043
	.p2align 4,,10
	.p2align 3
.L5097:
	cmpq	%rbp, %rcx
	jb	.L5347
	movq	%rbp, %rsi
	movl	$48, %r8d
	movq	%r14, %rcx
	xorl	%edx, %edx
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	cmpb	$0, 40(%rsp)
	je	.L5106
	movq	96(%rsp), %rbx
	movq	(%rsp), %rcx
	movb	$46, (%rbx,%rcx)
	jmp	.L5106
	.p2align 4,,10
	.p2align 3
.L5091:
	cmpq	$0, 8(%rsp)
	je	.L5167
	movq	(%rsp), %rax
	movzbl	39(%rsp), %r8d
	movl	$1, %r14d
	movb	%r8b, 40(%rsp)
	subq	%rsi, %rax
	jmp	.L5140
	.p2align 4,,10
	.p2align 3
.L5341:
	vmovq	%xmm3, %rax
	testq	%rax, %rax
	je	.L5113
	cmpq	$1, %rax
	je	.L5348
	vmovq	%xmm3, %rdx
	call	memcpy@PLT
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm3
.L5113:
	vmovq	%xmm3, 136(%rsp)
	vmovq	%xmm3, %r8
	movb	$0, (%rdi,%r8)
	movq	160(%rsp), %rdi
	jmp	.L5117
	.p2align 4,,10
	.p2align 3
.L5343:
	cmpq	%r13, %r8
	jnb	.L5305
	movq	%r8, %r11
	notq	%r11
	addq	%r13, %r11
	andl	$7, %r11d
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
	cmpq	%r13, %r8
	jnb	.L5305
	testq	%r11, %r11
	je	.L5088
	cmpq	$1, %r11
	je	.L5261
	cmpq	$2, %r11
	je	.L5262
	cmpq	$3, %r11
	je	.L5263
	cmpq	$4, %r11
	je	.L5264
	cmpq	$5, %r11
	je	.L5265
	cmpq	$6, %r11
	je	.L5266
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
.L5266:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
.L5265:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
.L5264:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
.L5263:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
.L5262:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
.L5261:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	addq	$1, %r8
	cmpq	%r13, %r8
	jnb	.L5305
.L5088:
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	leaq	1(%r8), %rdx
	cmpb	$48, (%rbx,%rdx)
	movq	%rdx, %r8
	jne	.L5089
	addq	$1, %r8
	cmpb	$48, (%rbx,%r8)
	jne	.L5089
	cmpb	$48, 2(%rbx,%rdx)
	leaq	2(%rdx), %r8
	jne	.L5089
	cmpb	$48, 3(%rbx,%rdx)
	leaq	3(%rdx), %r8
	jne	.L5089
	cmpb	$48, 4(%rbx,%rdx)
	leaq	4(%rdx), %r8
	jne	.L5089
	cmpb	$48, 5(%rbx,%rdx)
	leaq	5(%rdx), %r8
	jne	.L5089
	cmpb	$48, 6(%rbx,%rdx)
	leaq	6(%rdx), %r8
	jne	.L5089
	leaq	7(%rdx), %r8
	cmpq	%r13, %r8
	jb	.L5088
.L5305:
	movq	$-1, %r8
.L5089:
	movq	(%rsp), %rax
	subq	%r8, %rax
	jmp	.L5090
	.p2align 4,,10
	.p2align 3
.L5330:
	movq	(%rsp), %r8
	movq	%r13, %rdx
	leaq	(%rbx,%r8), %r15
	leaq	(%r14,%r8), %rdi
	subq	%r8, %rdx
	addq	%rbx, %rdi
	movq	%r15, %rsi
	call	memmove@PLT
	cmpb	$0, 40(%rsp)
	jne	.L5349
.L5095:
	movq	8(%rsp), %rdx
	movl	$48, %esi
	movq	%r15, %rdi
	addq	%r14, %r13
	call	memset@PLT
	movzbl	(%r12), %ecx
	jmp	.L5082
	.p2align 4,,10
	.p2align 3
.L5332:
	movq	104(%rsp), %rsi
	movq	%rdi, %rcx
	movl	$48, %r8d
	xorl	%edx, %edx
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc.isra.0
	jmp	.L5099
.L5167:
	movzbl	39(%rsp), %edi
	movl	$1, %r14d
	movb	%dil, 40(%rsp)
	jmp	.L5092
.L5069:
	xorl	%esi, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L5070
.L5071:
	movq	24(%rsp), %r10
	cmpq	%r10, %rdi
	je	.L5153
	movq	112(%rsp), %r11
	leaq	(%r11,%r11), %rbp
.L5068:
	movq	%rbp, %rsi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %r13
	vmovss	(%rsp), %xmm0
	leaq	-1(%r13,%rbp), %rsi
	leaq	1(%r13), %rdi
	call	_ZSt8to_charsPcS_f@PLT
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L5069
	movq	%rax, %rsi
	movq	%r15, %rdi
	subq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
.LEHE45:
	jmp	.L5322
.L5344:
	vmovss	(%rsp), %xmm0
	leaq	80(%rsp), %rdi
	movl	$0, 80(%rsp)
	call	frexpf@PLT
	movl	80(%rsp), %edi
	testl	%edi, %edi
	jle	.L5064
	imull	$4004, %edi, %edx
	imulq	$995517945, %rdx, %r8
	movq	%rdx, %rax
	shrq	$32, %r8
	subl	%r8d, %eax
	shrl	%eax
	addl	%r8d, %eax
	shrl	$13, %eax
	addl	$1, %eax
	addq	%rax, %rbp
.L5064:
	movl	$1, %ebx
	jmp	.L5063
.L5349:
	movq	(%rsp), %rdx
	movb	$46, (%r15)
	leaq	1(%rbx,%rdx), %r15
	jmp	.L5095
.L5339:
	movq	0(%rbp), %rdx
	movq	%rbp, %rdi
.LEHB46:
	call	*(%rdx)
.LEHE46:
	jmp	.L5131
.L5062:
	movq	8(%rsp), %r15
	movb	$0, 39(%rsp)
	xorl	%ebx, %ebx
	leaq	8(%r15), %rbp
	jmp	.L5063
.L5345:
	movl	8(%rsp), %r9d
	movl	%r9d, 40(%rsp)
.L5066:
	movq	24(%rsp), %r10
	cmpq	%r10, %rdi
	je	.L5155
	movq	112(%rsp), %r11
	leaq	(%r11,%r11), %rbp
.L5074:
	movq	%rbp, %rsi
	movq	%r15, %rdi
.LEHB47:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rbx
	movl	40(%rsp), %ecx
	movl	%r13d, %edx
	vmovss	(%rsp), %xmm0
	leaq	-1(%rbx,%rbp), %rsi
	leaq	1(%rbx), %rdi
	movq	%rbx, 48(%rsp)
	call	_ZSt8to_charsPcS_fSt12chars_formati@PLT
	movq	%rax, %rbp
	movl	%edx, %ebx
	testl	%edx, %edx
	jne	.L5075
	movq	48(%rsp), %r13
	movq	%rax, %rsi
	movq	%r15, %rdi
	subq	%r13, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	jmp	.L5322
.L5103:
	xorl	%edx, %edx
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	movq	96(%rsp), %rax
	jmp	.L5104
.L5162:
	movq	%r13, (%rsp)
	jmp	.L5083
.L5331:
	movl	$46, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
	jmp	.L5098
.L5073:
	xorl	%esi, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L5070
	jmp	.L5067
.L5075:
	xorl	%esi, %esi
	movq	%r15, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc.constprop.0
	movq	96(%rsp), %rdi
	movq	104(%rsp), %rcx
	cmpl	$75, %ebx
	jne	.L5070
	jmp	.L5066
.L5164:
	movl	$15, %r10d
	jmp	.L5102
.L5334:
	movzbl	(%rcx), %r11d
	movb	%r11b, (%rdi)
	movq	96(%rsp), %rax
	jmp	.L5104
.L5348:
	movzbl	176(%rsp), %ecx
	movb	%cl, (%rdi)
	movq	128(%rsp), %rdi
	vmovq	168(%rsp), %xmm3
	jmp	.L5113
.L5154:
	movl	$30, %ebp
	jmp	.L5072
.L5155:
	movl	$30, %ebp
	jmp	.L5074
.L5153:
	movl	$30, %ebp
	jmp	.L5068
.L5135:
	movq	%rbp, %rdi
	vzeroupper
	call	_ZNSt6localeD1Ev@PLT
.L5136:
	leaq	128(%rsp), %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	cmpb	$0, 88(%rsp)
	jne	.L5350
.L5137:
	leaq	96(%rsp), %r15
.L5138:
	movq	%r15, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	je	.L5139
.L5304:
	call	__stack_chk_fail@PLT
.L5346:
	movb	$0, 39(%rsp)
	movl	$4, %r13d
	jmp	.L5056
.L5333:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L5304
	leaq	.LC33(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L5347:
	movq	328(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L5304
	movq	(%rsp), %rdx
	leaq	.LC44(%rip), %rsi
	leaq	.LC45(%rip), %rdi
	xorl	%eax, %eax
	call	_ZSt24__throw_out_of_range_fmtPKcz@PLT
.LEHE47:
.L5044:
.L5178:
	endbr64
	movq	%rax, %r12
	jmp	.L5136
.L5177:
	endbr64
	movq	%rax, %r12
	jmp	.L5135
.L5176:
	endbr64
	movq	%rax, %r12
	jmp	.L5138
.L5350:
	leaq	80(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L5137
.L5139:
	movq	%r12, %rdi
.LEHB48:
	call	_Unwind_Resume@PLT
.LEHE48:
	.cfi_endproc
.LFE13675:
	.section	.gcc_except_table
.LLSDA13675:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13675-.LLSDACSB13675
.LLSDACSB13675:
	.uleb128 .LEHB40-.LFB13675
	.uleb128 .LEHE40-.LEHB40
	.uleb128 .L5176-.LFB13675
	.uleb128 0
	.uleb128 .LEHB41-.LFB13675
	.uleb128 .LEHE41-.LEHB41
	.uleb128 .L5178-.LFB13675
	.uleb128 0
	.uleb128 .LEHB42-.LFB13675
	.uleb128 .LEHE42-.LEHB42
	.uleb128 .L5177-.LFB13675
	.uleb128 0
	.uleb128 .LEHB43-.LFB13675
	.uleb128 .LEHE43-.LEHB43
	.uleb128 .L5176-.LFB13675
	.uleb128 0
	.uleb128 .LEHB44-.LFB13675
	.uleb128 .LEHE44-.LEHB44
	.uleb128 .L5178-.LFB13675
	.uleb128 0
	.uleb128 .LEHB45-.LFB13675
	.uleb128 .LEHE45-.LEHB45
	.uleb128 .L5176-.LFB13675
	.uleb128 0
	.uleb128 .LEHB46-.LFB13675
	.uleb128 .LEHE46-.LEHB46
	.uleb128 .L5178-.LFB13675
	.uleb128 0
	.uleb128 .LEHB47-.LFB13675
	.uleb128 .LEHE47-.LEHB47
	.uleb128 .L5176-.LFB13675
	.uleb128 0
	.uleb128 .LEHB48-.LFB13675
	.uleb128 .LEHE48-.LEHB48
	.uleb128 0
	.uleb128 0
.LLSDACSE13675:
	.section	.text._ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,"axG",@progbits,_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_,comdat
	.size	_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_, .-_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	.section	.rodata._ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE.str1.1,"aMS",@progbits,1
.LC49:
	.string	"true"
.LC50:
	.string	"false"
	.section	.rodata._ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE.str1.8,"aMS",@progbits,1
	.align 8
.LC51:
	.string	"format error: format-spec contains invalid formatting options for 'bool'"
	.align 8
.LC52:
	.string	"format error: format-spec contains invalid formatting options for 'charT'"
	.align 8
.LC53:
	.string	"00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899"
	.section	.text._ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,"axG",@progbits,_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE
	.type	_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE, @function
_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE:
.LFB13443:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13443
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	.L5354(%rip), %rcx
	movzbl	%dl, %edx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%rdi, %r12
	pushq	%rbx
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$392, %rsp
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movslq	(%rcx,%rdx,4), %rax
	addq	%rcx, %rax
	notrack jmp	*%rax
	.section	.rodata._ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,"aG",@progbits,_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,comdat
	.align 4
	.align 4
.L5354:
	.long	.L5370-.L5354
	.long	.L5369-.L5354
	.long	.L5368-.L5354
	.long	.L5367-.L5354
	.long	.L5366-.L5354
	.long	.L5365-.L5354
	.long	.L5364-.L5354
	.long	.L5363-.L5354
	.long	.L5362-.L5354
	.long	.L5361-.L5354
	.long	.L5360-.L5354
	.long	.L5359-.L5354
	.long	.L5358-.L5354
	.long	.L5357-.L5354
	.long	.L5356-.L5354
	.long	.L5355-.L5354
	.long	.L5353-.L5354
	.long	.L5353-.L5354
	.long	.L5353-.L5354
	.long	.L5353-.L5354
	.long	.L5353-.L5354
	.section	.text._ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,"axG",@progbits,_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,comdat
	.p2align 4,,10
	.p2align 3
.L5355:
	movq	(%rbx), %r14
	leaq	-392(%rbp), %r13
	movl	$1, %edx
	movabsq	$9007199254740992, %rsi
	movq	%rsi, -392(%rbp)
	movq	%r13, %rdi
	leaq	8(%r14), %rsi
.LEHB49:
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	(%rbx), %rdi
	movq	(%r12), %rsi
	movq	%rax, 8(%r14)
	movq	8(%r12), %rdx
	movq	48(%rdi), %rbx
	movq	%r13, %rdi
	movq	%rbx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
.L5777:
	movq	%rax, 16(%rbx)
.L5351:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L5353:
	.cfi_restore_state
	movq	(%rsi), %r15
	leaq	-392(%rbp), %r14
	movabsq	$9007199254740992, %rdi
	movq	%rdi, -392(%rbp)
	movq	%r14, %rdi
	leaq	8(%r15), %rsi
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	movq	(%rbx), %rsi
	vmovdqa	(%r12), %xmm0
	movq	%r14, %rdi
	movq	%rax, 8(%r15)
	movq	48(%rsi), %rbx
	movq	%rbx, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5369:
	movq	(%rsi), %r13
	leaq	-392(%rbp), %r14
	xorl	%edx, %edx
	movabsq	$9007199254740992, %r8
	movq	%r14, %rdi
	movq	%r8, -392(%rbp)
	leaq	8(%r13), %rsi
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
.LEHE49:
	movzbl	-391(%rbp), %edi
	andl	$120, %edi
	jne	.L5372
	movzbl	-392(%rbp), %r9d
	testb	$92, %r9b
	jne	.L5781
	movq	%rax, 8(%r13)
	movq	(%rbx), %rax
	andl	$32, %r9d
	movzbl	(%r12), %ebx
	leaq	-368(%rbp), %r12
	movq	$0, -376(%rbp)
	movq	%r12, -384(%rbp)
	movq	48(%rax), %r13
	movb	$0, -368(%rbp)
	jne	.L5782
	movl	%ebx, %edx
	leaq	.LC50(%rip), %rcx
	leaq	.LC49(%rip), %r10
	negb	%dl
	leaq	-384(%rbp), %r12
	sbbq	%r15, %r15
	testb	%bl, %bl
	movq	%r12, %rdi
	cmovne	%r10, %rcx
	leaq	5(%r15), %r8
	xorl	%edx, %edx
	xorl	%esi, %esi
.LEHB50:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	movq	-376(%rbp), %rdi
.L5385:
	movq	-384(%rbp), %rsi
	movq	%r14, %r8
	movq	%r13, %rcx
	movq	%rdi, %rdx
	movl	$1, %r9d
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE
.LEHE50:
	movq	%rax, %r14
	movq	%r12, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	%r14, %rax
	jmp	.L5378
	.p2align 4,,10
	.p2align 3
.L5368:
	movq	(%rsi), %r15
	movl	$7, %edx
	movabsq	$9007199254740992, %r13
	movq	%r13, -392(%rbp)
	leaq	-392(%rbp), %r13
	leaq	8(%r15), %rsi
	movq	%r13, %rdi
.LEHB51:
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movzbl	-391(%rbp), %edx
	movl	%edx, %ecx
	notl	%edx
	andl	$120, %ecx
	cmpb	$56, %cl
	sete	%r10b
	andl	$56, %edx
	jne	.L5391
	testb	$92, -392(%rbp)
	jne	.L5783
	movq	(%rbx), %r11
	movq	%rax, 8(%r15)
	movq	48(%r11), %rbx
	testb	%cl, %cl
	je	.L5521
	testb	%r10b, %r10b
	jne	.L5521
	movq	16(%rbx), %rax
	jmp	.L5777
	.p2align 4,,10
	.p2align 3
.L5367:
	movq	(%rsi), %r14
	leaq	-400(%rbp), %r13
	movl	$1, %edx
	movabsq	$9007199254740992, %rcx
	movq	%r13, %rdi
	movq	%rcx, -400(%rbp)
	leaq	8(%r14), %rsi
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	(%rbx), %rbx
	movq	%rax, 8(%r14)
	movzbl	-399(%rbp), %eax
	movq	48(%rbx), %rbx
	movl	(%r12), %r14d
	movl	%eax, %ecx
	andl	$120, %ecx
	cmpb	$56, %cl
	je	.L5784
	shrb	$3, %al
	movl	%r14d, %edx
	andl	$15, %eax
	testl	%r14d, %r14d
	js	.L5785
	cmpb	$4, %al
	je	.L5408
	ja	.L5409
	cmpb	$1, %al
	jbe	.L5410
	cmpb	$16, %cl
	leaq	.LC35(%rip), %r8
	leaq	.LC36(%rip), %r9
	cmovne	%r9, %r8
	testl	%r14d, %r14d
	jne	.L5489
	movl	$48, %r11d
	leaq	-316(%rbp), %rsi
	leaq	-317(%rbp), %r15
.L5415:
	movzbl	-400(%rbp), %r12d
	movb	%r11b, -317(%rbp)
	testb	$16, %r12b
	je	.L5507
.L5506:
	movq	$-2, %rdx
	movl	$2, %r10d
.L5419:
	addq	%r15, %rdx
	movl	%r10d, %r11d
	testl	%r10d, %r10d
	je	.L5420
	xorl	%edi, %edi
	leal	-1(%r10), %r10d
	movzbl	(%r8,%rdi), %r9d
	andl	$7, %r10d
	movb	%r9b, (%rdx,%rdi)
	movl	$1, %edi
	cmpl	%r11d, %edi
	jnb	.L5420
	testl	%r10d, %r10d
	je	.L5438
	cmpl	$1, %r10d
	je	.L5696
	cmpl	$2, %r10d
	je	.L5697
	cmpl	$3, %r10d
	je	.L5698
	cmpl	$4, %r10d
	je	.L5699
	cmpl	$5, %r10d
	je	.L5700
	cmpl	$6, %r10d
	je	.L5701
	movl	$1, %eax
	movl	$2, %edi
	movzbl	(%r8,%rax), %ecx
	movb	%cl, (%rdx,%rax)
.L5701:
	movl	%edi, %r10d
	addl	$1, %edi
	movzbl	(%r8,%r10), %r9d
	movb	%r9b, (%rdx,%r10)
.L5700:
	movl	%edi, %eax
	addl	$1, %edi
	movzbl	(%r8,%rax), %ecx
	movb	%cl, (%rdx,%rax)
.L5699:
	movl	%edi, %r10d
	addl	$1, %edi
	movzbl	(%r8,%r10), %r9d
	movb	%r9b, (%rdx,%r10)
.L5698:
	movl	%edi, %eax
	addl	$1, %edi
	movzbl	(%r8,%rax), %ecx
	movb	%cl, (%rdx,%rax)
.L5697:
	movl	%edi, %r10d
	addl	$1, %edi
	movzbl	(%r8,%r10), %r9d
	movb	%r9b, (%rdx,%r10)
.L5696:
	movl	%edi, %eax
	addl	$1, %edi
	movzbl	(%r8,%rax), %ecx
	movb	%cl, (%rdx,%rax)
	cmpl	%r11d, %edi
	jnb	.L5420
.L5438:
	movl	%edi, %r10d
	leal	1(%rdi), %eax
	movzbl	(%r8,%r10), %r9d
	movzbl	(%r8,%rax), %ecx
	movb	%r9b, (%rdx,%r10)
	leal	2(%rdi), %r10d
	movb	%cl, (%rdx,%rax)
	leal	3(%rdi), %eax
	movzbl	(%r8,%r10), %r9d
	movzbl	(%r8,%rax), %ecx
	movb	%r9b, (%rdx,%r10)
	leal	4(%rdi), %r10d
	movb	%cl, (%rdx,%rax)
	leal	5(%rdi), %eax
	movzbl	(%r8,%r10), %r9d
	movzbl	(%r8,%rax), %ecx
	movb	%r9b, (%rdx,%r10)
	leal	6(%rdi), %r10d
	movb	%cl, (%rdx,%rax)
	leal	7(%rdi), %eax
	movzbl	(%r8,%r10), %r9d
	addl	$8, %edi
	movzbl	(%r8,%rax), %ecx
	movb	%r9b, (%rdx,%r10)
	movb	%cl, (%rdx,%rax)
	cmpl	%r11d, %edi
	jb	.L5438
	.p2align 4,,10
	.p2align 3
.L5420:
	shrb	$2, %r12b
	leaq	-1(%rdx), %rax
	andl	$3, %r12d
	testl	%r14d, %r14d
	jns	.L5421
	movb	$45, -1(%rdx)
.L5440:
	movq	%rax, %rdx
.L5442:
	movq	%r15, %rcx
	subq	%rdx, %rsi
	movq	%rbx, %r8
	movq	%r13, %rdi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L5777
	.p2align 4,,10
	.p2align 3
.L5366:
	movq	(%rsi), %r15
	leaq	-392(%rbp), %r13
	movabsq	$9007199254740992, %rdx
	movq	%rdx, -392(%rbp)
	movq	%r13, %rdi
	movl	$1, %edx
	leaq	8(%r15), %rsi
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movl	(%r12), %edx
	movq	(%rbx), %r10
	movzbl	-391(%rbp), %r12d
	movq	%rax, 8(%r15)
	movq	48(%r10), %rbx
	movl	%r12d, %r15d
	andl	$120, %r15d
	cmpb	$56, %r15b
	je	.L5786
	shrb	$3, %r12b
	andl	$15, %r12d
	cmpb	$4, %r12b
	je	.L5446
	ja	.L5447
	cmpb	$1, %r12b
	jbe	.L5448
	cmpb	$16, %r15b
	leaq	.LC35(%rip), %r11
	leaq	.LC36(%rip), %r9
	cmovne	%r9, %r11
	testl	%edx, %edx
	jne	.L5787
	movl	$48, %r10d
	leaq	-268(%rbp), %r14
	leaq	-269(%rbp), %rcx
.L5453:
	movzbl	-392(%rbp), %r12d
	movb	%r10b, -269(%rbp)
	testb	$16, %r12b
	je	.L5776
.L5514:
	movq	$-2, %rdx
	movl	$2, %eax
.L5457:
	addq	%rcx, %rdx
	movl	%eax, %edi
	testl	%eax, %eax
	je	.L5458
	xorl	%r15d, %r15d
	leal	-1(%rax), %esi
	movl	$1, %eax
	movzbl	(%r11,%r15), %r9d
	andl	$7, %esi
	movb	%r9b, (%rdx,%r15)
	cmpl	%edi, %eax
	jnb	.L5458
	testl	%esi, %esi
	je	.L5467
	cmpl	$1, %esi
	je	.L5702
	cmpl	$2, %esi
	je	.L5703
	cmpl	$3, %esi
	je	.L5704
	cmpl	$4, %esi
	je	.L5705
	cmpl	$5, %esi
	je	.L5706
	cmpl	$6, %esi
	je	.L5707
	movl	$1, %r8d
	movl	$2, %eax
	movzbl	(%r11,%r8), %r10d
	movb	%r10b, (%rdx,%r8)
.L5707:
	movl	%eax, %esi
	addl	$1, %eax
	movzbl	(%r11,%rsi), %r15d
	movb	%r15b, (%rdx,%rsi)
.L5706:
	movl	%eax, %r8d
	addl	$1, %eax
	movzbl	(%r11,%r8), %r9d
	movb	%r9b, (%rdx,%r8)
.L5705:
	movl	%eax, %r10d
	addl	$1, %eax
	movzbl	(%r11,%r10), %esi
	movb	%sil, (%rdx,%r10)
.L5704:
	movl	%eax, %r15d
	addl	$1, %eax
	movzbl	(%r11,%r15), %r8d
	movb	%r8b, (%rdx,%r15)
.L5703:
	movl	%eax, %r10d
	addl	$1, %eax
	movzbl	(%r11,%r10), %r9d
	movb	%r9b, (%rdx,%r10)
.L5702:
	movl	%eax, %esi
	addl	$1, %eax
	movzbl	(%r11,%rsi), %r15d
	movb	%r15b, (%rdx,%rsi)
	cmpl	%edi, %eax
	jnb	.L5458
.L5467:
	movl	%eax, %r8d
	leal	1(%rax), %esi
	leal	2(%rax), %r15d
	movzbl	(%r11,%r8), %r10d
	movzbl	(%r11,%rsi), %r9d
	movb	%r10b, (%rdx,%r8)
	leal	3(%rax), %r10d
	movzbl	(%r11,%r15), %r8d
	movb	%r9b, (%rdx,%rsi)
	movzbl	(%r11,%r10), %esi
	movb	%r8b, (%rdx,%r15)
	leal	4(%rax), %r15d
	leal	5(%rax), %r8d
	movb	%sil, (%rdx,%r10)
	movzbl	(%r11,%r15), %r9d
	leal	6(%rax), %esi
	movzbl	(%r11,%r8), %r10d
	movb	%r9b, (%rdx,%r15)
	movzbl	(%r11,%rsi), %r15d
	movb	%r10b, (%rdx,%r8)
	leal	7(%rax), %r8d
	addl	$8, %eax
	movzbl	(%r11,%r8), %r9d
	movb	%r15b, (%rdx,%rsi)
	movb	%r9b, (%rdx,%r8)
	cmpl	%edi, %eax
	jb	.L5467
	.p2align 4,,10
	.p2align 3
.L5458:
	shrb	$2, %r12b
	andl	$3, %r12d
	cmpl	$1, %r12d
	je	.L5516
	cmpl	$3, %r12d
	je	.L5788
.L5470:
	movq	%r14, %rsi
	subq	%rdx, %rcx
	movq	%rbx, %r8
	movq	%r13, %rdi
	subq	%rdx, %rsi
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L5777
	.p2align 4,,10
	.p2align 3
.L5365:
	movq	(%rsi), %r15
	leaq	-392(%rbp), %r14
	movl	$1, %edx
	movabsq	$9007199254740992, %r11
	movq	%r14, %rdi
	movq	%r11, -392(%rbp)
	leaq	8(%r15), %rsi
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	(%rbx), %rsi
	movq	%r14, %rdi
	movq	%rax, 8(%r15)
	movq	48(%rsi), %rbx
	movq	(%r12), %rsi
	movq	%rbx, %rdx
	call	_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5364:
	movq	(%rsi), %r15
	leaq	-392(%rbp), %r14
	movl	$1, %edx
	movabsq	$9007199254740992, %r8
	movq	%r14, %rdi
	movq	%r8, -392(%rbp)
	leaq	8(%r15), %rsi
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	(%rbx), %r9
	movq	(%r12), %rsi
	movq	%r14, %rdi
	movq	%rax, 8(%r15)
	movq	48(%r9), %r13
	movq	%r13, %rdx
	call	_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%r13)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5363:
	movq	(%rsi), %r14
	leaq	-392(%rbp), %r13
	movabsq	$9007199254740992, %rcx
	movq	%r13, %rdi
	movq	%rcx, -392(%rbp)
	leaq	8(%r14), %rsi
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	movq	(%rbx), %rdi
	vmovss	(%r12), %xmm0
	movq	%rax, 8(%r14)
	movq	48(%rdi), %rbx
	movq	%r13, %rdi
	movq	%rbx, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5362:
	movq	(%rsi), %r14
	leaq	-392(%rbp), %r13
	movabsq	$9007199254740992, %r10
	movq	%r13, %rdi
	movq	%r10, -392(%rbp)
	leaq	8(%r14), %rsi
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	vmovsd	(%r12), %xmm0
	movq	%r13, %rdi
	movq	%rax, 8(%r14)
	movq	(%rbx), %rax
	movq	48(%rax), %r15
	movq	%r15, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%r15)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5361:
	movq	(%rbx), %r14
	leaq	-392(%rbp), %r15
	movabsq	$9007199254740992, %rsi
	movq	%rsi, -392(%rbp)
	movq	%r15, %rdi
	leaq	8(%r14), %rsi
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	movq	(%rbx), %rdx
	movq	%r15, %rdi
	movq	%rax, 8(%r14)
	movq	48(%rdx), %rbx
	pushq	8(%r12)
	pushq	(%r12)
	movq	%rbx, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	popq	%rdx
	popq	%rcx
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5360:
	movq	(%rsi), %r14
	leaq	-392(%rbp), %r15
	movabsq	$9007199254740992, %r9
	movq	%r15, %rdi
	movq	%r9, -392(%rbp)
	leaq	8(%r14), %rsi
	call	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE
	movq	(%rbx), %r11
	movq	%rax, 8(%r14)
	movq	(%r12), %r12
	movq	48(%r11), %r13
	movq	%r12, %rdi
	call	strlen@PLT
	movq	%r13, %rcx
	movq	%r12, %rdx
	movq	%r15, %rdi
	movq	%rax, %rsi
	call	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE
	movq	%rax, 16(%r13)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5359:
	movq	(%rsi), %r15
	leaq	-392(%rbp), %r13
	movabsq	$9007199254740992, %rdi
	movq	%rdi, -392(%rbp)
	movq	%r13, %rdi
	leaq	8(%r15), %rsi
	call	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE
	movq	(%rbx), %r8
	movq	(%r12), %rsi
	movq	%r13, %rdi
	movq	%rax, 8(%r15)
	movq	8(%r12), %rdx
	movq	48(%r8), %rbx
	movq	%rbx, %rcx
	call	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE
	movq	%rax, 16(%rbx)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5358:
	movabsq	$9007199254740992, %r11
	movq	(%rsi), %r15
	movq	$0, -400(%rbp)
	movq	%r11, -392(%rbp)
	movq	16(%r15), %r13
	movq	8(%r15), %rsi
	cmpq	%rsi, %r13
	je	.L5472
	cmpb	$125, (%rsi)
	je	.L5472
	leaq	-392(%rbp), %r14
	movq	%r13, %rdx
	movq	%r14, %rdi
	call	_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_
	movq	%rax, %rsi
	cmpq	%rax, %r13
	je	.L5472
	cmpb	$125, (%rax)
	je	.L5472
	leaq	8(%r15), %rcx
	movq	%r13, %rdx
	movq	%r14, %rdi
	call	_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE
	movq	%rax, %rsi
	cmpq	%rax, %r13
	je	.L5476
	movzbl	(%rax), %ecx
	cmpb	$112, %cl
	je	.L5789
.L5477:
	cmpb	$125, %cl
	jne	.L5790
.L5476:
	movl	-392(%rbp), %eax
	movl	-389(%rbp), %edi
	movq	(%rbx), %r8
	movl	%eax, -400(%rbp)
	movl	%edi, -397(%rbp)
.L5474:
	movq	%rsi, 8(%r15)
	movq	(%r12), %r12
	movq	48(%r8), %r15
	testq	%r12, %r12
	jne	.L5480
	movb	$48, -270(%rbp)
	movl	$3, %edi
.L5481:
	movl	$30768, %ecx
	movl	$2, %r9d
	movq	%rdi, %rdx
	movw	%cx, -272(%rbp)
	leaq	-272(%rbp), %rsi
	movq	%r15, %rcx
	leaq	-400(%rbp), %r8
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE
	movq	%rax, 16(%r15)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5357:
	movq	(%rsi), %rdi
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	movq	(%r12), %rdx
	movq	48(%rdi), %rsi
	addq	$8, %rdi
	movq	8(%r12), %r10
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	jmp	*%r10
	.p2align 4,,10
	.p2align 3
.L5356:
	.cfi_restore_state
	movq	(%rsi), %r15
	leaq	-392(%rbp), %r14
	movl	$1, %edx
	movabsq	$9007199254740992, %r8
	movq	%r14, %rdi
	movq	%r8, -392(%rbp)
	leaq	8(%r15), %rsi
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	(%rbx), %r9
	movq	(%r12), %rsi
	movq	%r14, %rdi
	movq	%rax, 8(%r15)
	movq	8(%r12), %rdx
	movq	48(%r9), %rbx
	movq	%rbx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5391:
	movq	(%rbx), %rbx
	movq	%rax, 8(%r15)
	movzbl	(%r12), %esi
	movq	48(%rbx), %rbx
	testb	%cl, %cl
	je	.L5487
	testb	%r10b, %r10b
	jne	.L5487
	movq	%rbx, %rdx
	movq	%r13, %rdi
	call	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	jmp	.L5777
	.p2align 4,,10
	.p2align 3
.L5372:
	movq	(%rbx), %rsi
	movq	%rax, 8(%r13)
	movq	48(%rsi), %r13
	movzbl	(%r12), %esi
	cmpb	$56, %dil
	jne	.L5377
	movb	%sil, -400(%rbp)
	movq	%r14, %rcx
	movq	%r13, %rdx
	movl	$1, %edi
	leaq	-400(%rbp), %rsi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
.L5378:
	movq	%rax, 16(%r13)
	jmp	.L5351
	.p2align 4,,10
	.p2align 3
.L5784:
	leal	128(%r14), %edx
	cmpl	$255, %edx
	ja	.L5398
	leaq	-392(%rbp), %rsi
	movq	%r13, %rcx
	movq	%rbx, %rdx
	movl	$1, %edi
	movb	%r14b, -392(%rbp)
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
	jmp	.L5777
	.p2align 4,,10
	.p2align 3
.L5786:
	cmpl	$127, %edx
	ja	.L5398
	movb	%dl, -400(%rbp)
.L5780:
	leaq	-400(%rbp), %rsi
	movq	%r13, %rcx
	movq	%rbx, %rdx
	movl	$1, %edi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
	jmp	.L5777
	.p2align 4,,10
	.p2align 3
.L5472:
	movl	-392(%rbp), %r9d
	movl	-389(%rbp), %ebx
	movq	%r15, %r8
	movl	%r9d, -400(%rbp)
	movl	%ebx, -397(%rbp)
	jmp	.L5474
	.p2align 4,,10
	.p2align 3
.L5480:
	bsrq	%r12, %rsi
	vmovdqa	.LC25(%rip), %xmm0
	leal	4(%rsi), %edi
	shrl	$2, %edi
	vmovdqa	%xmm0, -320(%rbp)
	leal	-1(%rdi), %r10d
	cmpq	$255, %r12
	jbe	.L5482
.L5483:
	movq	%r12, %rdx
	movq	%r12, %r13
	movl	%r10d, %r11d
	shrq	$8, %r12
	shrq	$4, %rdx
	leal	-1(%r10), %eax
	leal	-2(%r10), %r8d
	andl	$15, %r13d
	movzbl	-320(%rbp,%r13), %r14d
	andl	$15, %edx
	movzbl	-320(%rbp,%rdx), %ecx
	movb	%r14b, -270(%rbp,%r11)
	movb	%cl, -270(%rbp,%rax)
	cmpq	$255, %r12
	jbe	.L5482
	movq	%r12, %r11
	leal	-3(%r10), %esi
	leal	-4(%r10), %r14d
	movq	%r12, %r9
	shrq	$4, %r11
	andl	$15, %r9d
	shrq	$8, %r12
	movzbl	-320(%rbp,%r9), %ebx
	andl	$15, %r11d
	movzbl	-320(%rbp,%r11), %r13d
	movb	%bl, -270(%rbp,%r8)
	movb	%r13b, -270(%rbp,%rsi)
	cmpq	$255, %r12
	jbe	.L5482
	movq	%r12, %rcx
	leal	-5(%r10), %r8d
	leal	-6(%r10), %ebx
	movq	%r12, %rdx
	shrq	$4, %rcx
	andl	$15, %edx
	shrq	$8, %r12
	andl	$15, %ecx
	movzbl	-320(%rbp,%rdx), %eax
	movzbl	-320(%rbp,%rcx), %r9d
	movb	%al, -270(%rbp,%r14)
	movb	%r9b, -270(%rbp,%r8)
	cmpq	$255, %r12
	jbe	.L5482
	movq	%r12, %r13
	movq	%r12, %r11
	leal	-7(%r10), %r14d
	shrq	$8, %r12
	shrq	$4, %r13
	andl	$15, %r11d
	subl	$8, %r10d
	movzbl	-320(%rbp,%r11), %esi
	andl	$15, %r13d
	movzbl	-320(%rbp,%r13), %edx
	movb	%sil, -270(%rbp,%rbx)
	movb	%dl, -270(%rbp,%r14)
	cmpq	$255, %r12
	ja	.L5483
	.p2align 4,,10
	.p2align 3
.L5482:
	cmpq	$15, %r12
	jbe	.L5484
	movq	%r12, %r10
	shrq	$4, %r12
	andl	$15, %r10d
	movzbl	-320(%rbp,%r10), %eax
	movb	%al, -269(%rbp)
	movzbl	-320(%rbp,%r12), %r12d
.L5485:
	movb	%r12b, -270(%rbp)
	addl	$2, %edi
	jmp	.L5481
.L5410:
	testl	%r14d, %r14d
	jne	.L5406
	movzbl	-400(%rbp), %r12d
	movb	$48, -317(%rbp)
	leaq	-317(%rbp), %r15
	leaq	-316(%rbp), %rsi
	leaq	-318(%rbp), %rax
	movq	%r15, %rdx
	shrb	$2, %r12b
	andl	$3, %r12d
.L5421:
	movzbl	%r12b, %r14d
	cmpl	$1, %r14d
	je	.L5791
	cmpl	$3, %r14d
	jne	.L5442
	movb	$32, -1(%rdx)
	jmp	.L5440
	.p2align 4,,10
	.p2align 3
.L5785:
	negl	%edx
	cmpb	$3, %al
	jbe	.L5402
	cmpb	$4, %al
	je	.L5403
	cmpb	$40, %cl
	leaq	.LC39(%rip), %rsi
	leaq	.LC38(%rip), %r8
	cmovne	%r8, %rsi
.L5407:
	leaq	-317(%rbp), %r15
	movq	%rsi, -416(%rbp)
	leaq	-285(%rbp), %rsi
	movq	%r15, %rdi
	movb	%cl, -408(%rbp)
	call	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movzbl	-408(%rbp), %edi
	movzbl	-400(%rbp), %r12d
	movq	-416(%rbp), %r8
	movq	%rax, %rsi
	cmpb	$48, %dil
	je	.L5792
.L5435:
	testb	$16, %r12b
	jne	.L5506
	.p2align 4,,10
	.p2align 3
.L5507:
	movq	%r15, %rdx
	jmp	.L5420
.L5521:
	movzbl	(%r12), %esi
.L5487:
	movb	%sil, -400(%rbp)
	jmp	.L5780
	.p2align 4,,10
	.p2align 3
.L5377:
	movq	%r13, %rdx
	movq	%r14, %rdi
	call	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
.LEHE51:
	jmp	.L5378
	.p2align 4,,10
	.p2align 3
.L5484:
	movzbl	-320(%rbp,%r12), %r12d
	jmp	.L5485
	.p2align 4,,10
	.p2align 3
.L5516:
	movl	$43, %edi
.L5469:
	movb	%dil, -1(%rdx)
	subq	$1, %rdx
	jmp	.L5470
	.p2align 4,,10
	.p2align 3
.L5446:
	testl	%edx, %edx
	je	.L5461
	leaq	-269(%rbp), %rax
	leaq	-237(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -408(%rbp)
	call	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movzbl	-392(%rbp), %r12d
	movq	-408(%rbp), %rcx
	leaq	.LC37(%rip), %r11
	movq	%rax, %r14
	movl	$1, %eax
.L5462:
	movq	%rcx, %rdx
	testb	$16, %r12b
	je	.L5458
	movq	%rax, %rdx
	negq	%rdx
	jmp	.L5457
	.p2align 4,,10
	.p2align 3
.L5448:
	testl	%edx, %edx
	je	.L5461
	leaq	-269(%rbp), %r11
	leaq	-237(%rbp), %rsi
	movq	%r11, %rdi
	movq	%r11, -408(%rbp)
	call	_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movq	-408(%rbp), %rcx
	movq	%rax, %r14
.L5775:
	movzbl	-392(%rbp), %r12d
	movq	%rcx, %rdx
	jmp	.L5458
	.p2align 4,,10
	.p2align 3
.L5447:
	cmpb	$40, %r15b
	je	.L5793
	testl	%edx, %edx
	jne	.L5511
	movb	$48, -269(%rbp)
	movzbl	-392(%rbp), %r12d
	leaq	.LC38(%rip), %r11
	leaq	-268(%rbp), %r14
	leaq	-269(%rbp), %rcx
	cmpb	$48, %r15b
	je	.L5465
.L5464:
	testb	$16, %r12b
	jne	.L5514
.L5776:
	movq	%rcx, %rdx
	jmp	.L5458
	.p2align 4,,10
	.p2align 3
.L5461:
	movb	$48, -269(%rbp)
	leaq	-268(%rbp), %r14
	leaq	-269(%rbp), %rcx
	jmp	.L5775
.L5788:
	movl	$32, %edi
	jmp	.L5469
.L5402:
	cmpb	$1, %al
	jbe	.L5406
	cmpb	$16, %cl
	leaq	.LC35(%rip), %r8
	leaq	.LC36(%rip), %r15
	cmovne	%r15, %r8
.L5489:
	bsrl	%edx, %r12d
	movl	$32, %r10d
	movl	$31, %eax
	xorl	$31, %r12d
	subl	%r12d, %r10d
	subl	%r12d, %eax
	je	.L5418
	movl	%eax, %esi
	movl	$30, %r9d
	leaq	-320(%rbp,%rsi), %r11
	subl	%r12d, %r9d
	leaq	-321(%rbp,%rsi), %rsi
	subq	%r9, %rsi
	movq	%r11, %rdi
	subq	%rsi, %rdi
	andl	$7, %edi
	je	.L5417
	cmpq	$1, %rdi
	je	.L5669
	cmpq	$2, %rdi
	je	.L5670
	cmpq	$3, %rdi
	je	.L5671
	cmpq	$4, %rdi
	je	.L5672
	cmpq	$5, %rdi
	je	.L5673
	cmpq	$6, %rdi
	je	.L5674
	movl	%edx, %ecx
	subq	$1, %r11
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%r11)
.L5674:
	movl	%edx, %eax
	subq	$1, %r11
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r11)
.L5673:
	movl	%edx, %ecx
	subq	$1, %r11
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%r11)
.L5672:
	movl	%edx, %eax
	subq	$1, %r11
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r11)
.L5671:
	movl	%edx, %ecx
	subq	$1, %r11
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%r11)
.L5670:
	movl	%edx, %eax
	subq	$1, %r11
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r11)
.L5669:
	movl	%edx, %ecx
	subq	$1, %r11
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%r11)
	cmpq	%rsi, %r11
	je	.L5418
.L5417:
	movl	%edx, %eax
	movl	%edx, %ecx
	movl	%edx, %r15d
	movl	%edx, %r12d
	andl	$1, %eax
	shrl	$5, %ecx
	movl	%edx, %r9d
	movl	%edx, %edi
	addl	$48, %eax
	andl	$1, %ecx
	shrl	%r15d
	subq	$8, %r11
	addl	$48, %ecx
	movb	%al, 11(%r11)
	movl	%edx, %eax
	shrl	$2, %r12d
	movb	%cl, 6(%r11)
	shrl	$3, %r9d
	movl	%edx, %ecx
	shrl	$4, %edi
	shrl	$6, %eax
	andl	$1, %r15d
	andl	$1, %r12d
	andl	$1, %r9d
	andl	$1, %edi
	andl	$1, %eax
	shrb	$7, %cl
	addl	$48, %r15d
	addl	$48, %r12d
	addl	$48, %r9d
	addl	$48, %edi
	addl	$48, %eax
	addl	$48, %ecx
	movb	%r15b, 10(%r11)
	shrl	$8, %edx
	movb	%r12b, 9(%r11)
	movb	%r9b, 8(%r11)
	movb	%dil, 7(%r11)
	movb	%al, 5(%r11)
	movb	%cl, 4(%r11)
	cmpq	%rsi, %r11
	jne	.L5417
.L5418:
	leaq	-317(%rbp), %r15
	movslq	%r10d, %rsi
	movl	$49, %r11d
	addq	%r15, %rsi
	jmp	.L5415
.L5791:
	movb	$43, -1(%rdx)
	jmp	.L5440
.L5789:
	leaq	1(%rax), %rdx
	cmpq	%rdx, %r13
	je	.L5517
	movzbl	1(%rax), %ecx
	movq	%rdx, %rsi
	jmp	.L5477
.L5408:
	testl	%r14d, %r14d
	jne	.L5403
	movb	$48, -317(%rbp)
	xorl	%edx, %edx
	xorl	%r8d, %r8d
	xorl	%r10d, %r10d
	movzbl	-400(%rbp), %r12d
	leaq	-316(%rbp), %rsi
	leaq	-317(%rbp), %r15
.L5434:
	testb	$16, %r12b
	je	.L5507
	testb	%dl, %dl
	je	.L5507
	movq	%r10, %rdx
	negq	%rdx
	jmp	.L5419
.L5409:
	cmpb	$40, %cl
	je	.L5794
	testl	%r14d, %r14d
	jne	.L5503
	movb	$48, -317(%rbp)
	movzbl	-400(%rbp), %r12d
	cmpb	$48, %cl
	je	.L5504
	leaq	-316(%rbp), %rsi
	leaq	.LC38(%rip), %r8
	leaq	-317(%rbp), %r15
	jmp	.L5435
.L5406:
	cmpl	$9, %edx
	jbe	.L5496
	cmpl	$99, %edx
	jbe	.L5795
	cmpl	$999, %edx
	jbe	.L5497
	cmpl	$9999, %edx
	jbe	.L5796
	movl	%edx, %r8d
	movl	$5, %r12d
	cmpl	$99999, %edx
	jbe	.L5425
	cmpl	$999999, %edx
	jbe	.L5797
	cmpl	$9999999, %edx
	jbe	.L5499
	cmpl	$99999999, %edx
	jbe	.L5500
	cmpq	$999999999, %r8
	jbe	.L5501
	movl	$5, %r12d
.L5429:
	addl	$5, %r12d
.L5425:
	vmovdqa	.LC26(%rip), %ymm1
	vmovdqa	.LC27(%rip), %ymm2
	leal	-1(%r12), %r15d
	vmovdqa	.LC28(%rip), %ymm3
	vmovdqa	.LC29(%rip), %ymm4
	vmovdqa	.LC30(%rip), %ymm5
	vmovdqa	.LC31(%rip), %ymm6
	vmovdqu	%ymm1, -272(%rbp)
	vmovdqa	.LC32(%rip), %xmm7
	vmovdqu	%ymm2, -240(%rbp)
	vmovdqu	%ymm6, -112(%rbp)
	vmovdqu	%ymm3, -208(%rbp)
	vmovdqu	%ymm4, -176(%rbp)
	vmovdqu	%ymm5, -144(%rbp)
	vmovdqu	%xmm7, -87(%rbp)
.L5431:
	imulq	$1374389535, %r8, %r8
	movl	%edx, %r11d
	movl	%edx, %r10d
	movl	%r15d, %edi
	shrq	$37, %r8
	imull	$100, %r8d, %esi
	movl	%r8d, %edx
	subl	%esi, %r11d
	leal	-1(%r15), %esi
	addl	%r11d, %r11d
	movzbl	-272(%rbp,%r11), %ecx
	leal	1(%r11), %r9d
	leal	-2(%r15), %r11d
	movzbl	-272(%rbp,%r9), %eax
	movb	%al, -317(%rbp,%rdi)
	movb	%cl, -317(%rbp,%rsi)
	cmpl	$9999, %r10d
	jbe	.L5762
	movl	%r8d, %r10d
	movl	%r8d, %eax
	imulq	$1374389535, %r10, %rdi
	movl	%r8d, %r10d
	shrq	$37, %rdi
	imull	$100, %edi, %r9d
	movl	%edi, %edx
	subl	%r9d, %eax
	addl	%eax, %eax
	leal	1(%rax), %esi
	movzbl	-272(%rbp,%rax), %edi
	movzbl	-272(%rbp,%rsi), %ecx
	movb	%cl, -317(%rbp,%r11)
	leal	-3(%r15), %r11d
	subl	$4, %r15d
	movb	%dil, -317(%rbp,%r11)
	cmpl	$9999, %r8d
	jbe	.L5762
	movl	%edx, %r8d
	jmp	.L5431
.L5403:
	leaq	-317(%rbp), %r15
	leaq	-285(%rbp), %rsi
	movq	%r15, %rdi
	call	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movl	$1, %edx
	movzbl	-400(%rbp), %r12d
	leaq	.LC37(%rip), %r8
	movq	%rax, %rsi
	movl	$1, %r10d
	jmp	.L5434
.L5787:
	bsrl	%edx, %r14d
	movl	$32, %r10d
	movl	$31, %esi
	xorl	$31, %r14d
	subl	%r14d, %r10d
	subl	%r14d, %esi
	je	.L5456
	movl	%esi, %r12d
	movl	$30, %ecx
	leaq	-272(%rbp,%r12), %rax
	leaq	-273(%rbp,%r12), %rsi
	subl	%r14d, %ecx
	subq	%rcx, %rsi
	movq	%rax, %r8
	subq	%rsi, %r8
	andl	$7, %r8d
	je	.L5455
	cmpq	$1, %r8
	je	.L5682
	cmpq	$2, %r8
	je	.L5683
	cmpq	$3, %r8
	je	.L5684
	cmpq	$4, %r8
	je	.L5685
	cmpq	$5, %r8
	je	.L5686
	cmpq	$6, %r8
	je	.L5687
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
.L5687:
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
.L5686:
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
.L5685:
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
.L5684:
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
.L5683:
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
.L5682:
	movl	%edx, %ecx
	subq	$1, %rax
	shrl	%edx
	andl	$1, %ecx
	addl	$48, %ecx
	movb	%cl, 4(%rax)
	cmpq	%rax, %rsi
	je	.L5456
.L5455:
	movl	%edx, %ecx
	movl	%edx, %edi
	movl	%edx, %r15d
	movl	%edx, %r9d
	andl	$1, %ecx
	movl	%edx, %r14d
	movl	%edx, %r12d
	movl	%edx, %r8d
	shrl	%edi
	addl	$48, %ecx
	shrl	$2, %r15d
	subq	$8, %rax
	movb	%cl, 11(%rax)
	shrl	$3, %r9d
	movl	%edx, %ecx
	shrl	$4, %r14d
	shrl	$5, %r12d
	shrl	$6, %r8d
	andl	$1, %edi
	andl	$1, %r15d
	andl	$1, %r9d
	andl	$1, %r14d
	andl	$1, %r12d
	andl	$1, %r8d
	shrb	$7, %cl
	addl	$48, %edi
	addl	$48, %r15d
	addl	$48, %r9d
	addl	$48, %r14d
	addl	$48, %r12d
	addl	$48, %r8d
	addl	$48, %ecx
	movb	%dil, 10(%rax)
	shrl	$8, %edx
	movb	%r15b, 9(%rax)
	movb	%r9b, 8(%rax)
	movb	%r14b, 7(%rax)
	movb	%r12b, 6(%rax)
	movb	%r8b, 5(%rax)
	movb	%cl, 4(%rax)
	cmpq	%rax, %rsi
	jne	.L5455
.L5456:
	movslq	%r10d, %rdx
	leaq	-269(%rbp), %rcx
	movl	$49, %r10d
	leaq	(%rcx,%rdx), %r14
	jmp	.L5453
.L5762:
	cmpl	$999, %r10d
	jbe	.L5772
	vzeroupper
.L5424:
	addl	%edx, %edx
	leal	1(%rdx), %r15d
	movzbl	-272(%rbp,%rdx), %edx
	movzbl	-272(%rbp,%r15), %r9d
	movb	%r9b, -316(%rbp)
.L5432:
	leaq	-317(%rbp), %r15
	movb	%dl, -317(%rbp)
	leaq	(%r15,%r12), %rsi
	movq	%r15, %rdx
	movzbl	-400(%rbp), %r12d
	jmp	.L5420
.L5793:
	testl	%edx, %edx
	jne	.L5510
	movb	$48, -269(%rbp)
	movzbl	-392(%rbp), %r12d
	leaq	.LC39(%rip), %r11
	leaq	-268(%rbp), %r14
	leaq	-269(%rbp), %rcx
	jmp	.L5464
.L5511:
	leaq	.LC38(%rip), %rcx
.L5463:
	leaq	-269(%rbp), %r8
	leaq	-237(%rbp), %rsi
	movq	%rcx, -416(%rbp)
	movq	%r8, %rdi
	movq	%r8, -408(%rbp)
	call	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	cmpb	$48, %r15b
	movzbl	-392(%rbp), %r12d
	movq	-408(%rbp), %rcx
	movq	-416(%rbp), %r11
	movq	%rax, %r14
	jne	.L5464
	cmpq	%rcx, %rax
	je	.L5513
.L5465:
	movq	%r14, %rdi
	movq	%rcx, %r15
	subq	%rcx, %rdi
	andl	$7, %edi
	je	.L5466
	cmpq	$1, %rdi
	je	.L5688
	cmpq	$2, %rdi
	je	.L5689
	cmpq	$3, %rdi
	je	.L5690
	cmpq	$4, %rdi
	je	.L5691
	cmpq	$5, %rdi
	je	.L5692
	cmpq	$6, %rdi
	je	.L5693
	movsbl	(%rcx), %edi
	movq	%r11, -416(%rbp)
	leaq	-268(%rbp), %r15
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-416(%rbp), %r11
	movb	%al, (%rcx)
.L5693:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$1, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movq	-416(%rbp), %rcx
	movq	-408(%rbp), %r11
	movb	%al, -1(%r15)
.L5692:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$1, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movq	-416(%rbp), %rcx
	movq	-408(%rbp), %r11
	movb	%al, -1(%r15)
.L5691:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$1, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movq	-416(%rbp), %rcx
	movq	-408(%rbp), %r11
	movb	%al, -1(%r15)
.L5690:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$1, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movq	-416(%rbp), %rcx
	movq	-408(%rbp), %r11
	movb	%al, -1(%r15)
.L5689:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$1, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movq	-416(%rbp), %rcx
	movq	-408(%rbp), %r11
	movb	%al, -1(%r15)
.L5688:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$1, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %r11
	movq	-416(%rbp), %rcx
	movb	%al, -1(%r15)
	cmpq	%r14, %r15
	je	.L5513
.L5466:
	movsbl	(%r15), %edi
	movq	%rcx, -416(%rbp)
	addq	$8, %r15
	movq	%r11, -408(%rbp)
	call	toupper@PLT
	movsbl	-7(%r15), %edi
	movb	%al, -8(%r15)
	call	toupper@PLT
	movsbl	-6(%r15), %edi
	movb	%al, -7(%r15)
	call	toupper@PLT
	movsbl	-5(%r15), %edi
	movb	%al, -6(%r15)
	call	toupper@PLT
	movsbl	-4(%r15), %edi
	movb	%al, -5(%r15)
	call	toupper@PLT
	movsbl	-3(%r15), %edi
	movb	%al, -4(%r15)
	call	toupper@PLT
	movsbl	-2(%r15), %edi
	movb	%al, -3(%r15)
	call	toupper@PLT
	movsbl	-1(%r15), %edi
	movb	%al, -2(%r15)
	call	toupper@PLT
	movq	-408(%rbp), %r11
	movq	-416(%rbp), %rcx
	movb	%al, -1(%r15)
	cmpq	%r14, %r15
	jne	.L5466
.L5513:
	movl	$2, %eax
	jmp	.L5462
.L5782:
	cmpb	$0, 32(%r13)
	leaq	24(%r13), %r15
	je	.L5798
.L5379:
	leaq	-400(%rbp), %r12
	movq	%r15, %rsi
	movq	%r12, %rdi
	call	_ZNSt6localeC1ERKS_@PLT
	leaq	_ZNSt7__cxx118numpunctIcE2idE(%rip), %rdi
	call	_ZNKSt6locale2id5_M_idEv@PLT
	movq	-400(%rbp), %r11
	movq	8(%r11), %r8
	movq	(%r8,%rax,8), %r15
	testq	%r15, %r15
	je	.L5380
	movq	%r12, %rdi
	call	_ZNSt6localeD1Ev@PLT
	testb	%bl, %bl
	je	.L5799
	movq	(%r15), %rax
	leaq	-352(%rbp), %rbx
	leaq	-384(%rbp), %r12
	movq	%r15, %rsi
	movq	%rbx, %rdi
.LEHB52:
	call	*40(%rax)
.L5384:
	leaq	-384(%rbp), %r12
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0
	movq	%rbx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	-376(%rbp), %rdi
	jmp	.L5385
.L5772:
	vzeroupper
.L5422:
	addl	$48, %edx
	jmp	.L5432
.L5794:
	testl	%r14d, %r14d
	jne	.L5502
	movb	$48, -317(%rbp)
	movzbl	-400(%rbp), %r12d
	leaq	-316(%rbp), %rsi
	leaq	.LC39(%rip), %r8
	leaq	-317(%rbp), %r15
	jmp	.L5435
.L5792:
	cmpq	%r15, %rax
	je	.L5505
.L5436:
	movq	%rsi, %rax
	movq	%r15, %rcx
	subq	%r15, %rax
	andl	$7, %eax
	je	.L5437
	cmpq	$1, %rax
	je	.L5675
	cmpq	$2, %rax
	je	.L5676
	cmpq	$3, %rax
	je	.L5677
	cmpq	$4, %rax
	je	.L5678
	cmpq	$5, %rax
	je	.L5679
	cmpq	$6, %rax
	je	.L5680
	movsbl	(%r15), %edi
	movq	%rsi, -416(%rbp)
	movq	%r8, -408(%rbp)
	call	toupper@PLT
	movq	-416(%rbp), %rsi
	movq	-408(%rbp), %r8
	leaq	-316(%rbp), %rcx
	movb	%al, (%r15)
.L5680:
	movsbl	(%rcx), %edi
	movq	%rsi, -424(%rbp)
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, (%rcx)
	addq	$1, %rcx
.L5679:
	movsbl	(%rcx), %edi
	movq	%rsi, -424(%rbp)
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, (%rcx)
	addq	$1, %rcx
.L5678:
	movsbl	(%rcx), %edi
	movq	%rsi, -424(%rbp)
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, (%rcx)
	addq	$1, %rcx
.L5677:
	movsbl	(%rcx), %edi
	movq	%rsi, -424(%rbp)
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, (%rcx)
	addq	$1, %rcx
.L5676:
	movsbl	(%rcx), %edi
	movq	%rsi, -424(%rbp)
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, (%rcx)
	addq	$1, %rcx
.L5675:
	movsbl	(%rcx), %edi
	movq	%rsi, -424(%rbp)
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, (%rcx)
	addq	$1, %rcx
	cmpq	%rsi, %rcx
	je	.L5505
.L5437:
	movsbl	(%rcx), %edi
	movq	%r8, -416(%rbp)
	movq	%rcx, -408(%rbp)
	movq	%rsi, -424(%rbp)
	call	toupper@PLT
	movq	-408(%rbp), %rdx
	movb	%al, (%rdx)
	movsbl	1(%rdx), %edi
	call	toupper@PLT
	movq	-408(%rbp), %r10
	movb	%al, 1(%r10)
	movsbl	2(%r10), %edi
	call	toupper@PLT
	movq	-408(%rbp), %rsi
	movb	%al, 2(%rsi)
	movsbl	3(%rsi), %edi
	call	toupper@PLT
	movq	-408(%rbp), %r9
	movb	%al, 3(%r9)
	movsbl	4(%r9), %edi
	call	toupper@PLT
	movq	-408(%rbp), %r11
	movb	%al, 4(%r11)
	movsbl	5(%r11), %edi
	call	toupper@PLT
	movq	-408(%rbp), %rdi
	movb	%al, 5(%rdi)
	movsbl	6(%rdi), %edi
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movb	%al, 6(%rcx)
	movsbl	7(%rcx), %edi
	call	toupper@PLT
	movq	-408(%rbp), %rcx
	movq	-424(%rbp), %rsi
	movq	-416(%rbp), %r8
	movb	%al, 7(%rcx)
	addq	$8, %rcx
	cmpq	%rsi, %rcx
	jne	.L5437
.L5505:
	movl	$1, %edx
	movl	$2, %r10d
	jmp	.L5434
.L5510:
	leaq	.LC39(%rip), %rcx
	jmp	.L5463
.L5799:
	movq	(%r15), %r9
	leaq	-352(%rbp), %rbx
	leaq	-384(%rbp), %r12
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	*48(%r9)
.LEHE52:
	jmp	.L5384
.L5504:
	leaq	.LC38(%rip), %r8
	leaq	-316(%rbp), %rsi
	leaq	-317(%rbp), %r15
	jmp	.L5436
.L5798:
	movq	%r15, %rdi
	call	_ZNSt6localeC1Ev@PLT
	movb	$1, 32(%r13)
	jmp	.L5379
.L5517:
	movq	%r13, %rsi
	jmp	.L5476
.L5501:
	movl	$9, %r12d
	jmp	.L5425
.L5500:
	movl	$8, %r12d
	jmp	.L5425
.L5499:
	movl	$7, %r12d
	jmp	.L5425
.L5503:
	leaq	.LC38(%rip), %rsi
	jmp	.L5407
.L5781:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	je	.L5374
.L5779:
	call	__stack_chk_fail@PLT
.L5502:
	leaq	.LC39(%rip), %rsi
	jmp	.L5407
.L5795:
	leaq	-272(%rbp), %rdi
	leaq	.LC53(%rip), %rsi
	movl	$201, %ecx
	movl	$2, %r12d
	rep movsb
	jmp	.L5424
.L5496:
	movl	$1, %r12d
	jmp	.L5422
.L5796:
	movl	$4, %r12d
	movl	%edx, %r8d
	jmp	.L5425
.L5497:
	movl	$3, %r12d
	movl	%edx, %r8d
	jmp	.L5425
.L5380:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
.LEHB53:
	call	_ZSt16__throw_bad_castv@PLT
.LEHE53:
.L5797:
	movl	$1, %r12d
	jmp	.L5429
.L5519:
	endbr64
	movq	%rax, %r13
	jmp	.L5389
.L5398:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	leaq	.LC40(%rip), %rdi
.LEHB54:
	call	_ZSt20__throw_format_errorPKc
.L5783:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	leaq	.LC52(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.L5388:
	movq	%r12, %rdi
	vzeroupper
	leaq	-384(%rbp), %r12
	call	_ZNSt6localeD1Ev@PLT
.L5389:
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	movq	%r13, %rdi
	call	_Unwind_Resume@PLT
.L5370:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
.L5790:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L5779
	call	_ZNSt8__format29__failed_to_parse_format_specEv
.L5520:
	endbr64
	movq	%rax, %r13
	jmp	.L5388
.L5374:
	leaq	.LC51(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.LEHE54:
	.cfi_endproc
.LFE13443:
	.section	.gcc_except_table
.LLSDA13443:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13443-.LLSDACSB13443
.LLSDACSB13443:
	.uleb128 .LEHB49-.LFB13443
	.uleb128 .LEHE49-.LEHB49
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB50-.LFB13443
	.uleb128 .LEHE50-.LEHB50
	.uleb128 .L5519-.LFB13443
	.uleb128 0
	.uleb128 .LEHB51-.LFB13443
	.uleb128 .LEHE51-.LEHB51
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB52-.LFB13443
	.uleb128 .LEHE52-.LEHB52
	.uleb128 .L5519-.LFB13443
	.uleb128 0
	.uleb128 .LEHB53-.LFB13443
	.uleb128 .LEHE53-.LEHB53
	.uleb128 .L5520-.LFB13443
	.uleb128 0
	.uleb128 .LEHB54-.LFB13443
	.uleb128 .LEHE54-.LEHB54
	.uleb128 0
	.uleb128 0
.LLSDACSE13443:
	.section	.text._ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,"axG",@progbits,_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE,comdat
	.size	_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE, .-_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE
	.section	.rodata._ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE.str1.8,"aMS",@progbits,1
	.align 8
.LC57:
	.string	"format error: unmatched '}' in format string"
	.section	.text._ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE,"axG",@progbits,_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE,comdat
	.p2align 4
	.weak	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
	.type	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE, @function
_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE:
.LFB11569:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA11569
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	16+_ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE(%rip), %r11
	movq	%rsi, %r9
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movq	%rdx, %r13
	movq	%rcx, %rdx
	pushq	%r12
	leaq	0(%r13,%rsi), %r10
	.cfi_offset 12, -48
	movq	%rsi, %r12
	movq	%r13, %r15
	pushq	%rbx
	subq	$968, %rsp
	.cfi_offset 3, -56
	vmovq	.LC58(%rip), %xmm2
	movq	%rdi, 24(%rsp)
	leaq	16+_ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE(%rip), %rdi
	leaq	336(%rsp), %rbx
	vmovq	%rdi, %xmm1
	leaq	624(%rsp), %r14
	leaq	928(%rsp), %rcx
	vpinsrq	$1, %rbx, %xmm1, %xmm3
	movq	%fs:40, %rax
	movq	%rax, 952(%rsp)
	xorl	%eax, %eax
	leaq	656(%rsp), %rax
	movq	%rcx, 16(%rsp)
	vpinsrq	$1, %rax, %xmm2, %xmm0
	movq	%rax, 648(%rsp)
	leaq	192(%rsp), %rax
	movq	$256, 640(%rsp)
	movq	%rcx, 912(%rsp)
	movq	$0, 920(%rsp)
	movb	$0, 928(%rsp)
	movq	$256, 320(%rsp)
	movq	%rbx, 328(%rsp)
	movq	%r14, 592(%rsp)
	movq	$-1, 600(%rsp)
	movq	$0, 608(%rsp)
	movq	%rdx, 192(%rsp)
	vmovdqa	%xmm0, 624(%rsp)
	vmovdqa	%xmm3, 304(%rsp)
	movq	%r8, 200(%rsp)
	vmovdqa	.LC56(%rip), %xmm4
	movq	%r14, 208(%rsp)
	movq	$0, 216(%rsp)
	movb	$0, 224(%rsp)
	movq	%r13, 248(%rsp)
	movq	%r10, 256(%rsp)
	movl	$0, 264(%rsp)
	movq	%r11, 240(%rsp)
	movq	%rax, 288(%rsp)
	vmovdqa	%xmm4, 272(%rsp)
	cmpq	$2, %rsi
	je	.L6016
	testq	%rsi, %rsi
	jne	.L5802
.L5861:
	movq	920(%rsp), %rsi
	movq	632(%rsp), %rcx
	movq	648(%rsp), %r8
	movq	%rsi, %rax
	subq	%rcx, %r8
	jne	.L6017
	movq	24(%rsp), %rdi
	movq	912(%rsp), %r11
	movq	16(%rsp), %r12
	leaq	16(%rdi), %rcx
	movq	%rcx, (%rdi)
	cmpq	%r12, %r11
	je	.L5875
.L5874:
	movq	24(%rsp), %rdi
	movq	928(%rsp), %rcx
	movq	%rsi, %rax
	movq	%r11, (%rdi)
	movq	%rcx, 16(%rdi)
.L5876:
	movq	24(%rsp), %r12
	movq	%rax, 8(%r12)
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	addq	$968, %rsp
	movq	%r12, %rax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L6016:
	.cfi_restore_state
	cmpb	$123, 0(%r13)
	je	.L6018
.L5802:
	movq	%r12, %rdx
	movl	$123, %esi
	movq	%r13, %rdi
	movq	%r10, 40(%rsp)
	movq	%r9, 32(%rsp)
	movq	%rax, 8(%rsp)
	call	memchr@PLT
	movq	%r12, %rdx
	movl	$125, %esi
	movq	%r13, %rdi
	movq	%rax, %rbx
	movq	$-1, %r8
	subq	%r13, %rbx
	testq	%rax, %rax
	cmove	%r8, %rbx
	call	memchr@PLT
	movq	40(%rsp), %r10
	movq	32(%rsp), %r8
	movq	$-1, %rsi
	movq	%rax, %r12
	movq	8(%rsp), %rdx
	subq	%r13, %r12
	testq	%rax, %rax
	cmove	%rsi, %r12
	cmpq	%r12, %rbx
	je	.L6019
	.p2align 4,,10
	.p2align 3
.L5808:
	cmpq	%r12, %rbx
	jnb	.L6020
	leaq	1(%rbx), %rsi
	cmpq	%r8, %rsi
	je	.L5814
	movzbl	1(%r15,%rbx), %r11d
	cmpq	$-1, %r12
	je	.L6021
	cmpb	$123, %r11b
	movq	288(%rsp), %r14
	sete	%al
	sete	32(%rsp)
	movzbl	%al, %esi
	movq	16(%r14), %r15
	addq	%rbx, %rsi
	leaq	0(%r13,%rsi), %r8
	movq	%r8, 40(%rsp)
	jne	.L5889
	leaq	1(%r13), %r15
	movq	%r15, 248(%rsp)
.L5819:
	movzbl	(%r15), %ebx
	cmpb	$125, %bl
	je	.L6022
	cmpb	$58, %bl
	je	.L6023
	cmpb	$48, %bl
	je	.L6024
	leal	-49(%rbx), %edx
	cmpb	$8, %dl
	ja	.L5834
	movq	40(%rsp), %r9
	movsbw	%bl, %ax
	subl	$48, %eax
	leaq	2(%r9), %rdi
	movzbl	2(%r9), %r11d
	cmpq	%r10, %rdi
	je	.L5833
	leal	-48(%r11), %r13d
	cmpb	$9, %r13b
	ja	.L5833
	movq	%r10, %r12
	movq	%r15, %rdi
	xorl	%eax, %eax
	movl	$16, %r11d
	subq	%r15, %r12
	andl	$3, %r12d
	je	.L5842
	cmpq	$1, %r12
	je	.L5966
	cmpq	$2, %r12
	je	.L5967
	subl	$48, %ebx
	cmpb	$9, %bl
	ja	.L5835
	movl	$12, %r11d
	movzbl	%bl, %eax
	leaq	1(%r15), %rdi
.L5967:
	movzbl	(%rdi), %r8d
	leal	-48(%r8), %r14d
	cmpb	$9, %r14b
	ja	.L5835
	subl	$4, %r11d
	js	.L6025
	leal	(%rax,%rax,4), %eax
	movzbl	%r14b, %ebx
	leal	(%rbx,%rax,2), %eax
.L6005:
	addq	$1, %rdi
.L5966:
	movzbl	(%rdi), %edx
	leal	-48(%rdx), %r9d
	cmpb	$9, %r9b
	ja	.L5835
	subl	$4, %r11d
	js	.L6026
	leal	(%rax,%rax,4), %r8d
	movzbl	%r9b, %r14d
	leal	(%r14,%r8,2), %eax
.L6007:
	addq	$1, %rdi
	cmpq	%r10, %rdi
	jne	.L5842
.L5843:
	movzbl	(%rdi), %r11d
	jmp	.L5833
	.p2align 4,,10
	.p2align 3
.L6020:
	leaq	1(%r12), %r14
	cmpq	%r8, %r14
	je	.L5856
	cmpb	$125, 1(%r15,%r12)
	jne	.L5856
	movq	288(%rsp), %r15
	movq	16(%r15), %rcx
	movq	%rcx, 40(%rsp)
	testq	%r14, %r14
	jne	.L6027
.L5857:
	movq	40(%rsp), %r9
	movq	%r9, 16(%r15)
	leaq	2(%r13,%r12), %r15
	leaq	-1(%rbx), %r13
	movq	%r10, %r12
	subq	%r14, %r13
	cmpq	$-1, %rbx
	movq	%r15, 248(%rsp)
	cmovne	%r13, %rbx
	subq	%r15, %r12
	je	.L5811
	movq	%r12, %rdx
	movq	%r15, %rdi
	movl	$125, %esi
	movq	%r10, 32(%rsp)
	movq	%r12, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r8
	movq	32(%rsp), %r10
	movq	$-1, %rdi
	movq	%rax, %r12
	subq	%r15, %r12
	testq	%rax, %rax
	cmove	%rdi, %r12
.L5824:
	cmpq	%r12, %rbx
	je	.L6028
	movq	%r15, %r13
	jmp	.L5808
	.p2align 4,,10
	.p2align 3
.L6028:
	movq	288(%rsp), %r13
	movq	16(%r13), %r14
	testq	%r8, %r8
	jne	.L5809
.L5884:
	movq	%r14, 16(%r13)
	movq	%r10, 248(%rsp)
	.p2align 4,,10
	.p2align 3
.L5811:
	movzbl	224(%rsp), %r9d
	testb	%r9b, %r9b
	je	.L5861
	leaq	216(%rsp), %rdi
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L5861
	.p2align 4,,10
	.p2align 3
.L6027:
	movq	%r14, %rsi
	movq	%r13, %rdx
	movq	%rcx, %rdi
.LEHB55:
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	movq	256(%rsp), %r10
	jmp	.L5857
	.p2align 4,,10
	.p2align 3
.L6021:
	cmpb	$123, %r11b
	jne	.L5814
	movq	288(%rsp), %r14
	leaq	0(%r13,%rsi), %r10
	movb	$1, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	16(%r14), %r15
.L5889:
	movq	%r13, %rdx
	movq	%r15, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	movq	40(%rsp), %rsi
	movq	256(%rsp), %r10
	movq	%r15, 16(%r14)
	movq	%r10, %rcx
	leaq	1(%rsi), %r15
	subq	%r15, %rcx
	cmpb	$0, 32(%rsp)
	movq	%r15, 248(%rsp)
	je	.L5819
	cmpq	$-1, %r12
	je	.L6029
	movq	%r10, 40(%rsp)
	testq	%rcx, %rcx
	je	.L5811
	movq	%rcx, %rdx
	movl	$123, %esi
	movq	%r15, %rdi
	subq	$2, %r12
	movq	%rcx, 32(%rsp)
	subq	%rbx, %r12
	call	memchr@PLT
	movq	32(%rsp), %r8
	movq	40(%rsp), %r10
	testq	%rax, %rax
	movq	%rax, %rbx
	je	.L5895
.L5885:
	subq	%r15, %rbx
	jmp	.L5824
	.p2align 4,,10
	.p2align 3
.L6024:
	movq	40(%rsp), %r10
	xorl	%eax, %eax
	movzbl	2(%r10), %r11d
	leaq	2(%r10), %rdi
.L5833:
	cmpb	$125, %r11b
	je	.L5844
	cmpb	$58, %r11b
	jne	.L5834
.L5844:
	cmpl	$2, 264(%rsp)
	movzwl	%ax, %r15d
	je	.L6030
	movl	$1, 264(%rsp)
	xorl	%r13d, %r13d
	cmpb	$58, (%rdi)
	sete	%r13b
	addq	%rdi, %r13
	movq	%r13, 248(%rsp)
.L5828:
	movq	288(%rsp), %rbx
	movzbl	(%rbx), %ecx
	movl	%ecx, %edx
	andl	$15, %ecx
	andl	$15, %edx
	cmpq	%rcx, %r15
	jb	.L6031
	testb	%dl, %dl
	jne	.L5897
	movq	(%rbx), %r9
	shrq	$4, %r9
	cmpq	%r9, %r15
	jnb	.L5850
	salq	$5, %r15
	addq	8(%rbx), %r15
	vmovdqu	(%r15), %xmm5
	movzbl	16(%r15), %edx
	vmovdqa	%xmm5, 160(%rsp)
	.p2align 4,,10
	.p2align 3
.L5850:
	leaq	240(%rsp), %r12
	movb	%dl, 176(%rsp)
	vmovdqu	160(%rsp), %ymm7
	movzbl	%dl, %edx
	movq	%r12, 56(%rsp)
	leaq	56(%rsp), %rsi
	leaq	128(%rsp), %rdi
	vmovdqu	%ymm7, 128(%rsp)
	vzeroupper
	call	_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE
	movq	248(%rsp), %rax
	movq	256(%rsp), %rdi
	cmpq	%rdi, %rax
	je	.L5851
	cmpb	$125, (%rax)
	jne	.L5851
	leaq	1(%rax), %r15
	movq	%rdi, %r11
	movq	%r15, 248(%rsp)
	subq	%r15, %r11
	je	.L5811
	movq	%rdi, 32(%rsp)
	movq	%r11, %rdx
	movl	$123, %esi
	movq	%r15, %rdi
	movq	%r11, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r13
	movq	32(%rsp), %r10
	testq	%rax, %rax
	movq	%rax, %rbx
	je	.L5855
	movq	%r13, %rdx
	movl	$125, %esi
	movq	%r15, %rdi
	subq	%r15, %rbx
	call	memchr@PLT
	movq	40(%rsp), %r8
	movq	32(%rsp), %r10
	testq	%rax, %rax
	movq	%rax, %r12
	je	.L6032
.L5887:
	subq	%r15, %r12
	jmp	.L5824
	.p2align 4,,10
	.p2align 3
.L6029:
	testq	%rcx, %rcx
	je	.L5811
	movq	%rcx, %rdx
	movl	$123, %esi
	movq	%r15, %rdi
	movq	%r10, 32(%rsp)
	movq	%rcx, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r8
	movq	32(%rsp), %r10
	testq	%rax, %rax
	movq	%rax, %rbx
	jne	.L5885
.L6003:
	movq	288(%rsp), %r13
	movq	16(%r13), %r14
	.p2align 4,,10
	.p2align 3
.L5809:
	movq	%r8, %rsi
	movq	%r15, %rdx
	movq	%r14, %rdi
	call	_ZNSt8__format5_SinkIcE8_M_writeESt17basic_string_viewIcSt11char_traitsIcEE
	movq	256(%rsp), %r10
	jmp	.L5884
	.p2align 4,,10
	.p2align 3
.L6022:
	cmpl	$1, 264(%rsp)
	je	.L6033
	movq	272(%rsp), %r15
	movl	$2, 264(%rsp)
	leaq	1(%r15), %rsi
	movq	%rsi, 272(%rsp)
	jmp	.L5828
	.p2align 4,,10
	.p2align 3
.L5897:
	xorl	%edx, %edx
	jmp	.L5850
	.p2align 4,,10
	.p2align 3
.L6031:
	movq	(%rbx), %rdx
	leaq	(%r15,%r15,4), %rcx
	salq	$4, %r15
	addq	8(%rbx), %r15
	vmovdqa	(%r15), %xmm6
	shrq	$4, %rdx
	vmovdqa	%xmm6, 160(%rsp)
	shrq	%cl, %rdx
	andl	$31, %edx
	jmp	.L5850
	.p2align 4,,10
	.p2align 3
.L6017:
	movabsq	$9223372036854775807, %r10
	subq	%rsi, %r10
	cmpq	%r8, %r10
	jb	.L6034
	movq	912(%rsp), %r14
	movq	16(%rsp), %r13
	leaq	(%rsi,%r8), %rbx
	cmpq	%r13, %r14
	je	.L5900
	movq	928(%rsp), %r15
.L5870:
	cmpq	%rbx, %r15
	jb	.L5871
	leaq	(%r14,%rsi), %rdi
	cmpq	$1, %r8
	je	.L6035
	movq	%r8, %rdx
	movq	%rcx, %rsi
	call	memcpy@PLT
	jmp	.L5873
	.p2align 4,,10
	.p2align 3
.L6023:
	cmpl	$1, 264(%rsp)
	je	.L6036
	movq	272(%rsp), %r15
	movq	40(%rsp), %r14
	movl	$2, 264(%rsp)
	leaq	1(%r15), %r8
	addq	$2, %r14
	movq	%r8, 272(%rsp)
	movq	%r14, 248(%rsp)
	jmp	.L5828
	.p2align 4,,10
	.p2align 3
.L6018:
	cmpb	$125, 1(%r13)
	jne	.L5802
	movl	$2, 264(%rsp)
	leaq	1(%r13), %r15
	movq	%r15, 248(%rsp)
	movq	$1, 272(%rsp)
	testb	$15, 192(%rsp)
	jne	.L6037
	xorl	%esi, %esi
	shrq	$4, %rdx
	je	.L5810
	vmovdqu	(%r8), %xmm8
	movzbl	16(%r8), %esi
	vmovdqa	%xmm8, 96(%rsp)
.L5810:
	leaq	240(%rsp), %rdx
	movb	%sil, 112(%rsp)
	vmovdqu	96(%rsp), %ymm10
	leaq	64(%rsp), %rdi
	movq	%rdx, 48(%rsp)
	movzbl	%sil, %edx
	leaq	48(%rsp), %rsi
	vmovdqu	%ymm10, 64(%rsp)
	vzeroupper
	call	_ZNSt16basic_format_argISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE8_M_visitIZNS1_19_Formatting_scannerIS3_cE13_M_format_argEmEUlRT_E_EEDcOS9_NS1_6_Arg_tE
.LEHE55:
	jmp	.L5811
	.p2align 4,,10
	.p2align 3
.L5895:
	movq	$-1, %rbx
	jmp	.L5824
	.p2align 4,,10
	.p2align 3
.L5842:
	movzbl	(%rdi), %esi
	leal	-48(%rsi), %ecx
	cmpb	$9, %cl
	ja	.L5835
	subl	$4, %r11d
	js	.L5836
	leal	(%rax,%rax,4), %eax
	movzbl	%cl, %r9d
	leal	(%r9,%rax,2), %eax
.L5837:
	movzbl	1(%rdi), %r12d
	leaq	1(%rdi), %r13
	movq	%r13, %rdi
	leal	-48(%r12), %r8d
	cmpb	$9, %r8b
	ja	.L5835
	movl	%r11d, %edi
	subl	$4, %edi
	js	.L6038
	leal	(%rax,%rax,4), %ecx
	movzbl	%r8b, %ebx
	leal	(%rbx,%rcx,2), %eax
.L6009:
	movzbl	1(%r13), %edx
	leaq	1(%r13), %rdi
	leal	-48(%rdx), %r9d
	cmpb	$9, %r9b
	ja	.L5835
	movl	%r11d, %r12d
	subl	$8, %r12d
	js	.L6039
	leal	(%rax,%rax,4), %eax
	movzbl	%r9b, %r14d
	leal	(%r14,%rax,2), %eax
.L6011:
	movzbl	2(%r13), %esi
	leaq	2(%r13), %rdi
	leal	-48(%rsi), %ecx
	cmpb	$9, %cl
	ja	.L5835
	subl	$12, %r11d
	js	.L6040
	leal	(%rax,%rax,4), %r9d
	movzbl	%cl, %r12d
	leal	(%r12,%r9,2), %eax
.L6013:
	leaq	3(%r13), %rdi
	cmpq	%r10, %rdi
	jne	.L5842
	jmp	.L5843
	.p2align 4,,10
	.p2align 3
.L5871:
	leaq	912(%rsp), %r12
	xorl	%edx, %edx
	movq	%r12, %rdi
.LEHB56:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.LEHE56:
.L5873:
	movq	912(%rsp), %rsi
	movq	24(%rsp), %r9
	movq	%rbx, 920(%rsp)
	movq	16(%rsp), %rax
	movb	$0, (%rsi,%rbx)
	movq	632(%rsp), %rdx
	leaq	16(%r9), %rcx
	movq	912(%rsp), %r11
	movq	%rcx, (%r9)
	movq	%rdx, 648(%rsp)
	movq	920(%rsp), %rsi
	cmpq	%rax, %r11
	jne	.L5874
	movq	%rsi, %rax
.L5875:
	leaq	1(%rax), %r11
	cmpl	$8, %r11d
	jnb	.L5877
	testb	$4, %r11b
	jne	.L6041
	testl	%r11d, %r11d
	je	.L5876
	movzbl	928(%rsp), %r13d
	movb	%r13b, (%rcx)
	testb	$2, %r11b
	je	.L5876
	movq	16(%rsp), %r8
	movl	%r11d, %r15d
	movzwl	-2(%r8,%r15), %esi
	movw	%si, -2(%rcx,%r15)
	jmp	.L5876
	.p2align 4,,10
	.p2align 3
.L5855:
	movq	%r13, %rdx
	movl	$125, %esi
	movq	%r15, %rdi
	movq	%r10, 32(%rsp)
	movq	%r13, 40(%rsp)
	call	memchr@PLT
	movq	40(%rsp), %r8
	movq	32(%rsp), %r10
	testq	%rax, %rax
	movq	%rax, %r12
	je	.L6003
	movq	$-1, %rbx
	jmp	.L5887
	.p2align 4,,10
	.p2align 3
.L6032:
	movq	$-1, %r12
	jmp	.L5824
.L5877:
	movq	928(%rsp), %rdx
	movl	%r11d, %r9d
	leaq	8(%rcx), %r12
	andq	$-8, %r12
	movq	%rdx, (%rcx)
	movq	16(%rsp), %r8
	movq	-8(%r8,%r9), %rdi
	movq	%rdi, -8(%rcx,%r9)
	subq	%r12, %rcx
	addl	%ecx, %r11d
	subq	%rcx, %r8
	movl	%r11d, %edx
	andl	$-8, %edx
	cmpl	$8, %edx
	jb	.L5876
	xorl	%ecx, %ecx
	leal	-1(%rdx), %r11d
	movl	$8, %r14d
	movq	(%r8,%rcx), %r10
	shrl	$3, %r11d
	andl	$7, %r11d
	movq	%r10, (%r12,%rcx)
	cmpl	%edx, %r14d
	jnb	.L5876
	testl	%r11d, %r11d
	je	.L5881
	cmpl	$1, %r11d
	je	.L5969
	cmpl	$2, %r11d
	je	.L5970
	cmpl	$3, %r11d
	je	.L5971
	cmpl	$4, %r11d
	je	.L5972
	cmpl	$5, %r11d
	je	.L5973
	cmpl	$6, %r11d
	je	.L5974
	movq	(%r8,%r14), %rbx
	movq	%rbx, (%r12,%r14)
	movl	$16, %r14d
.L5974:
	movl	%r14d, %r13d
	addl	$8, %r14d
	movq	(%r8,%r13), %r15
	movq	%r15, (%r12,%r13)
.L5973:
	movl	%r14d, %esi
	addl	$8, %r14d
	movq	(%r8,%rsi), %r9
	movq	%r9, (%r12,%rsi)
.L5972:
	movl	%r14d, %edi
	addl	$8, %r14d
	movq	(%r8,%rdi), %r11
	movq	%r11, (%r12,%rdi)
.L5971:
	movl	%r14d, %ecx
	addl	$8, %r14d
	movq	(%r8,%rcx), %r10
	movq	%r10, (%r12,%rcx)
.L5970:
	movl	%r14d, %ebx
	addl	$8, %r14d
	movq	(%r8,%rbx), %r13
	movq	%r13, (%r12,%rbx)
.L5969:
	movl	%r14d, %r15d
	addl	$8, %r14d
	movq	(%r8,%r15), %rsi
	movq	%rsi, (%r12,%r15)
	cmpl	%edx, %r14d
	jnb	.L5876
.L5881:
	movl	%r14d, %edi
	leal	8(%r14), %r11d
	leal	16(%r14), %r10d
	movq	(%r8,%rdi), %r9
	leal	24(%r14), %r13d
	leal	32(%r14), %esi
	movq	%r9, (%r12,%rdi)
	movq	(%r8,%r11), %rcx
	movq	%rcx, (%r12,%r11)
	movq	(%r8,%r10), %rbx
	leal	40(%r14), %r11d
	leal	48(%r14), %ecx
	movq	%rbx, (%r12,%r10)
	movq	(%r8,%r13), %r15
	leal	56(%r14), %ebx
	addl	$64, %r14d
	movq	%r15, (%r12,%r13)
	movq	(%r8,%rsi), %rdi
	movq	%rdi, (%r12,%rsi)
	movq	(%r8,%r11), %r9
	movq	%r9, (%r12,%r11)
	movq	(%r8,%rcx), %r10
	movq	%r10, (%r12,%rcx)
	movq	(%r8,%rbx), %r13
	movq	%r13, (%r12,%rbx)
	cmpl	%edx, %r14d
	jb	.L5881
	jmp	.L5876
	.p2align 4,,10
	.p2align 3
.L6019:
	movq	%rdx, %r13
	jmp	.L5809
.L5900:
	movl	$15, %r15d
	jmp	.L5870
.L6035:
	movzbl	(%rcx), %r8d
	movb	%r8b, (%rdi)
	jmp	.L5873
.L5835:
	cmpq	%rdi, %r15
	jne	.L5843
.L5834:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
.LEHB57:
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
.LEHE57:
	.p2align 4,,10
	.p2align 3
.L5836:
	movl	$10, %ebx
	mulw	%bx
	jo	.L5834
	movzbl	%cl, %edx
	addw	%ax, %dx
	jc	.L5834
	movl	%edx, %eax
	jmp	.L5837
.L6038:
	movl	$10, %r14d
	mulw	%r14w
	jo	.L5834
	movzbl	%r8b, %esi
	addw	%ax, %si
	jc	.L5834
	movl	%esi, %eax
	jmp	.L6009
.L6039:
	movl	$10, %r8d
	mulw	%r8w
	jo	.L5834
	movzbl	%r9b, %edi
	addw	%ax, %di
	jc	.L5834
	movl	%edi, %eax
	jmp	.L6011
.L6040:
	movl	$10, %ebx
	mulw	%bx
	jo	.L5834
	movzbl	%cl, %edx
	addw	%ax, %dx
	jc	.L5834
	movl	%edx, %eax
	jmp	.L6013
.L6041:
	movl	928(%rsp), %r12d
	movl	%r11d, %r10d
	movl	%r12d, (%rcx)
	movq	16(%rsp), %rbx
	movl	-4(%rbx,%r10), %r14d
	movl	%r14d, -4(%rcx,%r10)
	jmp	.L5876
.L6037:
	vmovdqa	(%r8), %xmm9
	movq	%rdx, %rsi
	shrq	$4, %rsi
	andl	$31, %esi
	vmovdqa	%xmm9, 96(%rsp)
	jmp	.L5810
.L6026:
	movl	$10, %r13d
	mulw	%r13w
	jo	.L5834
	movzbl	%r9b, %r12d
	addw	%ax, %r12w
	jc	.L5834
	movl	%r12d, %eax
	jmp	.L6007
.L6025:
	movl	$10, %esi
	mulw	%si
	jo	.L5834
	movzbl	%r14b, %ecx
	addw	%ax, %cx
	jc	.L5834
	movl	%ecx, %eax
	jmp	.L6005
.L6033:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	je	.L5827
.L6001:
	call	__stack_chk_fail@PLT
.L6034:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	leaq	.LC33(%rip), %rdi
	leaq	912(%rsp), %r12
.LEHB58:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE58:
.L5903:
	endbr64
	movq	%rax, %r14
	jmp	.L5867
.L6030:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
.LEHB59:
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
.L6036:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
.L5814:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	call	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
.L5851:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	call	_ZNSt8__format39__unmatched_left_brace_in_format_stringEv
.L5904:
	endbr64
	movq	%rax, %r14
	jmp	.L5865
.L5856:
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	leaq	.LC57(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.LEHE59:
.L5865:
	cmpb	$0, 224(%rsp)
	jne	.L6042
.L5866:
	leaq	912(%rsp), %r12
.L5867:
	leaq	16+_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE(%rip), %rax
	movq	%r12, %rdi
	movq	%rax, 624(%rsp)
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	952(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6001
	movq	%r14, %rdi
.LEHB60:
	call	_Unwind_Resume@PLT
.LEHE60:
.L6042:
	leaq	216(%rsp), %rdi
	vzeroupper
	call	_ZNSt6localeD1Ev@PLT
	jmp	.L5866
.L5827:
.LEHB61:
	call	_ZNSt8__format39__conflicting_indexing_in_format_stringEv
.LEHE61:
	.cfi_endproc
.LFE11569:
	.section	.gcc_except_table
.LLSDA11569:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11569-.LLSDACSB11569
.LLSDACSB11569:
	.uleb128 .LEHB55-.LFB11569
	.uleb128 .LEHE55-.LEHB55
	.uleb128 .L5904-.LFB11569
	.uleb128 0
	.uleb128 .LEHB56-.LFB11569
	.uleb128 .LEHE56-.LEHB56
	.uleb128 .L5903-.LFB11569
	.uleb128 0
	.uleb128 .LEHB57-.LFB11569
	.uleb128 .LEHE57-.LEHB57
	.uleb128 .L5904-.LFB11569
	.uleb128 0
	.uleb128 .LEHB58-.LFB11569
	.uleb128 .LEHE58-.LEHB58
	.uleb128 .L5903-.LFB11569
	.uleb128 0
	.uleb128 .LEHB59-.LFB11569
	.uleb128 .LEHE59-.LEHB59
	.uleb128 .L5904-.LFB11569
	.uleb128 0
	.uleb128 .LEHB60-.LFB11569
	.uleb128 .LEHE60-.LEHB60
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB61-.LFB11569
	.uleb128 .LEHE61-.LEHB61
	.uleb128 .L5904-.LFB11569
	.uleb128 0
.LLSDACSE11569:
	.section	.text._ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE,"axG",@progbits,_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE,comdat
	.size	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE, .-_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
	.section	.rodata._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv.str1.1,"aMS",@progbits,1
.LC59:
	.string	"n is too large "
.LC60:
	.string	"Native"
	.section	.rodata._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv.str1.8,"aMS",@progbits,1
	.align 8
.LC62:
	.string	"{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}"
	.section	.rodata._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv.str1.1
.LC63:
	.string	"Opt"
.LC64:
	.string	"AVX2"
.LC65:
	.string	"Parallel"
.LC66:
	.string	"Parallel_AVX2"
.LC67:
	.string	"Dynamic_AVX2"
.LC68:
	.string	"Manual"
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv,comdat
	.align 2
	.p2align 4
	.weak	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv
	.type	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv, @function
_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv:
.LFB12470:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12470
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vxorps	%xmm1, %xmm1, %xmm1
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$264, %rsp
	.cfi_def_cfa_offset 320
	movq	(%rdi), %rcx
	movq	16(%rdi), %rsi
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	movq	8(%rdi), %rax
	vmovss	(%rcx), %xmm0
	movl	(%rax), %r8d
	imull	$10000, %r8d, %edx
	vcvtsi2ssl	%edx, %xmm1, %xmm2
	vdivss	%xmm2, %xmm0, %xmm3
	vcvttss2sil	%xmm3, %ebx
	cmpl	(%rsi), %ebx
	jg	.L6080
	movq	%rdi, %rbp
	movq	32(%rdi), %rdi
	movl	%ebx, %ecx
	movq	24(%rbp), %r9
	movq	(%rdi), %rdx
	leaq	_Z25native_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	movq	(%r9), %rsi
.LEHB62:
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	24(%rbp), %r11
	movq	32(%rbp), %r10
	movl	%ebx, %ecx
	movq	8(%rbp), %r8
	leaq	_Z22opt_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 104(%rsp)
	movq	(%r11), %rsi
	movq	(%r10), %rdx
	movl	(%r8), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r12
	movq	32(%rbp), %r13
	movl	%ebx, %ecx
	movq	24(%rbp), %r14
	leaq	_Z23avx2_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 8(%rsp)
	movq	0(%r13), %rdx
	movl	(%r12), %r8d
	imull	$10000, %ebx, %r13d
	movq	(%r14), %rsi
	leaq	.LC60(%rip), %r14
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	32(%rbp), %rax
	movq	8(%rbp), %r15
	leaq	_Z27parallel_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	movq	24(%rbp), %rcx
	vmovsd	%xmm0, 16(%rsp)
	movq	(%rax), %rdx
	movl	(%r15), %r8d
	leaq	144(%rsp), %r15
	movq	(%rcx), %rsi
	movl	%ebx, %ecx
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %rdi
	movq	32(%rbp), %rdx
	movl	%ebx, %ecx
	movq	24(%rbp), %rsi
	vmovsd	%xmm0, 24(%rsp)
	movl	(%rdi), %r8d
	movq	(%rdx), %rdx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	movq	(%rsi), %rsi
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r9
	movq	24(%rbp), %r10
	movl	%ebx, %ecx
	movq	32(%rbp), %r8
	leaq	_Z32parallel_avx2_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	movq	(%r10), %rsi
	movq	(%r8), %rdx
	movl	(%r9), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r11
	movq	32(%rbp), %r12
	movl	%ebx, %ecx
	movq	24(%rbp), %rbp
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi10000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	movq	(%r12), %rdx
	movl	(%r11), %r8d
	leaq	112(%rsp), %r12
	movq	0(%rbp), %rsi
	movslq	%ebx, %rbp
	salq	$3, %rbp
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r12, %rdi
	shrq	$10, %rbp
	vcvtsi2sdl	%r13d, %xmm5, %xmm9
	vmovsd	104(%rsp), %xmm14
	movl	$40, %esi
	movq	%r15, %r8
	movq	%r14, 192(%rsp)
	leaq	.LC62(%rip), %r14
	movabsq	$4434527798, %rcx
	vdivsd	8(%rsp), %xmm9, %xmm6
	movq	%r14, %rdx
	vdivsd	16(%rsp), %xmm9, %xmm7
	vdivsd	24(%rsp), %xmm9, %xmm10
	vdivsd	32(%rsp), %xmm9, %xmm11
	vdivsd	40(%rsp), %xmm9, %xmm12
	movl	%ebx, 160(%rsp)
	vdivsd	%xmm14, %xmm9, %xmm15
	movq	%rbp, 176(%rsp)
	movl	$10000, 144(%rsp)
	vmovsd	%xmm0, 96(%rsp)
	vmovsd	%xmm6, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	vmovsd	%xmm10, 64(%rsp)
	vmovsd	%xmm11, 72(%rsp)
	vmovsd	%xmm12, 80(%rsp)
	vmovsd	%xmm14, 208(%rsp)
	vdivsd	%xmm0, %xmm9, %xmm13
	vmulsd	.LC61(%rip), %xmm15, %xmm1
	vmovsd	%xmm1, 224(%rsp)
	vmovsd	%xmm13, 88(%rsp)
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE62:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB63:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE63:
	movq	112(%rsp), %rdi
	leaq	128(%rsp), %r13
	cmpq	%r13, %rdi
	je	.L6046
	movq	128(%rsp), %rax
	leaq	1(%rax), %rsi
	call	_ZdlPvm@PLT
.L6046:
	vmovsd	8(%rsp), %xmm2
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm0
	movl	$40, %esi
	leaq	.LC63(%rip), %rcx
	movl	$10000, 144(%rsp)
	vmulsd	48(%rsp), %xmm0, %xmm3
	movq	%rcx, 192(%rsp)
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm2, 208(%rsp)
	vmovsd	%xmm3, 224(%rsp)
.LEHB64:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE64:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB65:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE65:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6047
	movq	128(%rsp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
.L6047:
	vmovsd	.LC61(%rip), %xmm8
	movl	$40, %esi
	movq	%r14, %rdx
	movq	%r15, %r8
	vmulsd	56(%rsp), %xmm8, %xmm5
	vmovsd	16(%rsp), %xmm4
	leaq	.LC64(%rip), %rdi
	movabsq	$4434527798, %rcx
	movq	%rdi, 192(%rsp)
	movq	%r12, %rdi
	movl	$10000, 144(%rsp)
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm4, 208(%rsp)
	vmovsd	%xmm5, 224(%rsp)
.LEHB66:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE66:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB67:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE67:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6048
	movq	128(%rsp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L6048:
	vmovsd	24(%rsp), %xmm9
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm6
	vmulsd	64(%rsp), %xmm6, %xmm7
	leaq	.LC65(%rip), %r9
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$10000, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r9, 192(%rsp)
	vmovsd	%xmm9, 208(%rsp)
	vmovsd	%xmm7, 224(%rsp)
.LEHB68:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE68:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB69:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE69:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6049
	movq	128(%rsp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L6049:
	vmovsd	40(%rsp), %xmm10
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm11
	vmulsd	80(%rsp), %xmm11, %xmm12
	leaq	.LC66(%rip), %r10
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$10000, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r10, 192(%rsp)
	vmovsd	%xmm10, 208(%rsp)
	vmovsd	%xmm12, 224(%rsp)
.LEHB70:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE70:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB71:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE71:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6050
	movq	128(%rsp), %r11
	leaq	1(%r11), %rsi
	call	_ZdlPvm@PLT
.L6050:
	vmovsd	32(%rsp), %xmm13
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm14
	vmulsd	72(%rsp), %xmm14, %xmm15
	leaq	.LC67(%rip), %rax
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$10000, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%rax, 192(%rsp)
	vmovsd	%xmm13, 208(%rsp)
	vmovsd	%xmm15, 224(%rsp)
.LEHB72:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE72:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB73:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE73:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6051
	movq	128(%rsp), %rcx
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
.L6051:
	vmovsd	96(%rsp), %xmm1
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm2
	vmulsd	88(%rsp), %xmm2, %xmm0
	movl	%ebx, 160(%rsp)
	movl	$40, %esi
	leaq	.LC68(%rip), %rbx
	movq	%rbp, 176(%rsp)
	movabsq	$4434527798, %rcx
	movl	$10000, 144(%rsp)
	movq	%rbx, 192(%rsp)
	vmovsd	%xmm1, 208(%rsp)
	vmovsd	%xmm0, 224(%rsp)
.LEHB74:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE74:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB75:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE75:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6043
	movq	128(%rsp), %r12
	leaq	1(%r12), %rsi
	call	_ZdlPvm@PLT
.L6043:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6078
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6080:
	.cfi_restore_state
	leaq	_ZSt4cout(%rip), %r15
	movl	$15, %edx
	leaq	.LC59(%rip), %rsi
	movq	%r15, %rdi
.LEHB76:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r15, %rdi
	movl	%ebx, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rdi
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6078
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L6065:
	.cfi_restore_state
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L6066
.L6078:
	call	__stack_chk_fail@PLT
.L6074:
	endbr64
.L6079:
	movq	%rax, %rbp
	jmp	.L6065
.L6073:
	endbr64
	jmp	.L6079
.L6072:
	endbr64
	jmp	.L6079
.L6071:
	endbr64
	jmp	.L6079
.L6069:
	endbr64
	jmp	.L6079
.L6068:
	endbr64
	jmp	.L6079
.L6070:
	endbr64
	jmp	.L6079
.L6066:
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE76:
	.cfi_endproc
.LFE12470:
	.section	.gcc_except_table
.LLSDA12470:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12470-.LLSDACSB12470
.LLSDACSB12470:
	.uleb128 .LEHB62-.LFB12470
	.uleb128 .LEHE62-.LEHB62
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB63-.LFB12470
	.uleb128 .LEHE63-.LEHB63
	.uleb128 .L6068-.LFB12470
	.uleb128 0
	.uleb128 .LEHB64-.LFB12470
	.uleb128 .LEHE64-.LEHB64
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB65-.LFB12470
	.uleb128 .LEHE65-.LEHB65
	.uleb128 .L6069-.LFB12470
	.uleb128 0
	.uleb128 .LEHB66-.LFB12470
	.uleb128 .LEHE66-.LEHB66
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB67-.LFB12470
	.uleb128 .LEHE67-.LEHB67
	.uleb128 .L6070-.LFB12470
	.uleb128 0
	.uleb128 .LEHB68-.LFB12470
	.uleb128 .LEHE68-.LEHB68
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB69-.LFB12470
	.uleb128 .LEHE69-.LEHB69
	.uleb128 .L6071-.LFB12470
	.uleb128 0
	.uleb128 .LEHB70-.LFB12470
	.uleb128 .LEHE70-.LEHB70
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB71-.LFB12470
	.uleb128 .LEHE71-.LEHB71
	.uleb128 .L6072-.LFB12470
	.uleb128 0
	.uleb128 .LEHB72-.LFB12470
	.uleb128 .LEHE72-.LEHB72
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB73-.LFB12470
	.uleb128 .LEHE73-.LEHB73
	.uleb128 .L6073-.LFB12470
	.uleb128 0
	.uleb128 .LEHB74-.LFB12470
	.uleb128 .LEHE74-.LEHB74
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB75-.LFB12470
	.uleb128 .LEHE75-.LEHB75
	.uleb128 .L6074-.LFB12470
	.uleb128 0
	.uleb128 .LEHB76-.LFB12470
	.uleb128 .LEHE76-.LEHB76
	.uleb128 0
	.uleb128 0
.LLSDACSE12470:
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv,comdat
	.size	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv, .-_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv,comdat
	.align 2
	.p2align 4
	.weak	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv
	.type	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv, @function
_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv:
.LFB12466:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12466
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vxorps	%xmm1, %xmm1, %xmm1
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$264, %rsp
	.cfi_def_cfa_offset 320
	movq	(%rdi), %rcx
	movq	16(%rdi), %rsi
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	movq	8(%rdi), %rax
	vmovss	(%rcx), %xmm0
	movl	(%rax), %r8d
	movl	%r8d, %edx
	sall	$4, %edx
	vcvtsi2ssl	%edx, %xmm1, %xmm2
	vdivss	%xmm2, %xmm0, %xmm3
	vcvttss2sil	%xmm3, %ebx
	cmpl	(%rsi), %ebx
	jg	.L6118
	movq	%rdi, %rbp
	movq	32(%rdi), %rdi
	movl	%ebx, %ecx
	movq	24(%rbp), %r9
	movq	(%rdi), %rdx
	leaq	_Z25native_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	movq	(%r9), %rsi
.LEHB77:
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	24(%rbp), %r11
	movq	32(%rbp), %r10
	movl	%ebx, %ecx
	movq	8(%rbp), %r8
	leaq	_Z22opt_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 104(%rsp)
	movq	(%r11), %rsi
	movq	(%r10), %rdx
	movl	(%r8), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r12
	movq	32(%rbp), %r13
	movl	%ebx, %ecx
	movq	24(%rbp), %r14
	leaq	_Z23avx2_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 8(%rsp)
	movq	0(%r13), %rdx
	movl	(%r12), %r8d
	movl	%ebx, %r13d
	movq	(%r14), %rsi
	sall	$4, %r13d
	leaq	.LC60(%rip), %r14
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	32(%rbp), %rax
	movq	8(%rbp), %r15
	leaq	_Z27parallel_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	movq	24(%rbp), %rcx
	vmovsd	%xmm0, 16(%rsp)
	movq	(%rax), %rdx
	movl	(%r15), %r8d
	leaq	144(%rsp), %r15
	movq	(%rcx), %rsi
	movl	%ebx, %ecx
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %rdi
	movq	32(%rbp), %rdx
	movl	%ebx, %ecx
	movq	24(%rbp), %rsi
	vmovsd	%xmm0, 24(%rsp)
	movl	(%rdi), %r8d
	movq	(%rdx), %rdx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	movq	(%rsi), %rsi
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r9
	movq	24(%rbp), %r10
	movl	%ebx, %ecx
	movq	32(%rbp), %r8
	leaq	_Z32parallel_avx2_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	movq	(%r10), %rsi
	movq	(%r8), %rdx
	movl	(%r9), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r11
	movq	32(%rbp), %r12
	movl	%ebx, %ecx
	movq	24(%rbp), %rbp
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi16EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	movq	(%r12), %rdx
	movl	(%r11), %r8d
	leaq	112(%rsp), %r12
	movq	0(%rbp), %rsi
	movslq	%ebx, %rbp
	salq	$3, %rbp
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r12, %rdi
	shrq	$10, %rbp
	vcvtsi2sdl	%r13d, %xmm5, %xmm9
	vmovsd	104(%rsp), %xmm14
	movl	$40, %esi
	movq	%r15, %r8
	movq	%r14, 192(%rsp)
	leaq	.LC62(%rip), %r14
	movabsq	$4434527798, %rcx
	vdivsd	8(%rsp), %xmm9, %xmm6
	movq	%r14, %rdx
	vdivsd	16(%rsp), %xmm9, %xmm7
	vdivsd	24(%rsp), %xmm9, %xmm10
	vdivsd	32(%rsp), %xmm9, %xmm11
	vdivsd	40(%rsp), %xmm9, %xmm12
	movl	%ebx, 160(%rsp)
	vdivsd	%xmm14, %xmm9, %xmm15
	movq	%rbp, 176(%rsp)
	movl	$16, 144(%rsp)
	vmovsd	%xmm0, 96(%rsp)
	vmovsd	%xmm6, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	vmovsd	%xmm10, 64(%rsp)
	vmovsd	%xmm11, 72(%rsp)
	vmovsd	%xmm12, 80(%rsp)
	vmovsd	%xmm14, 208(%rsp)
	vdivsd	%xmm0, %xmm9, %xmm13
	vmulsd	.LC61(%rip), %xmm15, %xmm1
	vmovsd	%xmm1, 224(%rsp)
	vmovsd	%xmm13, 88(%rsp)
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE77:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB78:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE78:
	movq	112(%rsp), %rdi
	leaq	128(%rsp), %r13
	cmpq	%r13, %rdi
	je	.L6084
	movq	128(%rsp), %rax
	leaq	1(%rax), %rsi
	call	_ZdlPvm@PLT
.L6084:
	vmovsd	8(%rsp), %xmm2
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm0
	movl	$40, %esi
	leaq	.LC63(%rip), %rcx
	movl	$16, 144(%rsp)
	vmulsd	48(%rsp), %xmm0, %xmm3
	movq	%rcx, 192(%rsp)
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm2, 208(%rsp)
	vmovsd	%xmm3, 224(%rsp)
.LEHB79:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE79:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB80:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE80:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6085
	movq	128(%rsp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
.L6085:
	vmovsd	.LC61(%rip), %xmm8
	movl	$40, %esi
	movq	%r14, %rdx
	movq	%r15, %r8
	vmulsd	56(%rsp), %xmm8, %xmm5
	vmovsd	16(%rsp), %xmm4
	leaq	.LC64(%rip), %rdi
	movabsq	$4434527798, %rcx
	movq	%rdi, 192(%rsp)
	movq	%r12, %rdi
	movl	$16, 144(%rsp)
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm4, 208(%rsp)
	vmovsd	%xmm5, 224(%rsp)
.LEHB81:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE81:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB82:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE82:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6086
	movq	128(%rsp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L6086:
	vmovsd	24(%rsp), %xmm9
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm6
	vmulsd	64(%rsp), %xmm6, %xmm7
	leaq	.LC65(%rip), %r9
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$16, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r9, 192(%rsp)
	vmovsd	%xmm9, 208(%rsp)
	vmovsd	%xmm7, 224(%rsp)
.LEHB83:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE83:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB84:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE84:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6087
	movq	128(%rsp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L6087:
	vmovsd	40(%rsp), %xmm10
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm11
	vmulsd	80(%rsp), %xmm11, %xmm12
	leaq	.LC66(%rip), %r10
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$16, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r10, 192(%rsp)
	vmovsd	%xmm10, 208(%rsp)
	vmovsd	%xmm12, 224(%rsp)
.LEHB85:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE85:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB86:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE86:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6088
	movq	128(%rsp), %r11
	leaq	1(%r11), %rsi
	call	_ZdlPvm@PLT
.L6088:
	vmovsd	32(%rsp), %xmm13
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm14
	vmulsd	72(%rsp), %xmm14, %xmm15
	leaq	.LC67(%rip), %rax
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$16, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%rax, 192(%rsp)
	vmovsd	%xmm13, 208(%rsp)
	vmovsd	%xmm15, 224(%rsp)
.LEHB87:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE87:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB88:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE88:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6089
	movq	128(%rsp), %rcx
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
.L6089:
	vmovsd	96(%rsp), %xmm1
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm2
	vmulsd	88(%rsp), %xmm2, %xmm0
	movl	%ebx, 160(%rsp)
	movl	$40, %esi
	leaq	.LC68(%rip), %rbx
	movq	%rbp, 176(%rsp)
	movabsq	$4434527798, %rcx
	movl	$16, 144(%rsp)
	movq	%rbx, 192(%rsp)
	vmovsd	%xmm1, 208(%rsp)
	vmovsd	%xmm0, 224(%rsp)
.LEHB89:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE89:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB90:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE90:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6081
	movq	128(%rsp), %r12
	leaq	1(%r12), %rsi
	call	_ZdlPvm@PLT
.L6081:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6116
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6118:
	.cfi_restore_state
	leaq	_ZSt4cout(%rip), %r15
	movl	$15, %edx
	leaq	.LC59(%rip), %rsi
	movq	%r15, %rdi
.LEHB91:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r15, %rdi
	movl	%ebx, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rdi
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6116
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L6103:
	.cfi_restore_state
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L6104
.L6116:
	call	__stack_chk_fail@PLT
.L6112:
	endbr64
.L6117:
	movq	%rax, %rbp
	jmp	.L6103
.L6111:
	endbr64
	jmp	.L6117
.L6110:
	endbr64
	jmp	.L6117
.L6109:
	endbr64
	jmp	.L6117
.L6107:
	endbr64
	jmp	.L6117
.L6106:
	endbr64
	jmp	.L6117
.L6108:
	endbr64
	jmp	.L6117
.L6104:
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE91:
	.cfi_endproc
.LFE12466:
	.section	.gcc_except_table
.LLSDA12466:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12466-.LLSDACSB12466
.LLSDACSB12466:
	.uleb128 .LEHB77-.LFB12466
	.uleb128 .LEHE77-.LEHB77
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB78-.LFB12466
	.uleb128 .LEHE78-.LEHB78
	.uleb128 .L6106-.LFB12466
	.uleb128 0
	.uleb128 .LEHB79-.LFB12466
	.uleb128 .LEHE79-.LEHB79
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB80-.LFB12466
	.uleb128 .LEHE80-.LEHB80
	.uleb128 .L6107-.LFB12466
	.uleb128 0
	.uleb128 .LEHB81-.LFB12466
	.uleb128 .LEHE81-.LEHB81
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB82-.LFB12466
	.uleb128 .LEHE82-.LEHB82
	.uleb128 .L6108-.LFB12466
	.uleb128 0
	.uleb128 .LEHB83-.LFB12466
	.uleb128 .LEHE83-.LEHB83
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB84-.LFB12466
	.uleb128 .LEHE84-.LEHB84
	.uleb128 .L6109-.LFB12466
	.uleb128 0
	.uleb128 .LEHB85-.LFB12466
	.uleb128 .LEHE85-.LEHB85
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB86-.LFB12466
	.uleb128 .LEHE86-.LEHB86
	.uleb128 .L6110-.LFB12466
	.uleb128 0
	.uleb128 .LEHB87-.LFB12466
	.uleb128 .LEHE87-.LEHB87
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB88-.LFB12466
	.uleb128 .LEHE88-.LEHB88
	.uleb128 .L6111-.LFB12466
	.uleb128 0
	.uleb128 .LEHB89-.LFB12466
	.uleb128 .LEHE89-.LEHB89
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB90-.LFB12466
	.uleb128 .LEHE90-.LEHB90
	.uleb128 .L6112-.LFB12466
	.uleb128 0
	.uleb128 .LEHB91-.LFB12466
	.uleb128 .LEHE91-.LEHB91
	.uleb128 0
	.uleb128 0
.LLSDACSE12466:
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv,comdat
	.size	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv, .-_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv,comdat
	.align 2
	.p2align 4
	.weak	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv
	.type	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv, @function
_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv:
.LFB12340:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12340
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vxorps	%xmm1, %xmm1, %xmm1
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$264, %rsp
	.cfi_def_cfa_offset 320
	movq	(%rdi), %rcx
	movq	16(%rdi), %rsi
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	movq	8(%rdi), %rax
	vmovss	(%rcx), %xmm0
	movl	(%rax), %r8d
	leal	(%r8,%r8), %edx
	vcvtsi2ssl	%edx, %xmm1, %xmm2
	vdivss	%xmm2, %xmm0, %xmm3
	vcvttss2sil	%xmm3, %ebx
	cmpl	(%rsi), %ebx
	jg	.L6156
	movq	%rdi, %rbp
	movq	32(%rdi), %rdi
	movl	%ebx, %ecx
	movq	24(%rbp), %r9
	movq	(%rdi), %rdx
	leaq	_Z25native_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	movq	(%r9), %rsi
.LEHB92:
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	24(%rbp), %r11
	movq	32(%rbp), %r10
	movl	%ebx, %ecx
	movq	8(%rbp), %r8
	leaq	_Z22opt_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 104(%rsp)
	movq	(%r11), %rsi
	movq	(%r10), %rdx
	movl	(%r8), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r12
	movq	32(%rbp), %r13
	movl	%ebx, %ecx
	movq	24(%rbp), %r14
	leaq	_Z23avx2_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 8(%rsp)
	movq	0(%r13), %rdx
	movl	(%r12), %r8d
	leal	(%rbx,%rbx), %r13d
	movq	(%r14), %rsi
	leaq	.LC60(%rip), %r14
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	32(%rbp), %rax
	movq	8(%rbp), %r15
	leaq	_Z27parallel_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	movq	24(%rbp), %rcx
	vmovsd	%xmm0, 16(%rsp)
	movq	(%rax), %rdx
	movl	(%r15), %r8d
	leaq	144(%rsp), %r15
	movq	(%rcx), %rsi
	movl	%ebx, %ecx
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %rdi
	movq	32(%rbp), %rdx
	movl	%ebx, %ecx
	movq	24(%rbp), %rsi
	vmovsd	%xmm0, 24(%rsp)
	movl	(%rdi), %r8d
	movq	(%rdx), %rdx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	movq	(%rsi), %rsi
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r9
	movq	24(%rbp), %r10
	movl	%ebx, %ecx
	movq	32(%rbp), %r8
	leaq	_Z32parallel_avx2_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	movq	(%r10), %rsi
	movq	(%r8), %rdx
	movl	(%r9), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r11
	movq	32(%rbp), %r12
	movl	%ebx, %ecx
	movq	24(%rbp), %rbp
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi2EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	movq	(%r12), %rdx
	movl	(%r11), %r8d
	leaq	112(%rsp), %r12
	movq	0(%rbp), %rsi
	movslq	%ebx, %rbp
	salq	$3, %rbp
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r12, %rdi
	shrq	$10, %rbp
	vcvtsi2sdl	%r13d, %xmm5, %xmm9
	vmovsd	104(%rsp), %xmm14
	movl	$40, %esi
	movq	%r15, %r8
	movq	%r14, 192(%rsp)
	leaq	.LC62(%rip), %r14
	movabsq	$4434527798, %rcx
	vdivsd	8(%rsp), %xmm9, %xmm6
	movq	%r14, %rdx
	vdivsd	16(%rsp), %xmm9, %xmm7
	vdivsd	24(%rsp), %xmm9, %xmm10
	vdivsd	32(%rsp), %xmm9, %xmm11
	vdivsd	40(%rsp), %xmm9, %xmm12
	movl	%ebx, 160(%rsp)
	vdivsd	%xmm14, %xmm9, %xmm15
	movq	%rbp, 176(%rsp)
	movl	$2, 144(%rsp)
	vmovsd	%xmm0, 96(%rsp)
	vmovsd	%xmm6, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	vmovsd	%xmm10, 64(%rsp)
	vmovsd	%xmm11, 72(%rsp)
	vmovsd	%xmm12, 80(%rsp)
	vmovsd	%xmm14, 208(%rsp)
	vdivsd	%xmm0, %xmm9, %xmm13
	vmulsd	.LC61(%rip), %xmm15, %xmm1
	vmovsd	%xmm1, 224(%rsp)
	vmovsd	%xmm13, 88(%rsp)
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE92:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB93:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE93:
	movq	112(%rsp), %rdi
	leaq	128(%rsp), %r13
	cmpq	%r13, %rdi
	je	.L6122
	movq	128(%rsp), %rax
	leaq	1(%rax), %rsi
	call	_ZdlPvm@PLT
.L6122:
	vmovsd	8(%rsp), %xmm2
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm0
	movl	$40, %esi
	leaq	.LC63(%rip), %rcx
	movl	$2, 144(%rsp)
	vmulsd	48(%rsp), %xmm0, %xmm3
	movq	%rcx, 192(%rsp)
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm2, 208(%rsp)
	vmovsd	%xmm3, 224(%rsp)
.LEHB94:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE94:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB95:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE95:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6123
	movq	128(%rsp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
.L6123:
	vmovsd	.LC61(%rip), %xmm8
	movl	$40, %esi
	movq	%r14, %rdx
	movq	%r15, %r8
	vmulsd	56(%rsp), %xmm8, %xmm5
	vmovsd	16(%rsp), %xmm4
	leaq	.LC64(%rip), %rdi
	movabsq	$4434527798, %rcx
	movq	%rdi, 192(%rsp)
	movq	%r12, %rdi
	movl	$2, 144(%rsp)
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm4, 208(%rsp)
	vmovsd	%xmm5, 224(%rsp)
.LEHB96:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE96:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB97:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE97:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6124
	movq	128(%rsp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L6124:
	vmovsd	24(%rsp), %xmm9
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm6
	vmulsd	64(%rsp), %xmm6, %xmm7
	leaq	.LC65(%rip), %r9
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$2, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r9, 192(%rsp)
	vmovsd	%xmm9, 208(%rsp)
	vmovsd	%xmm7, 224(%rsp)
.LEHB98:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE98:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB99:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE99:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6125
	movq	128(%rsp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L6125:
	vmovsd	40(%rsp), %xmm10
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm11
	vmulsd	80(%rsp), %xmm11, %xmm12
	leaq	.LC66(%rip), %r10
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$2, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r10, 192(%rsp)
	vmovsd	%xmm10, 208(%rsp)
	vmovsd	%xmm12, 224(%rsp)
.LEHB100:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE100:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB101:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE101:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6126
	movq	128(%rsp), %r11
	leaq	1(%r11), %rsi
	call	_ZdlPvm@PLT
.L6126:
	vmovsd	32(%rsp), %xmm13
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm14
	vmulsd	72(%rsp), %xmm14, %xmm15
	leaq	.LC67(%rip), %rax
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$2, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%rax, 192(%rsp)
	vmovsd	%xmm13, 208(%rsp)
	vmovsd	%xmm15, 224(%rsp)
.LEHB102:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE102:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB103:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE103:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6127
	movq	128(%rsp), %rcx
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
.L6127:
	vmovsd	96(%rsp), %xmm1
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm2
	vmulsd	88(%rsp), %xmm2, %xmm0
	movl	%ebx, 160(%rsp)
	movl	$40, %esi
	leaq	.LC68(%rip), %rbx
	movq	%rbp, 176(%rsp)
	movabsq	$4434527798, %rcx
	movl	$2, 144(%rsp)
	movq	%rbx, 192(%rsp)
	vmovsd	%xmm1, 208(%rsp)
	vmovsd	%xmm0, 224(%rsp)
.LEHB104:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE104:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB105:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE105:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6119
	movq	128(%rsp), %r12
	leaq	1(%r12), %rsi
	call	_ZdlPvm@PLT
.L6119:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6154
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6156:
	.cfi_restore_state
	leaq	_ZSt4cout(%rip), %r15
	movl	$15, %edx
	leaq	.LC59(%rip), %rsi
	movq	%r15, %rdi
.LEHB106:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r15, %rdi
	movl	%ebx, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rdi
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6154
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L6141:
	.cfi_restore_state
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L6142
.L6154:
	call	__stack_chk_fail@PLT
.L6150:
	endbr64
.L6155:
	movq	%rax, %rbp
	jmp	.L6141
.L6149:
	endbr64
	jmp	.L6155
.L6148:
	endbr64
	jmp	.L6155
.L6147:
	endbr64
	jmp	.L6155
.L6145:
	endbr64
	jmp	.L6155
.L6144:
	endbr64
	jmp	.L6155
.L6146:
	endbr64
	jmp	.L6155
.L6142:
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE106:
	.cfi_endproc
.LFE12340:
	.section	.gcc_except_table
.LLSDA12340:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12340-.LLSDACSB12340
.LLSDACSB12340:
	.uleb128 .LEHB92-.LFB12340
	.uleb128 .LEHE92-.LEHB92
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB93-.LFB12340
	.uleb128 .LEHE93-.LEHB93
	.uleb128 .L6144-.LFB12340
	.uleb128 0
	.uleb128 .LEHB94-.LFB12340
	.uleb128 .LEHE94-.LEHB94
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB95-.LFB12340
	.uleb128 .LEHE95-.LEHB95
	.uleb128 .L6145-.LFB12340
	.uleb128 0
	.uleb128 .LEHB96-.LFB12340
	.uleb128 .LEHE96-.LEHB96
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB97-.LFB12340
	.uleb128 .LEHE97-.LEHB97
	.uleb128 .L6146-.LFB12340
	.uleb128 0
	.uleb128 .LEHB98-.LFB12340
	.uleb128 .LEHE98-.LEHB98
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB99-.LFB12340
	.uleb128 .LEHE99-.LEHB99
	.uleb128 .L6147-.LFB12340
	.uleb128 0
	.uleb128 .LEHB100-.LFB12340
	.uleb128 .LEHE100-.LEHB100
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB101-.LFB12340
	.uleb128 .LEHE101-.LEHB101
	.uleb128 .L6148-.LFB12340
	.uleb128 0
	.uleb128 .LEHB102-.LFB12340
	.uleb128 .LEHE102-.LEHB102
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB103-.LFB12340
	.uleb128 .LEHE103-.LEHB103
	.uleb128 .L6149-.LFB12340
	.uleb128 0
	.uleb128 .LEHB104-.LFB12340
	.uleb128 .LEHE104-.LEHB104
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB105-.LFB12340
	.uleb128 .LEHE105-.LEHB105
	.uleb128 .L6150-.LFB12340
	.uleb128 0
	.uleb128 .LEHB106-.LFB12340
	.uleb128 .LEHE106-.LEHB106
	.uleb128 0
	.uleb128 0
.LLSDACSE12340:
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv,comdat
	.size	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv, .-_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv,comdat
	.align 2
	.p2align 4
	.weak	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv
	.type	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv, @function
_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv:
.LFB12469:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12469
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vxorps	%xmm1, %xmm1, %xmm1
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$264, %rsp
	.cfi_def_cfa_offset 320
	movq	(%rdi), %rcx
	movq	16(%rdi), %rsi
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	movq	8(%rdi), %rax
	vmovss	(%rcx), %xmm0
	movl	(%rax), %r8d
	imull	$1000, %r8d, %edx
	vcvtsi2ssl	%edx, %xmm1, %xmm2
	vdivss	%xmm2, %xmm0, %xmm3
	vcvttss2sil	%xmm3, %ebx
	cmpl	(%rsi), %ebx
	jg	.L6194
	movq	%rdi, %rbp
	movq	32(%rdi), %rdi
	movl	%ebx, %ecx
	movq	24(%rbp), %r9
	movq	(%rdi), %rdx
	leaq	_Z25native_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	movq	(%r9), %rsi
.LEHB107:
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	24(%rbp), %r11
	movq	32(%rbp), %r10
	movl	%ebx, %ecx
	movq	8(%rbp), %r8
	leaq	_Z22opt_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 104(%rsp)
	movq	(%r11), %rsi
	movq	(%r10), %rdx
	movl	(%r8), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r12
	movq	32(%rbp), %r13
	movl	%ebx, %ecx
	movq	24(%rbp), %r14
	leaq	_Z23avx2_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 8(%rsp)
	movq	0(%r13), %rdx
	movl	(%r12), %r8d
	imull	$1000, %ebx, %r13d
	movq	(%r14), %rsi
	leaq	.LC60(%rip), %r14
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	32(%rbp), %rax
	movq	8(%rbp), %r15
	leaq	_Z27parallel_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	movq	24(%rbp), %rcx
	vmovsd	%xmm0, 16(%rsp)
	movq	(%rax), %rdx
	movl	(%r15), %r8d
	leaq	144(%rsp), %r15
	movq	(%rcx), %rsi
	movl	%ebx, %ecx
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %rdi
	movq	32(%rbp), %rdx
	movl	%ebx, %ecx
	movq	24(%rbp), %rsi
	vmovsd	%xmm0, 24(%rsp)
	movl	(%rdi), %r8d
	movq	(%rdx), %rdx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	movq	(%rsi), %rsi
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r9
	movq	24(%rbp), %r10
	movl	%ebx, %ecx
	movq	32(%rbp), %r8
	leaq	_Z32parallel_avx2_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	movq	(%r10), %rsi
	movq	(%r8), %rdx
	movl	(%r9), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r11
	movq	32(%rbp), %r12
	movl	%ebx, %ecx
	movq	24(%rbp), %rbp
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi1000EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	movq	(%r12), %rdx
	movl	(%r11), %r8d
	leaq	112(%rsp), %r12
	movq	0(%rbp), %rsi
	movslq	%ebx, %rbp
	salq	$3, %rbp
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r12, %rdi
	shrq	$10, %rbp
	vcvtsi2sdl	%r13d, %xmm5, %xmm9
	vmovsd	104(%rsp), %xmm14
	movl	$40, %esi
	movq	%r15, %r8
	movq	%r14, 192(%rsp)
	leaq	.LC62(%rip), %r14
	movabsq	$4434527798, %rcx
	vdivsd	8(%rsp), %xmm9, %xmm6
	movq	%r14, %rdx
	vdivsd	16(%rsp), %xmm9, %xmm7
	vdivsd	24(%rsp), %xmm9, %xmm10
	vdivsd	32(%rsp), %xmm9, %xmm11
	vdivsd	40(%rsp), %xmm9, %xmm12
	movl	%ebx, 160(%rsp)
	vdivsd	%xmm14, %xmm9, %xmm15
	movq	%rbp, 176(%rsp)
	movl	$1000, 144(%rsp)
	vmovsd	%xmm0, 96(%rsp)
	vmovsd	%xmm6, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	vmovsd	%xmm10, 64(%rsp)
	vmovsd	%xmm11, 72(%rsp)
	vmovsd	%xmm12, 80(%rsp)
	vmovsd	%xmm14, 208(%rsp)
	vdivsd	%xmm0, %xmm9, %xmm13
	vmulsd	.LC61(%rip), %xmm15, %xmm1
	vmovsd	%xmm1, 224(%rsp)
	vmovsd	%xmm13, 88(%rsp)
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE107:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB108:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE108:
	movq	112(%rsp), %rdi
	leaq	128(%rsp), %r13
	cmpq	%r13, %rdi
	je	.L6160
	movq	128(%rsp), %rax
	leaq	1(%rax), %rsi
	call	_ZdlPvm@PLT
.L6160:
	vmovsd	8(%rsp), %xmm2
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm0
	movl	$40, %esi
	leaq	.LC63(%rip), %rcx
	movl	$1000, 144(%rsp)
	vmulsd	48(%rsp), %xmm0, %xmm3
	movq	%rcx, 192(%rsp)
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm2, 208(%rsp)
	vmovsd	%xmm3, 224(%rsp)
.LEHB109:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE109:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB110:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE110:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6161
	movq	128(%rsp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
.L6161:
	vmovsd	.LC61(%rip), %xmm8
	movl	$40, %esi
	movq	%r14, %rdx
	movq	%r15, %r8
	vmulsd	56(%rsp), %xmm8, %xmm5
	vmovsd	16(%rsp), %xmm4
	leaq	.LC64(%rip), %rdi
	movabsq	$4434527798, %rcx
	movq	%rdi, 192(%rsp)
	movq	%r12, %rdi
	movl	$1000, 144(%rsp)
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm4, 208(%rsp)
	vmovsd	%xmm5, 224(%rsp)
.LEHB111:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE111:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB112:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE112:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6162
	movq	128(%rsp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L6162:
	vmovsd	24(%rsp), %xmm9
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm6
	vmulsd	64(%rsp), %xmm6, %xmm7
	leaq	.LC65(%rip), %r9
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$1000, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r9, 192(%rsp)
	vmovsd	%xmm9, 208(%rsp)
	vmovsd	%xmm7, 224(%rsp)
.LEHB113:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE113:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB114:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE114:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6163
	movq	128(%rsp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L6163:
	vmovsd	40(%rsp), %xmm10
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm11
	vmulsd	80(%rsp), %xmm11, %xmm12
	leaq	.LC66(%rip), %r10
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$1000, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r10, 192(%rsp)
	vmovsd	%xmm10, 208(%rsp)
	vmovsd	%xmm12, 224(%rsp)
.LEHB115:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE115:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB116:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE116:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6164
	movq	128(%rsp), %r11
	leaq	1(%r11), %rsi
	call	_ZdlPvm@PLT
.L6164:
	vmovsd	32(%rsp), %xmm13
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm14
	vmulsd	72(%rsp), %xmm14, %xmm15
	leaq	.LC67(%rip), %rax
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$1000, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%rax, 192(%rsp)
	vmovsd	%xmm13, 208(%rsp)
	vmovsd	%xmm15, 224(%rsp)
.LEHB117:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE117:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB118:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE118:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6165
	movq	128(%rsp), %rcx
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
.L6165:
	vmovsd	96(%rsp), %xmm1
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm2
	vmulsd	88(%rsp), %xmm2, %xmm0
	movl	%ebx, 160(%rsp)
	movl	$40, %esi
	leaq	.LC68(%rip), %rbx
	movq	%rbp, 176(%rsp)
	movabsq	$4434527798, %rcx
	movl	$1000, 144(%rsp)
	movq	%rbx, 192(%rsp)
	vmovsd	%xmm1, 208(%rsp)
	vmovsd	%xmm0, 224(%rsp)
.LEHB119:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE119:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB120:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE120:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6157
	movq	128(%rsp), %r12
	leaq	1(%r12), %rsi
	call	_ZdlPvm@PLT
.L6157:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6192
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6194:
	.cfi_restore_state
	leaq	_ZSt4cout(%rip), %r15
	movl	$15, %edx
	leaq	.LC59(%rip), %rsi
	movq	%r15, %rdi
.LEHB121:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r15, %rdi
	movl	%ebx, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rdi
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6192
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L6179:
	.cfi_restore_state
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L6180
.L6192:
	call	__stack_chk_fail@PLT
.L6188:
	endbr64
.L6193:
	movq	%rax, %rbp
	jmp	.L6179
.L6187:
	endbr64
	jmp	.L6193
.L6186:
	endbr64
	jmp	.L6193
.L6185:
	endbr64
	jmp	.L6193
.L6183:
	endbr64
	jmp	.L6193
.L6182:
	endbr64
	jmp	.L6193
.L6184:
	endbr64
	jmp	.L6193
.L6180:
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE121:
	.cfi_endproc
.LFE12469:
	.section	.gcc_except_table
.LLSDA12469:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12469-.LLSDACSB12469
.LLSDACSB12469:
	.uleb128 .LEHB107-.LFB12469
	.uleb128 .LEHE107-.LEHB107
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB108-.LFB12469
	.uleb128 .LEHE108-.LEHB108
	.uleb128 .L6182-.LFB12469
	.uleb128 0
	.uleb128 .LEHB109-.LFB12469
	.uleb128 .LEHE109-.LEHB109
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB110-.LFB12469
	.uleb128 .LEHE110-.LEHB110
	.uleb128 .L6183-.LFB12469
	.uleb128 0
	.uleb128 .LEHB111-.LFB12469
	.uleb128 .LEHE111-.LEHB111
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB112-.LFB12469
	.uleb128 .LEHE112-.LEHB112
	.uleb128 .L6184-.LFB12469
	.uleb128 0
	.uleb128 .LEHB113-.LFB12469
	.uleb128 .LEHE113-.LEHB113
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB114-.LFB12469
	.uleb128 .LEHE114-.LEHB114
	.uleb128 .L6185-.LFB12469
	.uleb128 0
	.uleb128 .LEHB115-.LFB12469
	.uleb128 .LEHE115-.LEHB115
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB116-.LFB12469
	.uleb128 .LEHE116-.LEHB116
	.uleb128 .L6186-.LFB12469
	.uleb128 0
	.uleb128 .LEHB117-.LFB12469
	.uleb128 .LEHE117-.LEHB117
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB118-.LFB12469
	.uleb128 .LEHE118-.LEHB118
	.uleb128 .L6187-.LFB12469
	.uleb128 0
	.uleb128 .LEHB119-.LFB12469
	.uleb128 .LEHE119-.LEHB119
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB120-.LFB12469
	.uleb128 .LEHE120-.LEHB120
	.uleb128 .L6188-.LFB12469
	.uleb128 0
	.uleb128 .LEHB121-.LFB12469
	.uleb128 .LEHE121-.LEHB121
	.uleb128 0
	.uleb128 0
.LLSDACSE12469:
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv,comdat
	.size	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv, .-_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv,comdat
	.align 2
	.p2align 4
	.weak	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv
	.type	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv, @function
_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv:
.LFB12468:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12468
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vxorps	%xmm1, %xmm1, %xmm1
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$264, %rsp
	.cfi_def_cfa_offset 320
	movq	(%rdi), %rcx
	movq	16(%rdi), %rsi
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	movq	8(%rdi), %rax
	vmovss	(%rcx), %xmm0
	movl	(%rax), %r8d
	imull	$100, %r8d, %edx
	vcvtsi2ssl	%edx, %xmm1, %xmm2
	vdivss	%xmm2, %xmm0, %xmm3
	vcvttss2sil	%xmm3, %ebx
	cmpl	(%rsi), %ebx
	jg	.L6232
	movq	%rdi, %rbp
	movq	32(%rdi), %rdi
	movl	%ebx, %ecx
	movq	24(%rbp), %r9
	movq	(%rdi), %rdx
	leaq	_Z25native_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	movq	(%r9), %rsi
.LEHB122:
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	24(%rbp), %r11
	movq	32(%rbp), %r10
	movl	%ebx, %ecx
	movq	8(%rbp), %r8
	leaq	_Z22opt_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 104(%rsp)
	movq	(%r11), %rsi
	movq	(%r10), %rdx
	movl	(%r8), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r12
	movq	32(%rbp), %r13
	movl	%ebx, %ecx
	movq	24(%rbp), %r14
	leaq	_Z23avx2_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 8(%rsp)
	movq	0(%r13), %rdx
	movl	(%r12), %r8d
	imull	$100, %ebx, %r13d
	movq	(%r14), %rsi
	leaq	.LC60(%rip), %r14
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	32(%rbp), %rax
	movq	8(%rbp), %r15
	leaq	_Z27parallel_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	movq	24(%rbp), %rcx
	vmovsd	%xmm0, 16(%rsp)
	movq	(%rax), %rdx
	movl	(%r15), %r8d
	leaq	144(%rsp), %r15
	movq	(%rcx), %rsi
	movl	%ebx, %ecx
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %rdi
	movq	32(%rbp), %rdx
	movl	%ebx, %ecx
	movq	24(%rbp), %rsi
	vmovsd	%xmm0, 24(%rsp)
	movl	(%rdi), %r8d
	movq	(%rdx), %rdx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	movq	(%rsi), %rsi
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r9
	movq	24(%rbp), %r10
	movl	%ebx, %ecx
	movq	32(%rbp), %r8
	leaq	_Z32parallel_avx2_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	movq	(%r10), %rsi
	movq	(%r8), %rdx
	movl	(%r9), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r11
	movq	32(%rbp), %r12
	movl	%ebx, %ecx
	movq	24(%rbp), %rbp
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi100EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	movq	(%r12), %rdx
	movl	(%r11), %r8d
	leaq	112(%rsp), %r12
	movq	0(%rbp), %rsi
	movslq	%ebx, %rbp
	salq	$3, %rbp
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r12, %rdi
	shrq	$10, %rbp
	vcvtsi2sdl	%r13d, %xmm5, %xmm9
	vmovsd	104(%rsp), %xmm14
	movl	$40, %esi
	movq	%r15, %r8
	movq	%r14, 192(%rsp)
	leaq	.LC62(%rip), %r14
	movabsq	$4434527798, %rcx
	vdivsd	8(%rsp), %xmm9, %xmm6
	movq	%r14, %rdx
	vdivsd	16(%rsp), %xmm9, %xmm7
	vdivsd	24(%rsp), %xmm9, %xmm10
	vdivsd	32(%rsp), %xmm9, %xmm11
	vdivsd	40(%rsp), %xmm9, %xmm12
	movl	%ebx, 160(%rsp)
	vdivsd	%xmm14, %xmm9, %xmm15
	movq	%rbp, 176(%rsp)
	movl	$100, 144(%rsp)
	vmovsd	%xmm0, 96(%rsp)
	vmovsd	%xmm6, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	vmovsd	%xmm10, 64(%rsp)
	vmovsd	%xmm11, 72(%rsp)
	vmovsd	%xmm12, 80(%rsp)
	vmovsd	%xmm14, 208(%rsp)
	vdivsd	%xmm0, %xmm9, %xmm13
	vmulsd	.LC61(%rip), %xmm15, %xmm1
	vmovsd	%xmm1, 224(%rsp)
	vmovsd	%xmm13, 88(%rsp)
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE122:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB123:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE123:
	movq	112(%rsp), %rdi
	leaq	128(%rsp), %r13
	cmpq	%r13, %rdi
	je	.L6198
	movq	128(%rsp), %rax
	leaq	1(%rax), %rsi
	call	_ZdlPvm@PLT
.L6198:
	vmovsd	8(%rsp), %xmm2
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm0
	movl	$40, %esi
	leaq	.LC63(%rip), %rcx
	movl	$100, 144(%rsp)
	vmulsd	48(%rsp), %xmm0, %xmm3
	movq	%rcx, 192(%rsp)
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm2, 208(%rsp)
	vmovsd	%xmm3, 224(%rsp)
.LEHB124:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE124:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB125:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE125:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6199
	movq	128(%rsp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
.L6199:
	vmovsd	.LC61(%rip), %xmm8
	movl	$40, %esi
	movq	%r14, %rdx
	movq	%r15, %r8
	vmulsd	56(%rsp), %xmm8, %xmm5
	vmovsd	16(%rsp), %xmm4
	leaq	.LC64(%rip), %rdi
	movabsq	$4434527798, %rcx
	movq	%rdi, 192(%rsp)
	movq	%r12, %rdi
	movl	$100, 144(%rsp)
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm4, 208(%rsp)
	vmovsd	%xmm5, 224(%rsp)
.LEHB126:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE126:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB127:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE127:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6200
	movq	128(%rsp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L6200:
	vmovsd	24(%rsp), %xmm9
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm6
	vmulsd	64(%rsp), %xmm6, %xmm7
	leaq	.LC65(%rip), %r9
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$100, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r9, 192(%rsp)
	vmovsd	%xmm9, 208(%rsp)
	vmovsd	%xmm7, 224(%rsp)
.LEHB128:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE128:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB129:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE129:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6201
	movq	128(%rsp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L6201:
	vmovsd	40(%rsp), %xmm10
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm11
	vmulsd	80(%rsp), %xmm11, %xmm12
	leaq	.LC66(%rip), %r10
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$100, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r10, 192(%rsp)
	vmovsd	%xmm10, 208(%rsp)
	vmovsd	%xmm12, 224(%rsp)
.LEHB130:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE130:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB131:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE131:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6202
	movq	128(%rsp), %r11
	leaq	1(%r11), %rsi
	call	_ZdlPvm@PLT
.L6202:
	vmovsd	32(%rsp), %xmm13
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm14
	vmulsd	72(%rsp), %xmm14, %xmm15
	leaq	.LC67(%rip), %rax
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$100, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%rax, 192(%rsp)
	vmovsd	%xmm13, 208(%rsp)
	vmovsd	%xmm15, 224(%rsp)
.LEHB132:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE132:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB133:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE133:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6203
	movq	128(%rsp), %rcx
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
.L6203:
	vmovsd	96(%rsp), %xmm1
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm2
	vmulsd	88(%rsp), %xmm2, %xmm0
	movl	%ebx, 160(%rsp)
	movl	$40, %esi
	leaq	.LC68(%rip), %rbx
	movq	%rbp, 176(%rsp)
	movabsq	$4434527798, %rcx
	movl	$100, 144(%rsp)
	movq	%rbx, 192(%rsp)
	vmovsd	%xmm1, 208(%rsp)
	vmovsd	%xmm0, 224(%rsp)
.LEHB134:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE134:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB135:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE135:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6195
	movq	128(%rsp), %r12
	leaq	1(%r12), %rsi
	call	_ZdlPvm@PLT
.L6195:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6230
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6232:
	.cfi_restore_state
	leaq	_ZSt4cout(%rip), %r15
	movl	$15, %edx
	leaq	.LC59(%rip), %rsi
	movq	%r15, %rdi
.LEHB136:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r15, %rdi
	movl	%ebx, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rdi
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6230
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L6217:
	.cfi_restore_state
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L6218
.L6230:
	call	__stack_chk_fail@PLT
.L6226:
	endbr64
.L6231:
	movq	%rax, %rbp
	jmp	.L6217
.L6225:
	endbr64
	jmp	.L6231
.L6224:
	endbr64
	jmp	.L6231
.L6223:
	endbr64
	jmp	.L6231
.L6221:
	endbr64
	jmp	.L6231
.L6220:
	endbr64
	jmp	.L6231
.L6222:
	endbr64
	jmp	.L6231
.L6218:
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE136:
	.cfi_endproc
.LFE12468:
	.section	.gcc_except_table
.LLSDA12468:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12468-.LLSDACSB12468
.LLSDACSB12468:
	.uleb128 .LEHB122-.LFB12468
	.uleb128 .LEHE122-.LEHB122
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB123-.LFB12468
	.uleb128 .LEHE123-.LEHB123
	.uleb128 .L6220-.LFB12468
	.uleb128 0
	.uleb128 .LEHB124-.LFB12468
	.uleb128 .LEHE124-.LEHB124
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB125-.LFB12468
	.uleb128 .LEHE125-.LEHB125
	.uleb128 .L6221-.LFB12468
	.uleb128 0
	.uleb128 .LEHB126-.LFB12468
	.uleb128 .LEHE126-.LEHB126
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB127-.LFB12468
	.uleb128 .LEHE127-.LEHB127
	.uleb128 .L6222-.LFB12468
	.uleb128 0
	.uleb128 .LEHB128-.LFB12468
	.uleb128 .LEHE128-.LEHB128
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB129-.LFB12468
	.uleb128 .LEHE129-.LEHB129
	.uleb128 .L6223-.LFB12468
	.uleb128 0
	.uleb128 .LEHB130-.LFB12468
	.uleb128 .LEHE130-.LEHB130
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB131-.LFB12468
	.uleb128 .LEHE131-.LEHB131
	.uleb128 .L6224-.LFB12468
	.uleb128 0
	.uleb128 .LEHB132-.LFB12468
	.uleb128 .LEHE132-.LEHB132
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB133-.LFB12468
	.uleb128 .LEHE133-.LEHB133
	.uleb128 .L6225-.LFB12468
	.uleb128 0
	.uleb128 .LEHB134-.LFB12468
	.uleb128 .LEHE134-.LEHB134
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB135-.LFB12468
	.uleb128 .LEHE135-.LEHB135
	.uleb128 .L6226-.LFB12468
	.uleb128 0
	.uleb128 .LEHB136-.LFB12468
	.uleb128 .LEHE136-.LEHB136
	.uleb128 0
	.uleb128 0
.LLSDACSE12468:
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv,comdat
	.size	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv, .-_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv,comdat
	.align 2
	.p2align 4
	.weak	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv
	.type	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv, @function
_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv:
.LFB12467:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12467
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vxorps	%xmm1, %xmm1, %xmm1
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$264, %rsp
	.cfi_def_cfa_offset 320
	movq	(%rdi), %rdx
	movq	16(%rdi), %rsi
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	movq	8(%rdi), %rax
	vmovss	(%rdx), %xmm0
	movl	(%rax), %r8d
	leal	(%r8,%r8,8), %ecx
	addl	%ecx, %ecx
	vcvtsi2ssl	%ecx, %xmm1, %xmm2
	vdivss	%xmm2, %xmm0, %xmm3
	vcvttss2sil	%xmm3, %ebx
	cmpl	(%rsi), %ebx
	jg	.L6270
	movq	%rdi, %rbp
	movq	32(%rdi), %rdi
	movl	%ebx, %ecx
	movq	24(%rbp), %r9
	movq	(%rdi), %rdx
	leaq	_Z25native_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	movq	(%r9), %rsi
.LEHB137:
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	24(%rbp), %r11
	movq	32(%rbp), %r10
	movl	%ebx, %ecx
	movq	8(%rbp), %r8
	leaq	_Z22opt_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 104(%rsp)
	movq	(%r11), %rsi
	movq	(%r10), %rdx
	movl	(%r8), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r12
	movq	32(%rbp), %r13
	movl	%ebx, %ecx
	movq	24(%rbp), %r14
	leaq	_Z23avx2_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 8(%rsp)
	movq	0(%r13), %rdx
	movl	(%r12), %r8d
	leal	(%rbx,%rbx,8), %r13d
	movq	(%r14), %rsi
	addl	%r13d, %r13d
	leaq	.LC60(%rip), %r14
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	32(%rbp), %rax
	movq	8(%rbp), %r15
	leaq	_Z27parallel_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	movq	24(%rbp), %rcx
	vmovsd	%xmm0, 16(%rsp)
	movq	(%rax), %rdx
	movl	(%r15), %r8d
	leaq	144(%rsp), %r15
	movq	(%rcx), %rsi
	movl	%ebx, %ecx
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %rdi
	movq	32(%rbp), %rdx
	movl	%ebx, %ecx
	movq	24(%rbp), %rsi
	vmovsd	%xmm0, 24(%rsp)
	movl	(%rdi), %r8d
	movq	(%rdx), %rdx
	leaq	_Z40dynamic_avx2_parallel_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	movq	(%rsi), %rsi
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r9
	movq	24(%rbp), %r10
	movl	%ebx, %ecx
	movq	32(%rbp), %r8
	leaq	_Z32parallel_avx2_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	movq	(%r10), %rsi
	movq	(%r8), %rdx
	movl	(%r9), %r8d
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	movq	8(%rbp), %r11
	movq	32(%rbp), %r12
	movl	%ebx, %ecx
	movq	24(%rbp), %rbp
	leaq	_Z39manual_avx2_parallel_cube_root_templateILi18EEvPKfPfi(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	movq	(%r12), %rdx
	movl	(%r11), %r8d
	leaq	112(%rsp), %r12
	movq	0(%rbp), %rsi
	movslq	%ebx, %rbp
	salq	$3, %rbp
	call	_Z9benchmarkPFvPKfPfiES0_S1_iii.constprop.0
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r12, %rdi
	shrq	$10, %rbp
	vcvtsi2sdl	%r13d, %xmm5, %xmm9
	vmovsd	104(%rsp), %xmm14
	movl	$40, %esi
	movq	%r15, %r8
	movq	%r14, 192(%rsp)
	leaq	.LC62(%rip), %r14
	movabsq	$4434527798, %rcx
	vdivsd	8(%rsp), %xmm9, %xmm6
	movq	%r14, %rdx
	vdivsd	16(%rsp), %xmm9, %xmm7
	vdivsd	24(%rsp), %xmm9, %xmm10
	vdivsd	32(%rsp), %xmm9, %xmm11
	vdivsd	40(%rsp), %xmm9, %xmm12
	movl	%ebx, 160(%rsp)
	vdivsd	%xmm14, %xmm9, %xmm15
	movq	%rbp, 176(%rsp)
	movl	$18, 144(%rsp)
	vmovsd	%xmm0, 96(%rsp)
	vmovsd	%xmm6, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	vmovsd	%xmm10, 64(%rsp)
	vmovsd	%xmm11, 72(%rsp)
	vmovsd	%xmm12, 80(%rsp)
	vmovsd	%xmm14, 208(%rsp)
	vdivsd	%xmm0, %xmm9, %xmm13
	vmulsd	.LC61(%rip), %xmm15, %xmm1
	vmovsd	%xmm1, 224(%rsp)
	vmovsd	%xmm13, 88(%rsp)
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE137:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB138:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE138:
	movq	112(%rsp), %rdi
	leaq	128(%rsp), %r13
	cmpq	%r13, %rdi
	je	.L6236
	movq	128(%rsp), %rax
	leaq	1(%rax), %rsi
	call	_ZdlPvm@PLT
.L6236:
	vmovsd	8(%rsp), %xmm2
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm0
	movl	$40, %esi
	leaq	.LC63(%rip), %rcx
	movl	$18, 144(%rsp)
	vmulsd	48(%rsp), %xmm0, %xmm3
	movq	%rcx, 192(%rsp)
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm2, 208(%rsp)
	vmovsd	%xmm3, 224(%rsp)
.LEHB139:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE139:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB140:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE140:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6237
	movq	128(%rsp), %rdx
	leaq	1(%rdx), %rsi
	call	_ZdlPvm@PLT
.L6237:
	vmovsd	.LC61(%rip), %xmm8
	movl	$40, %esi
	movq	%r14, %rdx
	movq	%r15, %r8
	vmulsd	56(%rsp), %xmm8, %xmm5
	vmovsd	16(%rsp), %xmm4
	leaq	.LC64(%rip), %rdi
	movabsq	$4434527798, %rcx
	movq	%rdi, 192(%rsp)
	movq	%r12, %rdi
	movl	$18, 144(%rsp)
	movl	%ebx, 160(%rsp)
	movq	%rbp, 176(%rsp)
	vmovsd	%xmm4, 208(%rsp)
	vmovsd	%xmm5, 224(%rsp)
.LEHB141:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE141:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB142:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE142:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6238
	movq	128(%rsp), %rsi
	leaq	1(%rsi), %rsi
	call	_ZdlPvm@PLT
.L6238:
	vmovsd	24(%rsp), %xmm9
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm6
	vmulsd	64(%rsp), %xmm6, %xmm7
	leaq	.LC65(%rip), %r9
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$18, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r9, 192(%rsp)
	vmovsd	%xmm9, 208(%rsp)
	vmovsd	%xmm7, 224(%rsp)
.LEHB143:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE143:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB144:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE144:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6239
	movq	128(%rsp), %r8
	leaq	1(%r8), %rsi
	call	_ZdlPvm@PLT
.L6239:
	vmovsd	40(%rsp), %xmm10
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm11
	vmulsd	80(%rsp), %xmm11, %xmm12
	leaq	.LC66(%rip), %r10
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$18, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%r10, 192(%rsp)
	vmovsd	%xmm10, 208(%rsp)
	vmovsd	%xmm12, 224(%rsp)
.LEHB145:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE145:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB146:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE146:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6240
	movq	128(%rsp), %r11
	leaq	1(%r11), %rsi
	call	_ZdlPvm@PLT
.L6240:
	vmovsd	32(%rsp), %xmm13
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm14
	vmulsd	72(%rsp), %xmm14, %xmm15
	leaq	.LC67(%rip), %rax
	movl	$40, %esi
	movabsq	$4434527798, %rcx
	movl	%ebx, 160(%rsp)
	movl	$18, 144(%rsp)
	movq	%rbp, 176(%rsp)
	movq	%rax, 192(%rsp)
	vmovsd	%xmm13, 208(%rsp)
	vmovsd	%xmm15, 224(%rsp)
.LEHB147:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE147:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB148:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE148:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6241
	movq	128(%rsp), %rcx
	leaq	1(%rcx), %rsi
	call	_ZdlPvm@PLT
.L6241:
	vmovsd	96(%rsp), %xmm1
	movq	%r14, %rdx
	movq	%r12, %rdi
	movq	%r15, %r8
	vmovsd	.LC61(%rip), %xmm2
	vmulsd	88(%rsp), %xmm2, %xmm0
	movl	%ebx, 160(%rsp)
	movl	$40, %esi
	leaq	.LC68(%rip), %rbx
	movq	%rbp, 176(%rsp)
	movabsq	$4434527798, %rcx
	movl	$18, 144(%rsp)
	movq	%rbx, 192(%rsp)
	vmovsd	%xmm1, 208(%rsp)
	vmovsd	%xmm0, 224(%rsp)
.LEHB149:
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE149:
	movq	120(%rsp), %rdx
	movq	112(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB150:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE150:
	movq	112(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.L6233
	movq	128(%rsp), %r12
	leaq	1(%r12), %rsi
	call	_ZdlPvm@PLT
.L6233:
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6268
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6270:
	.cfi_restore_state
	leaq	_ZSt4cout(%rip), %r15
	movl	$15, %edx
	leaq	.LC59(%rip), %rsi
	movq	%r15, %rdi
.LEHB151:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r15, %rdi
	movl	%ebx, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rdi
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6268
	addq	$264, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L6255:
	.cfi_restore_state
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	je	.L6256
.L6268:
	call	__stack_chk_fail@PLT
.L6264:
	endbr64
.L6269:
	movq	%rax, %rbp
	jmp	.L6255
.L6263:
	endbr64
	jmp	.L6269
.L6262:
	endbr64
	jmp	.L6269
.L6261:
	endbr64
	jmp	.L6269
.L6259:
	endbr64
	jmp	.L6269
.L6258:
	endbr64
	jmp	.L6269
.L6260:
	endbr64
	jmp	.L6269
.L6256:
	movq	%rbp, %rdi
	call	_Unwind_Resume@PLT
.LEHE151:
	.cfi_endproc
.LFE12467:
	.section	.gcc_except_table
.LLSDA12467:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12467-.LLSDACSB12467
.LLSDACSB12467:
	.uleb128 .LEHB137-.LFB12467
	.uleb128 .LEHE137-.LEHB137
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB138-.LFB12467
	.uleb128 .LEHE138-.LEHB138
	.uleb128 .L6258-.LFB12467
	.uleb128 0
	.uleb128 .LEHB139-.LFB12467
	.uleb128 .LEHE139-.LEHB139
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB140-.LFB12467
	.uleb128 .LEHE140-.LEHB140
	.uleb128 .L6259-.LFB12467
	.uleb128 0
	.uleb128 .LEHB141-.LFB12467
	.uleb128 .LEHE141-.LEHB141
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB142-.LFB12467
	.uleb128 .LEHE142-.LEHB142
	.uleb128 .L6260-.LFB12467
	.uleb128 0
	.uleb128 .LEHB143-.LFB12467
	.uleb128 .LEHE143-.LEHB143
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB144-.LFB12467
	.uleb128 .LEHE144-.LEHB144
	.uleb128 .L6261-.LFB12467
	.uleb128 0
	.uleb128 .LEHB145-.LFB12467
	.uleb128 .LEHE145-.LEHB145
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB146-.LFB12467
	.uleb128 .LEHE146-.LEHB146
	.uleb128 .L6262-.LFB12467
	.uleb128 0
	.uleb128 .LEHB147-.LFB12467
	.uleb128 .LEHE147-.LEHB147
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB148-.LFB12467
	.uleb128 .LEHE148-.LEHB148
	.uleb128 .L6263-.LFB12467
	.uleb128 0
	.uleb128 .LEHB149-.LFB12467
	.uleb128 .LEHE149-.LEHB149
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB150-.LFB12467
	.uleb128 .LEHE150-.LEHB150
	.uleb128 .L6264-.LFB12467
	.uleb128 0
	.uleb128 .LEHB151-.LFB12467
	.uleb128 .LEHE151-.LEHB151
	.uleb128 0
	.uleb128 0
.LLSDACSE12467:
	.section	.text._ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv,"axG",@progbits,_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv,comdat
	.size	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv, .-_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv
	.section	.rodata.str1.1
.LC71:
	.string	"newton"
.LC72:
	.string	"size"
.LC73:
	.string	"KB"
.LC74:
	.string	"algo"
.LC75:
	.string	"ms"
.LC76:
	.string	"MIter/s"
	.section	.rodata.str1.8
	.align 8
.LC77:
	.string	"{:>8} {:>8} {:>6} {:>20} {:>10} {:>10}"
	.section	.text.unlikely
.LCOLDB79:
	.section	.text.startup,"ax",@progbits
.LHOTB79:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB11763:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA11763
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$4194304, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$576, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movq	%fs:40, %rax
	movq	%rax, 568(%rsp)
	xorl	%eax, %eax
.LEHB152:
	call	_Znam@PLT
	movl	$4194304, %edi
	movq	%rax, %rbx
	call	_Znam@PLT
	movl	$8, %ecx
	vmovdqa	.LC69(%rip), %ymm1
	leaq	4194304(%rbx), %rdx
	vmovd	%ecx, %xmm0
	movq	%rax, %r13
	movq	%rbx, %rax
	vpbroadcastd	%xmm0, %ymm3
.L6272:
	vpaddd	%ymm3, %ymm1, %ymm2
	addq	$256, %rax
	vcvtdq2ps	%ymm1, %ymm1
	vmovups	%ymm1, -256(%rax)
	vpaddd	%ymm3, %ymm2, %ymm4
	vcvtdq2ps	%ymm2, %ymm5
	vmovups	%ymm5, -224(%rax)
	vpaddd	%ymm3, %ymm4, %ymm6
	vcvtdq2ps	%ymm4, %ymm7
	vmovups	%ymm7, -192(%rax)
	vpaddd	%ymm3, %ymm6, %ymm8
	vcvtdq2ps	%ymm6, %ymm9
	vmovups	%ymm9, -160(%rax)
	vpaddd	%ymm3, %ymm8, %ymm10
	vcvtdq2ps	%ymm8, %ymm11
	vmovups	%ymm11, -128(%rax)
	vpaddd	%ymm3, %ymm10, %ymm12
	vcvtdq2ps	%ymm10, %ymm13
	vmovups	%ymm13, -96(%rax)
	vpaddd	%ymm3, %ymm12, %ymm14
	vcvtdq2ps	%ymm12, %ymm15
	vmovups	%ymm15, -64(%rax)
	vcvtdq2ps	%ymm14, %ymm0
	vpaddd	%ymm3, %ymm14, %ymm1
	vmovups	%ymm0, -32(%rax)
	cmpq	%rax, %rdx
	jne	.L6272
	leaq	.LC71(%rip), %rsi
	leaq	.LC72(%rip), %rdi
	movabsq	$5541893286, %rcx
	leaq	.LC73(%rip), %r8
	leaq	.LC74(%rip), %r9
	movq	%rsi, 464(%rsp)
	movl	$38, %esi
	leaq	.LC75(%rip), %r10
	leaq	.LC76(%rip), %r11
	movq	%rdi, 480(%rsp)
	movq	%r8, 496(%rsp)
	leaq	432(%rsp), %r12
	leaq	.LC77(%rip), %rdx
	movq	%r9, 512(%rsp)
	movq	%r12, %rdi
	leaq	464(%rsp), %r8
	movq	%r10, 528(%rsp)
	movq	%r11, 544(%rsp)
	vzeroupper
	call	_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE
.LEHE152:
	movq	440(%rsp), %rdx
	movq	432(%rsp), %rsi
	leaq	_ZSt4cout(%rip), %rdi
.LEHB153:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE153:
	leaq	44(%rsp), %rax
	leaq	36(%rsp), %rdx
	movq	%r12, %rdi
	vmovq	%rax, %xmm3
	vmovq	%rdx, %xmm2
	leaq	40(%rsp), %rcx
	leaq	48(%rsp), %rsi
	vpinsrq	$1, %rcx, %xmm3, %xmm5
	leaq	56(%rsp), %r12
	vpinsrq	$1, %rsi, %xmm2, %xmm4
	vinserti128	$0x1, %xmm4, %ymm5, %ymm6
	vmovdqa	%ymm6, (%rsp)
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	vmovdqa	(%rsp), %ymm7
	movq	%rbx, 48(%rsp)
	movq	%r13, 56(%rsp)
	leaq	384(%rsp), %rdi
	movl	$1048576, 36(%rsp)
	movl	$100, 40(%rsp)
	movl	$0x4e6e6b28, 44(%rsp)
	movq	%r12, 416(%rsp)
	vmovdqa	%ymm7, 384(%rsp)
	vzeroupper
.LEHB154:
	call	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE_clEv
	vmovdqa	(%rsp), %ymm8
	movq	%r12, 352(%rsp)
	leaq	320(%rsp), %rdi
	vmovdqa	%ymm8, 320(%rsp)
	vzeroupper
	call	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE0_clEv
	vmovdqa	(%rsp), %ymm9
	movq	%r12, 288(%rsp)
	leaq	256(%rsp), %rdi
	vmovdqa	%ymm9, 256(%rsp)
	vzeroupper
	call	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE1_clEv
	vmovdqa	(%rsp), %ymm10
	movq	%r12, 224(%rsp)
	leaq	192(%rsp), %rdi
	vmovdqa	%ymm10, 192(%rsp)
	vzeroupper
	call	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE2_clEv
	vmovdqa	(%rsp), %ymm11
	movq	%r12, 160(%rsp)
	leaq	128(%rsp), %rdi
	vmovdqa	%ymm11, 128(%rsp)
	vzeroupper
	call	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE3_clEv
	vmovdqa	(%rsp), %ymm12
	movq	%r12, 96(%rsp)
	leaq	64(%rsp), %rdi
	vmovdqa	%ymm12, 64(%rsp)
	vzeroupper
	call	_ZZ15run_all_configsIJLi2ELi16ELi18ELi100ELi1000ELi10000EEEvPKfPfiifENKUlvE4_clEv
.LEHE154:
	movq	%rbx, %rdi
	call	_ZdaPv@PLT
	movq	%r13, %rdi
	call	_ZdaPv@PLT
	movq	568(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6287
	leaq	-24(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L6287:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
.L6276:
	endbr64
	movq	%rax, %rbx
	jmp	.L6273
	.section	.gcc_except_table
.LLSDA11763:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11763-.LLSDACSB11763
.LLSDACSB11763:
	.uleb128 .LEHB152-.LFB11763
	.uleb128 .LEHE152-.LEHB152
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB153-.LFB11763
	.uleb128 .LEHE153-.LEHB153
	.uleb128 .L6276-.LFB11763
	.uleb128 0
	.uleb128 .LEHB154-.LFB11763
	.uleb128 .LEHE154-.LEHB154
	.uleb128 0
	.uleb128 0
.LLSDACSE11763:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC11763
	.type	main.cold, @function
main.cold:
.LFSB11763:
.L6273:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -40
	.cfi_offset 6, -16
	.cfi_offset 12, -32
	.cfi_offset 13, -24
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	568(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L6288
	movq	%rbx, %rdi
.LEHB155:
	call	_Unwind_Resume@PLT
.LEHE155:
.L6288:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE11763:
	.section	.gcc_except_table
.LLSDAC11763:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC11763-.LLSDACSBC11763
.LLSDACSBC11763:
	.uleb128 .LEHB155-.LCOLDB79
	.uleb128 .LEHE155-.LEHB155
	.uleb128 0
	.uleb128 0
.LLSDACSEC11763:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE79:
	.section	.text.startup
.LHOTE79:
	.section	.text._ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,"axG",@progbits,_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm
	.type	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm, @function
_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm:
.LFB13440:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA13440
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$456, %rsp
	movq	48(%rdi), %r12
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movzbl	(%r12), %eax
	movl	%eax, %edx
	andl	$15, %eax
	andl	$15, %edx
	cmpq	%rax, %rsi
	jb	.L6725
	testb	%dl, %dl
	jne	.L6292
	movq	(%r12), %rcx
	shrq	$4, %rcx
	cmpq	%rcx, %rsi
	jnb	.L6292
	salq	$5, %rsi
	addq	8(%r12), %rsi
	vmovdqu	(%rsi), %xmm3
	movzbl	16(%rsi), %edi
	vmovdqa	%xmm3, -448(%rbp)
	jmp	.L6291
	.p2align 4,,10
	.p2align 3
.L6725:
	movq	(%r12), %rdi
	leaq	(%rsi,%rsi,4), %rcx
	salq	$4, %rsi
	addq	8(%r12), %rsi
	vmovdqa	(%rsi), %xmm2
	shrq	$4, %rdi
	shrq	%cl, %rdi
	vmovdqa	%xmm2, -448(%rbp)
	andl	$31, %edi
.L6291:
	leaq	.L6295(%rip), %r8
	movzbl	%dil, %esi
	movb	%dil, -432(%rbp)
	vmovdqu	-448(%rbp), %ymm1
	movslq	(%r8,%rsi,4), %r9
	vmovdqu	%ymm1, -416(%rbp)
	addq	%r8, %r9
	notrack jmp	*%r9
	.section	.rodata._ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,"aG",@progbits,_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,comdat
	.align 4
	.align 4
.L6295:
	.long	.L6713-.L6295
	.long	.L6310-.L6295
	.long	.L6309-.L6295
	.long	.L6308-.L6295
	.long	.L6307-.L6295
	.long	.L6306-.L6295
	.long	.L6305-.L6295
	.long	.L6304-.L6295
	.long	.L6303-.L6295
	.long	.L6302-.L6295
	.long	.L6301-.L6295
	.long	.L6300-.L6295
	.long	.L6299-.L6295
	.long	.L6298-.L6295
	.long	.L6297-.L6295
	.long	.L6296-.L6295
	.long	.L6294-.L6295
	.long	.L6294-.L6295
	.long	.L6294-.L6295
	.long	.L6294-.L6295
	.long	.L6294-.L6295
	.section	.text._ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,"axG",@progbits,_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,comdat
	.p2align 4,,10
	.p2align 3
.L6296:
	leaq	-456(%rbp), %r12
	leaq	8(%rbx), %rsi
	movl	$1, %edx
	movabsq	$9007199254740992, %r10
	movq	%r10, -456(%rbp)
	movq	%r12, %rdi
	vzeroupper
.LEHB156:
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	-416(%rbp), %rsi
	movq	%r12, %rdi
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movq	-408(%rbp), %rdx
	movq	%rbx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE6formatIoNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
.L6721:
	movq	%rax, 16(%rbx)
.L6289:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L6297:
	.cfi_restore_state
	leaq	-456(%rbp), %r13
	leaq	8(%rbx), %rsi
	movl	$1, %edx
	movabsq	$9007199254740992, %r11
	movq	%r11, -456(%rbp)
	movq	%r13, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	48(%rbx), %r14
	movq	%r13, %rdi
	movq	%rax, 8(%rbx)
	movq	-416(%rbp), %rsi
	movq	-408(%rbp), %rdx
	movq	%r14, %rcx
	call	_ZNKSt8__format15__formatter_intIcE6formatInNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%r14)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6298:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6726
	movq	-416(%rbp), %rdx
	movq	-408(%rbp), %rax
	vzeroupper
	leaq	-40(%rbp), %rsp
	leaq	8(%rbx), %rdi
	movq	%r12, %rsi
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	jmp	*%rax
	.p2align 4,,10
	.p2align 3
.L6299:
	.cfi_restore_state
	movq	16(%rbx), %r13
	movq	8(%rbx), %rsi
	movabsq	$9007199254740992, %r15
	movq	$0, -464(%rbp)
	movq	%r15, -456(%rbp)
	cmpq	%rsi, %r13
	je	.L6412
	cmpb	$125, (%rsi)
	je	.L6412
	leaq	-456(%rbp), %r14
	movq	%r13, %rdx
	vzeroupper
	movq	%r14, %rdi
	call	_ZNSt8__format5_SpecIcE23_M_parse_fill_and_alignEPKcS3_
	movq	%rax, %rsi
	cmpq	%rax, %r13
	je	.L6413
	cmpb	$125, (%rax)
	je	.L6413
	leaq	8(%rbx), %rcx
	movq	%r13, %rdx
	movq	%r14, %rdi
	call	_ZNSt8__format5_SpecIcE14_M_parse_widthEPKcS3_RSt26basic_format_parse_contextIcE
	movq	%rax, %rsi
	cmpq	%rax, %r13
	je	.L6417
	movzbl	(%rax), %edi
	cmpb	$112, %dil
	je	.L6727
.L6418:
	cmpb	$125, %dil
	jne	.L6728
.L6417:
	movl	-456(%rbp), %ecx
	movl	-453(%rbp), %r8d
	movq	48(%rbx), %r12
	movl	%ecx, -464(%rbp)
	movl	%r8d, -461(%rbp)
.L6415:
	movq	%rsi, 8(%rbx)
	movq	-416(%rbp), %rbx
	testq	%rbx, %rbx
	jne	.L6421
	movb	$48, -270(%rbp)
	movl	$3, %edi
.L6422:
	movl	$30768, %r9d
	leaq	-272(%rbp), %rsi
	movq	%r12, %rcx
	movq	%rdi, %rdx
	movw	%r9w, -272(%rbp)
	leaq	-464(%rbp), %r8
	movl	$2, %r9d
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE
.L6720:
	movq	%rax, 16(%r12)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6300:
	movabsq	$9007199254740992, %rdi
	leaq	-456(%rbp), %r12
	leaq	8(%rbx), %rsi
	movq	%rdi, -456(%rbp)
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE
	movq	48(%rbx), %r13
	movq	%r12, %rdi
	movq	%rax, 8(%rbx)
	movq	-416(%rbp), %rsi
	movq	-408(%rbp), %rdx
	movq	%r13, %rcx
	call	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE
	movq	%rax, 16(%r13)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6301:
	movabsq	$9007199254740992, %r10
	leaq	-456(%rbp), %r14
	leaq	8(%rbx), %rsi
	movq	%r10, -456(%rbp)
	movq	%r14, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_strIcE5parseERSt26basic_format_parse_contextIcE
	movq	-416(%rbp), %r15
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movq	%r15, %rdi
	call	strlen@PLT
	movq	%rbx, %rcx
	movq	%r15, %rdx
	movq	%r14, %rdi
	movq	%rax, %rsi
	call	_ZNKSt8__format15__formatter_strIcE6formatINS_10_Sink_iterIcEEEET_St17basic_string_viewIcSt11char_traitsIcEERSt20basic_format_contextIS5_cE
	movq	%rax, 16(%rbx)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6302:
	movabsq	$9007199254740992, %r11
	leaq	-456(%rbp), %r12
	leaq	8(%rbx), %rsi
	movq	%r11, -456(%rbp)
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	movq	48(%rbx), %r13
	movq	%r12, %rdi
	movq	%rax, 8(%rbx)
	pushq	-408(%rbp)
	movq	%r13, %rsi
	pushq	-416(%rbp)
	call	_ZNKSt8__format14__formatter_fpIcE6formatIeNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%r13)
	popq	%rdx
	popq	%rcx
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6303:
	movabsq	$9007199254740992, %rax
	leaq	-456(%rbp), %r14
	leaq	8(%rbx), %rsi
	movq	%rax, -456(%rbp)
	movq	%r14, %rdi
	vzeroupper
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	vmovsd	-416(%rbp), %xmm0
	movq	%r14, %rdi
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movq	%rbx, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIdNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6304:
	movabsq	$9007199254740992, %rsi
	leaq	-456(%rbp), %r15
	movq	%rsi, -456(%rbp)
	movq	%r15, %rdi
	leaq	8(%rbx), %rsi
	vzeroupper
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	movq	48(%rbx), %r12
	movq	%r15, %rdi
	movq	%rax, 8(%rbx)
	vmovss	-416(%rbp), %xmm0
	movq	%r12, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIfNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%r12)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6305:
	movabsq	$9007199254740992, %rdx
	leaq	-456(%rbp), %r13
	leaq	8(%rbx), %rsi
	movq	%rdx, -456(%rbp)
	movq	%r13, %rdi
	movl	$1, %edx
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	48(%rbx), %r14
	movq	%r13, %rdi
	movq	%rax, 8(%rbx)
	movq	-416(%rbp), %rsi
	movq	%r14, %rdx
	call	_ZNKSt8__format15__formatter_intIcE6formatIyNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%r14)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6306:
	leaq	-456(%rbp), %r15
	leaq	8(%rbx), %rsi
	movl	$1, %edx
	movabsq	$9007199254740992, %rcx
	movq	%rcx, -456(%rbp)
	movq	%r15, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movq	-416(%rbp), %rsi
	movq	%r15, %rdi
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movq	%rbx, %rdx
	call	_ZNKSt8__format15__formatter_intIcE6formatIxNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6307:
	leaq	-456(%rbp), %r12
	leaq	8(%rbx), %rsi
	movl	$1, %edx
	movabsq	$9007199254740992, %r8
	movq	%r8, -456(%rbp)
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movzbl	-455(%rbp), %r9d
	movl	-416(%rbp), %edx
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movl	%r9d, %r13d
	andl	$120, %r13d
	cmpb	$56, %r13b
	je	.L6729
	shrb	$3, %r9b
	andl	$15, %r9d
	cmpb	$4, %r9b
	je	.L6387
	ja	.L6388
	cmpb	$1, %r9b
	jbe	.L6389
	cmpb	$16, %r13b
	leaq	.LC35(%rip), %r11
	leaq	.LC36(%rip), %rax
	cmovne	%rax, %r11
	testl	%edx, %edx
	jne	.L6730
	movl	$48, %edx
	leaq	-268(%rbp), %rsi
	leaq	-269(%rbp), %r15
.L6394:
	movzbl	-456(%rbp), %r14d
	movb	%dl, -269(%rbp)
	testb	$16, %r14b
	je	.L6719
.L6453:
	movq	$-2, %rdx
	movl	$2, %ecx
.L6398:
	addq	%r15, %rdx
	movl	%ecx, %r8d
	testl	%ecx, %ecx
	je	.L6399
	xorl	%r13d, %r13d
	leal	-1(%rcx), %edi
	movl	$1, %ecx
	movzbl	(%r11,%r13), %r9d
	andl	$7, %edi
	movb	%r9b, (%rdx,%r13)
	cmpl	%r8d, %ecx
	jnb	.L6399
	testl	%edi, %edi
	je	.L6408
	cmpl	$1, %edi
	je	.L6642
	cmpl	$2, %edi
	je	.L6643
	cmpl	$3, %edi
	je	.L6644
	cmpl	$4, %edi
	je	.L6645
	cmpl	$5, %edi
	je	.L6646
	cmpl	$6, %edi
	je	.L6647
	movl	$1, %r10d
	movl	$2, %ecx
	movzbl	(%r11,%r10), %eax
	movb	%al, (%rdx,%r10)
.L6647:
	movl	%ecx, %edi
	addl	$1, %ecx
	movzbl	(%r11,%rdi), %r13d
	movb	%r13b, (%rdx,%rdi)
.L6646:
	movl	%ecx, %r10d
	addl	$1, %ecx
	movzbl	(%r11,%r10), %r9d
	movb	%r9b, (%rdx,%r10)
.L6645:
	movl	%ecx, %eax
	addl	$1, %ecx
	movzbl	(%r11,%rax), %edi
	movb	%dil, (%rdx,%rax)
.L6644:
	movl	%ecx, %r13d
	addl	$1, %ecx
	movzbl	(%r11,%r13), %r10d
	movb	%r10b, (%rdx,%r13)
.L6643:
	movl	%ecx, %eax
	addl	$1, %ecx
	movzbl	(%r11,%rax), %r9d
	movb	%r9b, (%rdx,%rax)
.L6642:
	movl	%ecx, %edi
	addl	$1, %ecx
	movzbl	(%r11,%rdi), %r13d
	movb	%r13b, (%rdx,%rdi)
	cmpl	%r8d, %ecx
	jnb	.L6399
.L6408:
	movl	%ecx, %r10d
	leal	1(%rcx), %edi
	leal	2(%rcx), %r13d
	movzbl	(%r11,%r10), %eax
	movzbl	(%r11,%rdi), %r9d
	movb	%al, (%rdx,%r10)
	leal	3(%rcx), %eax
	movzbl	(%r11,%r13), %r10d
	movb	%r9b, (%rdx,%rdi)
	movzbl	(%r11,%rax), %edi
	movb	%r10b, (%rdx,%r13)
	leal	4(%rcx), %r13d
	leal	5(%rcx), %r10d
	movb	%dil, (%rdx,%rax)
	movzbl	(%r11,%r13), %r9d
	leal	6(%rcx), %edi
	movzbl	(%r11,%r10), %eax
	movb	%r9b, (%rdx,%r13)
	movzbl	(%r11,%rdi), %r13d
	movb	%al, (%rdx,%r10)
	leal	7(%rcx), %r10d
	addl	$8, %ecx
	movzbl	(%r11,%r10), %r9d
	movb	%r13b, (%rdx,%rdi)
	movb	%r9b, (%rdx,%r10)
	cmpl	%r8d, %ecx
	jb	.L6408
	.p2align 4,,10
	.p2align 3
.L6399:
	shrb	$2, %r14b
	andl	$3, %r14d
	cmpl	$1, %r14d
	je	.L6455
	cmpl	$3, %r14d
	je	.L6731
.L6411:
	movq	%r15, %rcx
	subq	%rdx, %rsi
	movq	%rbx, %r8
	movq	%r12, %rdi
	subq	%rdx, %rcx
	call	_ZNKSt8__format15__formatter_intIcE13_M_format_intINS_10_Sink_iterIcEEEENSt20basic_format_contextIT_cE8iteratorESt17basic_string_viewIcSt11char_traitsIcEEmRS7_
	jmp	.L6721
	.p2align 4,,10
	.p2align 3
.L6308:
	leaq	-464(%rbp), %r12
	leaq	8(%rbx), %rsi
	movl	$1, %edx
	movabsq	$9007199254740992, %r15
	movq	%r15, -464(%rbp)
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movzbl	-463(%rbp), %esi
	movl	-416(%rbp), %r13d
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movl	%esi, %r14d
	andl	$120, %r14d
	cmpb	$56, %r14b
	je	.L6732
	shrb	$3, %sil
	movl	%r13d, %edx
	andl	$15, %esi
	testl	%r13d, %r13d
	js	.L6733
	cmpb	$4, %sil
	je	.L6349
	ja	.L6350
	cmpb	$1, %sil
	jbe	.L6351
	cmpb	$16, %r14b
	leaq	.LC35(%rip), %r11
	leaq	.LC36(%rip), %rdi
	cmovne	%rdi, %r11
	testl	%r13d, %r13d
	jne	.L6428
	movl	$48, %r10d
	leaq	-316(%rbp), %rsi
	leaq	-317(%rbp), %r15
.L6356:
	movzbl	-464(%rbp), %r8d
	movb	%r10b, -317(%rbp)
	testb	$16, %r8b
	je	.L6446
.L6445:
	movq	$-2, %rdx
	movl	$2, %eax
.L6360:
	addq	%r15, %rdx
	movl	%eax, %ecx
	testl	%eax, %eax
	je	.L6361
	xorl	%r9d, %r9d
	leal	-1(%rax), %edi
	movzbl	(%r11,%r9), %r10d
	andl	$7, %edi
	movb	%r10b, (%rdx,%r9)
	movl	$1, %r9d
	cmpl	%eax, %r9d
	jnb	.L6361
	testl	%edi, %edi
	je	.L6379
	cmpl	$1, %edi
	je	.L6636
	cmpl	$2, %edi
	je	.L6637
	cmpl	$3, %edi
	je	.L6638
	cmpl	$4, %edi
	je	.L6639
	cmpl	$5, %edi
	je	.L6640
	cmpl	$6, %edi
	je	.L6641
	movl	$1, %r14d
	movl	$2, %r9d
	movzbl	(%r11,%r14), %eax
	movb	%al, (%rdx,%r14)
.L6641:
	movl	%r9d, %edi
	addl	$1, %r9d
	movzbl	(%r11,%rdi), %r10d
	movb	%r10b, (%rdx,%rdi)
.L6640:
	movl	%r9d, %r14d
	addl	$1, %r9d
	movzbl	(%r11,%r14), %eax
	movb	%al, (%rdx,%r14)
.L6639:
	movl	%r9d, %edi
	addl	$1, %r9d
	movzbl	(%r11,%rdi), %r10d
	movb	%r10b, (%rdx,%rdi)
.L6638:
	movl	%r9d, %r14d
	addl	$1, %r9d
	movzbl	(%r11,%r14), %eax
	movb	%al, (%rdx,%r14)
.L6637:
	movl	%r9d, %edi
	addl	$1, %r9d
	movzbl	(%r11,%rdi), %r10d
	movb	%r10b, (%rdx,%rdi)
.L6636:
	movl	%r9d, %r14d
	addl	$1, %r9d
	movzbl	(%r11,%r14), %eax
	movb	%al, (%rdx,%r14)
	cmpl	%ecx, %r9d
	jnb	.L6361
.L6379:
	movl	%r9d, %edi
	leal	1(%r9), %r14d
	movzbl	(%r11,%rdi), %r10d
	movzbl	(%r11,%r14), %eax
	movb	%r10b, (%rdx,%rdi)
	leal	2(%r9), %edi
	movb	%al, (%rdx,%r14)
	leal	3(%r9), %r14d
	movzbl	(%r11,%rdi), %r10d
	movzbl	(%r11,%r14), %eax
	movb	%r10b, (%rdx,%rdi)
	leal	4(%r9), %edi
	movb	%al, (%rdx,%r14)
	leal	5(%r9), %r14d
	movzbl	(%r11,%rdi), %r10d
	movzbl	(%r11,%r14), %eax
	movb	%r10b, (%rdx,%rdi)
	leal	6(%r9), %edi
	movb	%al, (%rdx,%r14)
	leal	7(%r9), %r14d
	movzbl	(%r11,%rdi), %r10d
	addl	$8, %r9d
	movzbl	(%r11,%r14), %eax
	movb	%r10b, (%rdx,%rdi)
	movb	%al, (%rdx,%r14)
	cmpl	%ecx, %r9d
	jb	.L6379
	.p2align 4,,10
	.p2align 3
.L6361:
	shrb	$2, %r8b
	leaq	-1(%rdx), %r11
	andl	$3, %r8d
	testl	%r13d, %r13d
	jns	.L6362
	movb	$45, -1(%rdx)
.L6381:
	movq	%r11, %rdx
	jmp	.L6411
	.p2align 4,,10
	.p2align 3
.L6309:
	movabsq	$9007199254740992, %r12
	movl	$7, %edx
	leaq	8(%rbx), %rsi
	movq	%r12, -456(%rbp)
	leaq	-456(%rbp), %r12
	movq	%r12, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
	movzbl	-455(%rbp), %edx
	movl	%edx, %ecx
	notl	%edx
	andl	$120, %ecx
	andl	$56, %edx
	jne	.L6331
	testb	$92, -456(%rbp)
	jne	.L6734
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	cmpb	$56, %cl
	je	.L6735
	movq	16(%rbx), %rax
	jmp	.L6721
	.p2align 4,,10
	.p2align 3
.L6310:
	movabsq	$9007199254740992, %rax
	leaq	-456(%rbp), %r13
	leaq	8(%rbx), %rsi
	xorl	%edx, %edx
	movq	%rax, -456(%rbp)
	movq	%r13, %rdi
	vzeroupper
	call	_ZNSt8__format15__formatter_intIcE11_M_do_parseERSt26basic_format_parse_contextIcENS_10_Pres_typeE
.LEHE156:
	movzbl	-455(%rbp), %r10d
	andl	$120, %r10d
	jne	.L6312
	movzbl	-456(%rbp), %r14d
	testb	$92, %r14b
	jne	.L6736
	leaq	-368(%rbp), %rsi
	andl	$32, %r14d
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %r12
	movq	%rsi, -384(%rbp)
	movzbl	-416(%rbp), %ebx
	movq	$0, -376(%rbp)
	movb	$0, -368(%rbp)
	jne	.L6737
	movl	%ebx, %edx
	leaq	.LC50(%rip), %rcx
	leaq	.LC49(%rip), %r9
	negb	%dl
	leaq	-384(%rbp), %r14
	sbbq	%rdi, %rdi
	testb	%bl, %bl
	cmovne	%r9, %rcx
	leaq	5(%rdi), %r8
	xorl	%edx, %edx
	xorl	%esi, %esi
	movq	%r14, %rdi
.LEHB157:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm.isra.0
	movq	-376(%rbp), %rdi
.L6325:
	movq	-384(%rbp), %rsi
	movq	%r13, %r8
	movq	%r12, %rcx
	movq	%rdi, %rdx
	movl	$1, %r9d
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE
.LEHE157:
	movq	%rax, %r13
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	%r13, %rax
	jmp	.L6720
	.p2align 4,,10
	.p2align 3
.L6294:
	movabsq	$9007199254740992, %r14
	leaq	-456(%rbp), %r15
	leaq	8(%rbx), %rsi
	movq	%r14, -456(%rbp)
	movq	%r15, %rdi
	vzeroupper
.LEHB158:
	call	_ZNSt8__format14__formatter_fpIcE5parseERSt26basic_format_parse_contextIcE
	vmovdqa	-416(%rbp), %xmm0
	movq	%r15, %rdi
	movq	%rax, 8(%rbx)
	movq	48(%rbx), %rbx
	movq	%rbx, %rsi
	call	_ZNKSt8__format14__formatter_fpIcE6formatIDF128_NS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	movq	%rax, 16(%rbx)
	jmp	.L6289
	.p2align 4,,10
	.p2align 3
.L6713:
	vzeroupper
.L6292:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
	call	_ZNSt8__format33__invalid_arg_id_in_format_stringEv
	.p2align 4,,10
	.p2align 3
.L6312:
	movq	48(%rbx), %r12
	movzbl	-416(%rbp), %esi
	movq	%rax, 8(%rbx)
	cmpb	$56, %r10b
	jne	.L6317
	movb	%sil, -464(%rbp)
	movq	%r13, %rcx
	movq	%r12, %rdx
	movl	$1, %edi
	leaq	-464(%rbp), %rsi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
	jmp	.L6720
	.p2align 4,,10
	.p2align 3
.L6331:
	movq	%rax, 8(%rbx)
	movzbl	-416(%rbp), %esi
	movq	48(%rbx), %rbx
	testb	%cl, %cl
	je	.L6334
	movq	%rbx, %rdx
	movq	%r12, %rdi
	call	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
	jmp	.L6721
	.p2align 4,,10
	.p2align 3
.L6729:
	cmpl	$127, %edx
	ja	.L6339
	movb	%dl, -464(%rbp)
.L6724:
	leaq	-464(%rbp), %rsi
	movq	%r12, %rcx
	movq	%rbx, %rdx
	movl	$1, %edi
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
	jmp	.L6721
	.p2align 4,,10
	.p2align 3
.L6732:
	leal	128(%r13), %edx
	cmpl	$255, %edx
	ja	.L6339
	leaq	-456(%rbp), %rsi
	movq	%r12, %rcx
	movq	%rbx, %rdx
	movl	$1, %edi
	movb	%r13b, -456(%rbp)
	call	_ZNSt8__format22__write_padded_as_specIcNS_10_Sink_iterIcEEEET0_St17basic_string_viewINSt13type_identityIT_E4typeESt11char_traitsIS8_EEmRSt20basic_format_contextIS3_S6_ERKNS_5_SpecIS6_EENS_6_AlignE.constprop.0
	jmp	.L6721
	.p2align 4,,10
	.p2align 3
.L6412:
	movl	-456(%rbp), %r11d
	movl	-453(%rbp), %eax
	movl	%r11d, -464(%rbp)
	movl	%eax, -461(%rbp)
	vzeroupper
	jmp	.L6415
	.p2align 4,,10
	.p2align 3
.L6421:
	bsrq	%rbx, %rsi
	vmovdqa	.LC25(%rip), %xmm0
	leal	4(%rsi), %edi
	shrl	$2, %edi
	vmovdqa	%xmm0, -320(%rbp)
	leal	-1(%rdi), %r15d
	cmpq	$255, %rbx
	jbe	.L6423
.L6424:
	movq	%rbx, %rcx
	movq	%rbx, %r14
	movl	%r15d, %r13d
	shrq	$8, %rbx
	shrq	$4, %rcx
	leal	-1(%r15), %r8d
	leal	-2(%r15), %r10d
	andl	$15, %r14d
	movzbl	-320(%rbp,%r14), %edx
	andl	$15, %ecx
	movzbl	-320(%rbp,%rcx), %r9d
	movb	%dl, -270(%rbp,%r13)
	movb	%r9b, -270(%rbp,%r8)
	cmpq	$255, %rbx
	jbe	.L6423
	movq	%rbx, %r13
	leal	-3(%r15), %esi
	leal	-4(%r15), %edx
	movq	%rbx, %r11
	shrq	$4, %r13
	andl	$15, %r11d
	shrq	$8, %rbx
	movzbl	-320(%rbp,%r11), %eax
	andl	$15, %r13d
	movzbl	-320(%rbp,%r13), %r14d
	movb	%al, -270(%rbp,%r10)
	movb	%r14b, -270(%rbp,%rsi)
	cmpq	$255, %rbx
	jbe	.L6423
	movq	%rbx, %r9
	leal	-5(%r15), %r10d
	leal	-6(%r15), %eax
	movq	%rbx, %rcx
	shrq	$4, %r9
	andl	$15, %ecx
	shrq	$8, %rbx
	movzbl	-320(%rbp,%rcx), %r8d
	andl	$15, %r9d
	movzbl	-320(%rbp,%r9), %r11d
	movb	%r8b, -270(%rbp,%rdx)
	movb	%r11b, -270(%rbp,%r10)
	cmpq	$255, %rbx
	jbe	.L6423
	movq	%rbx, %r14
	movq	%rbx, %r13
	leal	-7(%r15), %edx
	shrq	$8, %rbx
	shrq	$4, %r14
	andl	$15, %r13d
	subl	$8, %r15d
	movzbl	-320(%rbp,%r13), %esi
	andl	$15, %r14d
	movzbl	-320(%rbp,%r14), %ecx
	movb	%sil, -270(%rbp,%rax)
	movb	%cl, -270(%rbp,%rdx)
	cmpq	$255, %rbx
	ja	.L6424
	.p2align 4,,10
	.p2align 3
.L6423:
	cmpq	$15, %rbx
	jbe	.L6425
	movq	%rbx, %r15
	shrq	$4, %rbx
	andl	$15, %r15d
	movzbl	-320(%rbp,%r15), %r8d
	movb	%r8b, -269(%rbp)
	movzbl	-320(%rbp,%rbx), %ebx
.L6426:
	movb	%bl, -270(%rbp)
	addl	$2, %edi
	jmp	.L6422
.L6351:
	testl	%r13d, %r13d
	jne	.L6347
	movzbl	-464(%rbp), %r8d
	movb	$48, -317(%rbp)
	leaq	-317(%rbp), %r15
	leaq	-316(%rbp), %rsi
	leaq	-318(%rbp), %r11
	movq	%r15, %rdx
	shrb	$2, %r8b
	andl	$3, %r8d
.L6362:
	movzbl	%r8b, %r13d
	cmpl	$1, %r13d
	je	.L6738
	cmpl	$3, %r13d
	jne	.L6411
	movb	$32, -1(%rdx)
	jmp	.L6381
	.p2align 4,,10
	.p2align 3
.L6733:
	negl	%edx
	cmpb	$3, %sil
	jbe	.L6343
	cmpb	$4, %sil
	je	.L6344
	cmpb	$40, %r14b
	leaq	.LC39(%rip), %r8
	leaq	.LC38(%rip), %rcx
	cmovne	%rcx, %r8
.L6348:
	leaq	-317(%rbp), %r15
	leaq	-285(%rbp), %rsi
	movq	%r8, -472(%rbp)
	movq	%r15, %rdi
	call	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	cmpb	$48, %r14b
	movzbl	-464(%rbp), %r8d
	movq	-472(%rbp), %r11
	movq	%rax, %rsi
	je	.L6739
.L6376:
	testb	$16, %r8b
	jne	.L6445
	.p2align 4,,10
	.p2align 3
.L6446:
	movq	%r15, %rdx
	jmp	.L6361
	.p2align 4,,10
	.p2align 3
.L6735:
	movzbl	-416(%rbp), %esi
.L6334:
	movb	%sil, -464(%rbp)
	jmp	.L6724
	.p2align 4,,10
	.p2align 3
.L6317:
	movq	%r12, %rdx
	movq	%r13, %rdi
	call	_ZNKSt8__format15__formatter_intIcE6formatIhNS_10_Sink_iterIcEEEENSt20basic_format_contextIT0_cE8iteratorET_RS7_
.LEHE158:
	jmp	.L6720
	.p2align 4,,10
	.p2align 3
.L6413:
	movl	-456(%rbp), %r9d
	movl	-453(%rbp), %r10d
	movl	%r9d, -464(%rbp)
	movl	%r10d, -461(%rbp)
	jmp	.L6415
	.p2align 4,,10
	.p2align 3
.L6425:
	movzbl	-320(%rbp,%rbx), %ebx
	jmp	.L6426
	.p2align 4,,10
	.p2align 3
.L6455:
	movl	$43, %r11d
.L6410:
	movb	%r11b, -1(%rdx)
	subq	$1, %rdx
	jmp	.L6411
	.p2align 4,,10
	.p2align 3
.L6389:
	testl	%edx, %edx
	je	.L6402
	leaq	-269(%rbp), %r15
	leaq	-237(%rbp), %rsi
	movq	%r15, %rdi
	call	_ZNSt8__detail13__to_chars_10IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movq	%rax, %rsi
.L6718:
	movzbl	-456(%rbp), %r14d
	movq	%r15, %rdx
	jmp	.L6399
	.p2align 4,,10
	.p2align 3
.L6388:
	cmpb	$40, %r13b
	je	.L6740
	testl	%edx, %edx
	jne	.L6450
	movb	$48, -269(%rbp)
	movzbl	-456(%rbp), %r14d
	leaq	.LC38(%rip), %r11
	leaq	-268(%rbp), %rsi
	leaq	-269(%rbp), %r15
	cmpb	$48, %r13b
	je	.L6406
.L6405:
	testb	$16, %r14b
	jne	.L6453
.L6719:
	movq	%r15, %rdx
	jmp	.L6399
	.p2align 4,,10
	.p2align 3
.L6387:
	testl	%edx, %edx
	je	.L6402
	leaq	-269(%rbp), %r15
	leaq	-237(%rbp), %rsi
	movq	%r15, %rdi
	call	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movzbl	-456(%rbp), %r14d
	movl	$1, %ecx
	leaq	.LC37(%rip), %r11
	movq	%rax, %rsi
.L6403:
	movq	%r15, %rdx
	testb	$16, %r14b
	je	.L6399
	movq	%rcx, %rdx
	negq	%rdx
	jmp	.L6398
	.p2align 4,,10
	.p2align 3
.L6402:
	movb	$48, -269(%rbp)
	leaq	-268(%rbp), %rsi
	leaq	-269(%rbp), %r15
	jmp	.L6718
.L6731:
	movl	$32, %r11d
	jmp	.L6410
.L6343:
	cmpb	$1, %sil
	jbe	.L6347
	cmpb	$16, %r14b
	leaq	.LC35(%rip), %r11
	leaq	.LC36(%rip), %rsi
	cmovne	%rsi, %r11
.L6428:
	bsrl	%edx, %r15d
	movl	$32, %r10d
	movl	$31, %eax
	xorl	$31, %r15d
	subl	%r15d, %r10d
	subl	%r15d, %eax
	je	.L6359
	movl	%eax, %edi
	movl	$30, %r9d
	leaq	-320(%rbp,%rdi), %rcx
	subl	%r15d, %r9d
	leaq	-321(%rbp,%rdi), %rdi
	subq	%r9, %rdi
	movq	%rcx, %r14
	subq	%rdi, %r14
	andl	$7, %r14d
	je	.L6358
	cmpq	$1, %r14
	je	.L6609
	cmpq	$2, %r14
	je	.L6610
	cmpq	$3, %r14
	je	.L6611
	cmpq	$4, %r14
	je	.L6612
	cmpq	$5, %r14
	je	.L6613
	cmpq	$6, %r14
	je	.L6614
	movl	%edx, %esi
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%rcx)
.L6614:
	movl	%edx, %eax
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%rcx)
.L6613:
	movl	%edx, %esi
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%rcx)
.L6612:
	movl	%edx, %eax
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%rcx)
.L6611:
	movl	%edx, %esi
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%rcx)
.L6610:
	movl	%edx, %eax
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%rcx)
.L6609:
	movl	%edx, %esi
	subq	$1, %rcx
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%rcx)
	cmpq	%rdi, %rcx
	je	.L6359
.L6358:
	movl	%edx, %eax
	movl	%edx, %esi
	movl	%edx, %r8d
	movl	%edx, %r15d
	andl	$1, %eax
	shrl	$5, %esi
	movl	%edx, %r9d
	movl	%edx, %r14d
	addl	$48, %eax
	andl	$1, %esi
	shrl	%r8d
	subq	$8, %rcx
	addl	$48, %esi
	movb	%al, 11(%rcx)
	movl	%edx, %eax
	shrl	$2, %r15d
	movb	%sil, 6(%rcx)
	shrl	$3, %r9d
	movl	%edx, %esi
	shrl	$4, %r14d
	shrl	$6, %eax
	andl	$1, %r8d
	andl	$1, %r15d
	andl	$1, %r9d
	andl	$1, %r14d
	andl	$1, %eax
	shrb	$7, %sil
	addl	$48, %r8d
	addl	$48, %r15d
	addl	$48, %r9d
	addl	$48, %r14d
	addl	$48, %eax
	addl	$48, %esi
	movb	%r8b, 10(%rcx)
	shrl	$8, %edx
	movb	%r15b, 9(%rcx)
	movb	%r9b, 8(%rcx)
	movb	%r14b, 7(%rcx)
	movb	%al, 5(%rcx)
	movb	%sil, 4(%rcx)
	cmpq	%rdi, %rcx
	jne	.L6358
.L6359:
	movslq	%r10d, %rsi
	leaq	-317(%rbp), %r15
	movl	$49, %r10d
	addq	%r15, %rsi
	jmp	.L6356
.L6738:
	movb	$43, -1(%rdx)
	jmp	.L6381
.L6727:
	leaq	1(%rax), %rdx
	cmpq	%rdx, %r13
	je	.L6456
	movzbl	1(%rax), %edi
	movq	%rdx, %rsi
	jmp	.L6418
.L6349:
	testl	%r13d, %r13d
	jne	.L6344
	movb	$48, -317(%rbp)
	xorl	%edx, %edx
	xorl	%r11d, %r11d
	xorl	%eax, %eax
	movzbl	-464(%rbp), %r8d
	leaq	-316(%rbp), %rsi
	leaq	-317(%rbp), %r15
.L6375:
	testb	$16, %r8b
	je	.L6446
	testb	%dl, %dl
	je	.L6446
	movq	%rax, %rdx
	negq	%rdx
	jmp	.L6360
.L6350:
	cmpb	$40, %r14b
	je	.L6741
	testl	%r13d, %r13d
	jne	.L6442
	movb	$48, -317(%rbp)
	movzbl	-464(%rbp), %r8d
	cmpb	$48, %r14b
	je	.L6443
	leaq	-316(%rbp), %rsi
	leaq	.LC38(%rip), %r11
	leaq	-317(%rbp), %r15
	jmp	.L6376
.L6347:
	cmpl	$9, %edx
	jbe	.L6435
	cmpl	$99, %edx
	jbe	.L6742
	cmpl	$999, %edx
	jbe	.L6436
	cmpl	$9999, %edx
	jbe	.L6743
	movl	%edx, %r8d
	movl	$5, %r11d
	cmpl	$99999, %edx
	jbe	.L6366
	cmpl	$999999, %edx
	jbe	.L6744
	cmpl	$9999999, %edx
	jbe	.L6438
	cmpl	$99999999, %edx
	jbe	.L6439
	cmpq	$999999999, %r8
	jbe	.L6440
	movl	$5, %r11d
.L6370:
	addl	$5, %r11d
.L6366:
	vmovdqa	.LC26(%rip), %ymm4
	vmovdqa	.LC27(%rip), %ymm5
	leal	-1(%r11), %r15d
	vmovdqa	.LC28(%rip), %ymm6
	vmovdqa	.LC29(%rip), %ymm7
	vmovdqa	.LC30(%rip), %ymm8
	vmovdqa	.LC31(%rip), %ymm9
	vmovdqu	%ymm4, -272(%rbp)
	vmovdqa	.LC32(%rip), %xmm10
	vmovdqu	%ymm5, -240(%rbp)
	vmovdqu	%ymm9, -112(%rbp)
	vmovdqu	%ymm6, -208(%rbp)
	vmovdqu	%ymm7, -176(%rbp)
	vmovdqu	%ymm8, -144(%rbp)
	vmovdqu	%xmm10, -87(%rbp)
.L6372:
	imulq	$1374389535, %r8, %r8
	movl	%edx, %ecx
	movl	%edx, %r9d
	movl	%r15d, %edi
	leal	-1(%r15), %eax
	shrq	$37, %r8
	imull	$100, %r8d, %esi
	movl	%r8d, %edx
	subl	%esi, %ecx
	addl	%ecx, %ecx
	leal	1(%rcx), %r10d
	movzbl	-272(%rbp,%rcx), %esi
	movzbl	-272(%rbp,%r10), %r14d
	leal	-2(%r15), %r10d
	movb	%r14b, -317(%rbp,%rdi)
	movb	%sil, -317(%rbp,%rax)
	cmpl	$9999, %r9d
	jbe	.L6704
	movl	%r8d, %r9d
	movl	%r8d, %edi
	movl	%r10d, %eax
	imulq	$1374389535, %r9, %rcx
	movl	%r8d, %r9d
	shrq	$37, %rcx
	imull	$100, %ecx, %r14d
	movl	%ecx, %edx
	leal	-3(%r15), %ecx
	subl	$4, %r15d
	subl	%r14d, %edi
	addl	%edi, %edi
	movzbl	-272(%rbp,%rdi), %r14d
	leal	1(%rdi), %esi
	movzbl	-272(%rbp,%rsi), %r10d
	movb	%r10b, -317(%rbp,%rax)
	movb	%r14b, -317(%rbp,%rcx)
	cmpl	$9999, %r8d
	jbe	.L6704
	movl	%edx, %r8d
	jmp	.L6372
.L6730:
	bsrl	%edx, %r14d
	movl	$32, %ecx
	movl	$31, %esi
	xorl	$31, %r14d
	subl	%r14d, %ecx
	subl	%r14d, %esi
	je	.L6397
	movl	%esi, %r15d
	movl	$30, %r9d
	leaq	-272(%rbp,%r15), %r8
	leaq	-273(%rbp,%r15), %rdi
	subl	%r14d, %r9d
	subq	%r9, %rdi
	movq	%r8, %r10
	subq	%rdi, %r10
	andl	$7, %r10d
	je	.L6396
	cmpq	$1, %r10
	je	.L6622
	cmpq	$2, %r10
	je	.L6623
	cmpq	$3, %r10
	je	.L6624
	cmpq	$4, %r10
	je	.L6625
	cmpq	$5, %r10
	je	.L6626
	cmpq	$6, %r10
	je	.L6627
	movl	%edx, %eax
	subq	$1, %r8
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r8)
.L6627:
	movl	%edx, %esi
	subq	$1, %r8
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%r8)
.L6626:
	movl	%edx, %eax
	subq	$1, %r8
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r8)
.L6625:
	movl	%edx, %esi
	subq	$1, %r8
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%r8)
.L6624:
	movl	%edx, %eax
	subq	$1, %r8
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r8)
.L6623:
	movl	%edx, %esi
	subq	$1, %r8
	shrl	%edx
	andl	$1, %esi
	addl	$48, %esi
	movb	%sil, 4(%r8)
.L6622:
	movl	%edx, %eax
	subq	$1, %r8
	shrl	%edx
	andl	$1, %eax
	addl	$48, %eax
	movb	%al, 4(%r8)
	cmpq	%rdi, %r8
	je	.L6397
.L6396:
	movl	%edx, %esi
	movl	%edx, %r13d
	movl	%edx, %r14d
	movl	%edx, %r15d
	andl	$1, %esi
	movl	%edx, %r9d
	movl	%edx, %r10d
	movl	%edx, %eax
	shrl	%r13d
	addl	$48, %esi
	shrl	$2, %r14d
	subq	$8, %r8
	movb	%sil, 11(%r8)
	shrl	$3, %r15d
	movl	%edx, %esi
	shrl	$4, %r9d
	shrl	$5, %r10d
	shrl	$6, %eax
	andl	$1, %r13d
	andl	$1, %r14d
	andl	$1, %r15d
	andl	$1, %r9d
	andl	$1, %r10d
	andl	$1, %eax
	shrb	$7, %sil
	addl	$48, %r13d
	addl	$48, %r14d
	addl	$48, %r15d
	addl	$48, %r9d
	addl	$48, %r10d
	addl	$48, %eax
	addl	$48, %esi
	movb	%r13b, 10(%r8)
	shrl	$8, %edx
	movb	%r14b, 9(%r8)
	movb	%r15b, 8(%r8)
	movb	%r9b, 7(%r8)
	movb	%r10b, 6(%r8)
	movb	%al, 5(%r8)
	movb	%sil, 4(%r8)
	cmpq	%rdi, %r8
	jne	.L6396
.L6397:
	leaq	-269(%rbp), %r15
	movslq	%ecx, %rsi
	movl	$49, %edx
	addq	%r15, %rsi
	jmp	.L6394
.L6344:
	leaq	-317(%rbp), %r15
	leaq	-285(%rbp), %rsi
	movq	%r15, %rdi
	call	_ZNSt8__detail12__to_chars_8IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	movzbl	-464(%rbp), %r8d
	movl	$1, %edx
	leaq	.LC37(%rip), %r11
	movq	%rax, %rsi
	movl	$1, %eax
	jmp	.L6375
.L6704:
	cmpl	$999, %r9d
	jbe	.L6714
	vzeroupper
.L6365:
	addl	%edx, %edx
	leal	1(%rdx), %r15d
	movzbl	-272(%rbp,%rdx), %edx
	movzbl	-272(%rbp,%r15), %r9d
	movb	%r9b, -316(%rbp)
.L6373:
	leaq	-317(%rbp), %r15
	movl	%r11d, %esi
	movb	%dl, -317(%rbp)
	movzbl	-464(%rbp), %r8d
	addq	%r15, %rsi
	movq	%r15, %rdx
	jmp	.L6361
.L6740:
	testl	%edx, %edx
	jne	.L6449
	movb	$48, -269(%rbp)
	movzbl	-456(%rbp), %r14d
	leaq	.LC39(%rip), %r11
	leaq	-268(%rbp), %rsi
	leaq	-269(%rbp), %r15
	jmp	.L6405
.L6450:
	leaq	.LC38(%rip), %rdi
.L6404:
	leaq	-269(%rbp), %r15
	movq	%rdi, -472(%rbp)
	leaq	-237(%rbp), %rsi
	movq	%r15, %rdi
	call	_ZNSt8__detail13__to_chars_16IjEENSt9enable_ifIXsrSt5__or_IJS2_IJSt7is_sameINSt9remove_cvIT_E4typeEaES3_IS7_sES3_IS7_iES3_IS7_lES3_IS7_xEEES2_IJS3_IS7_hES3_IS7_tES3_IS7_jES3_IS7_mES3_IS7_yEEES3_IS5_nES3_IS5_oES3_IcS7_EEE5valueESt15to_chars_resultE4typeEPcSR_S5_
	cmpb	$48, %r13b
	movzbl	-456(%rbp), %r14d
	movq	-472(%rbp), %r11
	movq	%rax, %rsi
	jne	.L6405
	cmpq	%r15, %rax
	je	.L6452
.L6406:
	movq	%rsi, %r10
	movq	%r15, %r13
	subq	%r15, %r10
	andl	$7, %r10d
	je	.L6407
	cmpq	$1, %r10
	je	.L6628
	cmpq	$2, %r10
	je	.L6629
	cmpq	$3, %r10
	je	.L6630
	cmpq	$4, %r10
	je	.L6631
	cmpq	$5, %r10
	je	.L6632
	cmpq	$6, %r10
	je	.L6633
	movsbl	(%r15), %edi
	movq	%r11, -480(%rbp)
	leaq	-268(%rbp), %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-480(%rbp), %r11
	movq	-472(%rbp), %rsi
	movb	%al, (%r15)
.L6633:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$1, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-480(%rbp), %r11
	movq	-472(%rbp), %rsi
	movb	%al, -1(%r13)
.L6632:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$1, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-480(%rbp), %r11
	movq	-472(%rbp), %rsi
	movb	%al, -1(%r13)
.L6631:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$1, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-480(%rbp), %r11
	movq	-472(%rbp), %rsi
	movb	%al, -1(%r13)
.L6630:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$1, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-480(%rbp), %r11
	movq	-472(%rbp), %rsi
	movb	%al, -1(%r13)
.L6629:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$1, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-480(%rbp), %r11
	movq	-472(%rbp), %rsi
	movb	%al, -1(%r13)
.L6628:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$1, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movq	-472(%rbp), %rsi
	movq	-480(%rbp), %r11
	movb	%al, -1(%r13)
	cmpq	%r13, %rsi
	je	.L6452
.L6407:
	movsbl	0(%r13), %edi
	movq	%r11, -480(%rbp)
	addq	$8, %r13
	movq	%rsi, -472(%rbp)
	call	toupper@PLT
	movsbl	-7(%r13), %edi
	movb	%al, -8(%r13)
	call	toupper@PLT
	movsbl	-6(%r13), %edi
	movb	%al, -7(%r13)
	call	toupper@PLT
	movsbl	-5(%r13), %edi
	movb	%al, -6(%r13)
	call	toupper@PLT
	movsbl	-4(%r13), %edi
	movb	%al, -5(%r13)
	call	toupper@PLT
	movsbl	-3(%r13), %edi
	movb	%al, -4(%r13)
	call	toupper@PLT
	movsbl	-2(%r13), %edi
	movb	%al, -3(%r13)
	call	toupper@PLT
	movsbl	-1(%r13), %edi
	movb	%al, -2(%r13)
	call	toupper@PLT
	movq	-472(%rbp), %rsi
	movq	-480(%rbp), %r11
	movb	%al, -1(%r13)
	cmpq	%r13, %rsi
	jne	.L6407
.L6452:
	movl	$2, %ecx
	jmp	.L6403
.L6737:
	cmpb	$0, 32(%r12)
	leaq	24(%r12), %r15
	je	.L6745
.L6319:
	leaq	-464(%rbp), %r14
	movq	%r15, %rsi
	movq	%r14, %rdi
	call	_ZNSt6localeC1ERKS_@PLT
	leaq	_ZNSt7__cxx118numpunctIcE2idE(%rip), %rdi
	call	_ZNKSt6locale2id5_M_idEv@PLT
	movq	-464(%rbp), %r8
	movq	8(%r8), %rcx
	movq	(%rcx,%rax,8), %r15
	testq	%r15, %r15
	je	.L6320
	movq	%r14, %rdi
	call	_ZNSt6localeD1Ev@PLT
	testb	%bl, %bl
	je	.L6746
	movq	(%r15), %r10
	leaq	-352(%rbp), %rbx
	leaq	-384(%rbp), %r14
	movq	%r15, %rsi
	movq	%rbx, %rdi
.LEHB159:
	call	*40(%r10)
.L6324:
	leaq	-384(%rbp), %r14
	movq	%rbx, %rsi
	movq	%r14, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0
	movq	%rbx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	-376(%rbp), %rdi
	jmp	.L6325
.L6714:
	vzeroupper
.L6363:
	addl	$48, %edx
	jmp	.L6373
.L6741:
	testl	%r13d, %r13d
	jne	.L6441
	movb	$48, -317(%rbp)
	movzbl	-464(%rbp), %r8d
	leaq	-316(%rbp), %rsi
	leaq	.LC39(%rip), %r11
	leaq	-317(%rbp), %r15
	jmp	.L6376
.L6739:
	cmpq	%r15, %rax
	je	.L6444
.L6377:
	movq	%rsi, %r9
	movq	%r15, %r14
	subq	%r15, %r9
	andl	$7, %r9d
	je	.L6378
	cmpq	$1, %r9
	je	.L6615
	cmpq	$2, %r9
	je	.L6616
	cmpq	$3, %r9
	je	.L6617
	cmpq	$4, %r9
	je	.L6618
	cmpq	$5, %r9
	je	.L6619
	cmpq	$6, %r9
	je	.L6620
	movsbl	(%r15), %edi
	movb	%r8b, -481(%rbp)
	leaq	-316(%rbp), %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movzbl	-481(%rbp), %r8d
	movq	-480(%rbp), %rsi
	movb	%al, (%r15)
	movq	-472(%rbp), %r11
.L6620:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$1, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movzbl	-481(%rbp), %r8d
	movq	-480(%rbp), %rsi
	movb	%al, -1(%r14)
	movq	-472(%rbp), %r11
.L6619:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$1, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movzbl	-481(%rbp), %r8d
	movq	-480(%rbp), %rsi
	movb	%al, -1(%r14)
	movq	-472(%rbp), %r11
.L6618:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$1, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movzbl	-481(%rbp), %r8d
	movq	-480(%rbp), %rsi
	movb	%al, -1(%r14)
	movq	-472(%rbp), %r11
.L6617:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$1, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movzbl	-481(%rbp), %r8d
	movq	-480(%rbp), %rsi
	movb	%al, -1(%r14)
	movq	-472(%rbp), %r11
.L6616:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$1, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movzbl	-481(%rbp), %r8d
	movq	-480(%rbp), %rsi
	movb	%al, -1(%r14)
	movq	-472(%rbp), %r11
.L6615:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$1, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movb	%al, -1(%r14)
.L6717:
	movq	-480(%rbp), %rsi
	movq	-472(%rbp), %r11
	movzbl	-481(%rbp), %r8d
	cmpq	%rsi, %r14
	je	.L6444
.L6378:
	movsbl	(%r14), %edi
	movb	%r8b, -481(%rbp)
	addq	$8, %r14
	movq	%rsi, -480(%rbp)
	movq	%r11, -472(%rbp)
	call	toupper@PLT
	movsbl	-7(%r14), %edi
	movb	%al, -8(%r14)
	call	toupper@PLT
	movsbl	-6(%r14), %edi
	movb	%al, -7(%r14)
	call	toupper@PLT
	movsbl	-5(%r14), %edi
	movb	%al, -6(%r14)
	call	toupper@PLT
	movsbl	-4(%r14), %edi
	movb	%al, -5(%r14)
	call	toupper@PLT
	movsbl	-3(%r14), %edi
	movb	%al, -4(%r14)
	call	toupper@PLT
	movsbl	-2(%r14), %edi
	movb	%al, -3(%r14)
	call	toupper@PLT
	movsbl	-1(%r14), %edi
	movb	%al, -2(%r14)
	call	toupper@PLT
	movb	%al, -1(%r14)
	jmp	.L6717
.L6444:
	movl	$1, %edx
	movl	$2, %eax
	jmp	.L6375
.L6449:
	leaq	.LC39(%rip), %rdi
	jmp	.L6404
.L6746:
	movq	(%r15), %rax
	leaq	-352(%rbp), %rbx
	leaq	-384(%rbp), %r14
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	*48(%rax)
.LEHE159:
	jmp	.L6324
.L6443:
	leaq	.LC38(%rip), %r11
	leaq	-316(%rbp), %rsi
	leaq	-317(%rbp), %r15
	jmp	.L6377
.L6456:
	movq	%r13, %rsi
	jmp	.L6417
.L6745:
	movq	%r15, %rdi
	call	_ZNSt6localeC1Ev@PLT
	movb	$1, 32(%r12)
	jmp	.L6319
.L6440:
	movl	$9, %r11d
	jmp	.L6366
.L6439:
	movl	$8, %r11d
	jmp	.L6366
.L6438:
	movl	$7, %r11d
	jmp	.L6366
.L6442:
	leaq	.LC38(%rip), %r8
	jmp	.L6348
.L6736:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	je	.L6314
.L6723:
	call	__stack_chk_fail@PLT
.L6441:
	leaq	.LC39(%rip), %r8
	jmp	.L6348
.L6435:
	movl	$1, %r11d
	jmp	.L6363
.L6743:
	movl	$4, %r11d
	movl	%edx, %r8d
	jmp	.L6366
.L6436:
	movl	$3, %r11d
	movl	%edx, %r8d
	jmp	.L6366
.L6742:
	leaq	-272(%rbp), %rdi
	leaq	.LC53(%rip), %rsi
	movl	$201, %ecx
	movl	$2, %r11d
	rep movsb
	jmp	.L6365
.L6726:
	vzeroupper
	call	__stack_chk_fail@PLT
.L6320:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
.LEHB160:
	call	_ZSt16__throw_bad_castv@PLT
.LEHE160:
.L6744:
	movl	$1, %r11d
	jmp	.L6370
.L6459:
	endbr64
	movq	%rax, %r12
	jmp	.L6328
.L6458:
	endbr64
	movq	%rax, %r12
	jmp	.L6329
.L6728:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
.LEHB161:
	call	_ZNSt8__format29__failed_to_parse_format_specEv
.L6328:
	movq	%r14, %rdi
	vzeroupper
	leaq	-384(%rbp), %r14
	call	_ZNSt6localeD1Ev@PLT
.L6329:
	movq	%r14, %rdi
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
	movq	%r12, %rdi
	call	_Unwind_Resume@PLT
.L6314:
	leaq	.LC51(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.L6734:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
	leaq	.LC52(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.L6339:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L6723
	leaq	.LC40(%rip), %rdi
	call	_ZSt20__throw_format_errorPKc
.LEHE161:
	.cfi_endproc
.LFE13440:
	.section	.gcc_except_table
.LLSDA13440:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE13440-.LLSDACSB13440
.LLSDACSB13440:
	.uleb128 .LEHB156-.LFB13440
	.uleb128 .LEHE156-.LEHB156
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB157-.LFB13440
	.uleb128 .LEHE157-.LEHB157
	.uleb128 .L6458-.LFB13440
	.uleb128 0
	.uleb128 .LEHB158-.LFB13440
	.uleb128 .LEHE158-.LEHB158
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB159-.LFB13440
	.uleb128 .LEHE159-.LEHB159
	.uleb128 .L6458-.LFB13440
	.uleb128 0
	.uleb128 .LEHB160-.LFB13440
	.uleb128 .LEHE160-.LEHB160
	.uleb128 .L6459-.LFB13440
	.uleb128 0
	.uleb128 .LEHB161-.LFB13440
	.uleb128 .LEHE161-.LEHB161
	.uleb128 0
	.uleb128 0
.LLSDACSE13440:
	.section	.text._ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,"axG",@progbits,_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm,comdat
	.size	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm, .-_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm
	.section	.rodata
	.align 32
	.type	CSWTCH.915, @object
	.size	CSWTCH.915, 56
CSWTCH.915:
	.long	3
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1
	.long	0
	.long	2
	.weak	_ZTSSt12format_error
	.section	.rodata._ZTSSt12format_error,"aG",@progbits,_ZTSSt12format_error,comdat
	.align 16
	.type	_ZTSSt12format_error, @object
	.size	_ZTSSt12format_error, 17
_ZTSSt12format_error:
	.string	"St12format_error"
	.weak	_ZTISt12format_error
	.section	.data.rel.ro._ZTISt12format_error,"awG",@progbits,_ZTISt12format_error,comdat
	.align 8
	.type	_ZTISt12format_error, @object
	.size	_ZTISt12format_error, 24
_ZTISt12format_error:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSSt12format_error
	.quad	_ZTISt13runtime_error
	.weak	_ZTSNSt8__format5_SinkIcEE
	.section	.rodata._ZTSNSt8__format5_SinkIcEE,"aG",@progbits,_ZTSNSt8__format5_SinkIcEE,comdat
	.align 16
	.type	_ZTSNSt8__format5_SinkIcEE, @object
	.size	_ZTSNSt8__format5_SinkIcEE, 23
_ZTSNSt8__format5_SinkIcEE:
	.string	"NSt8__format5_SinkIcEE"
	.weak	_ZTINSt8__format5_SinkIcEE
	.section	.data.rel.ro._ZTINSt8__format5_SinkIcEE,"awG",@progbits,_ZTINSt8__format5_SinkIcEE,comdat
	.align 8
	.type	_ZTINSt8__format5_SinkIcEE, @object
	.size	_ZTINSt8__format5_SinkIcEE, 16
_ZTINSt8__format5_SinkIcEE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSNSt8__format5_SinkIcEE
	.weak	_ZTSNSt8__format9_Buf_sinkIcEE
	.section	.rodata._ZTSNSt8__format9_Buf_sinkIcEE,"aG",@progbits,_ZTSNSt8__format9_Buf_sinkIcEE,comdat
	.align 16
	.type	_ZTSNSt8__format9_Buf_sinkIcEE, @object
	.size	_ZTSNSt8__format9_Buf_sinkIcEE, 27
_ZTSNSt8__format9_Buf_sinkIcEE:
	.string	"NSt8__format9_Buf_sinkIcEE"
	.weak	_ZTINSt8__format9_Buf_sinkIcEE
	.section	.data.rel.ro._ZTINSt8__format9_Buf_sinkIcEE,"awG",@progbits,_ZTINSt8__format9_Buf_sinkIcEE,comdat
	.align 8
	.type	_ZTINSt8__format9_Buf_sinkIcEE, @object
	.size	_ZTINSt8__format9_Buf_sinkIcEE, 24
_ZTINSt8__format9_Buf_sinkIcEE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSNSt8__format9_Buf_sinkIcEE
	.quad	_ZTINSt8__format5_SinkIcEE
	.weak	_ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE
	.section	.rodata._ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE,"aG",@progbits,_ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE,comdat
	.align 32
	.type	_ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE, @object
	.size	_ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE, 78
_ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE:
	.string	"NSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE"
	.weak	_ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE
	.section	.data.rel.ro._ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE,"awG",@progbits,_ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE,comdat
	.align 8
	.type	_ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE, @object
	.size	_ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE, 24
_ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE
	.quad	_ZTINSt8__format9_Buf_sinkIcEE
	.weak	_ZTSNSt8__format8_ScannerIcEE
	.section	.rodata._ZTSNSt8__format8_ScannerIcEE,"aG",@progbits,_ZTSNSt8__format8_ScannerIcEE,comdat
	.align 16
	.type	_ZTSNSt8__format8_ScannerIcEE, @object
	.size	_ZTSNSt8__format8_ScannerIcEE, 26
_ZTSNSt8__format8_ScannerIcEE:
	.string	"NSt8__format8_ScannerIcEE"
	.weak	_ZTINSt8__format8_ScannerIcEE
	.section	.data.rel.ro._ZTINSt8__format8_ScannerIcEE,"awG",@progbits,_ZTINSt8__format8_ScannerIcEE,comdat
	.align 8
	.type	_ZTINSt8__format8_ScannerIcEE, @object
	.size	_ZTINSt8__format8_ScannerIcEE, 16
_ZTINSt8__format8_ScannerIcEE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSNSt8__format8_ScannerIcEE
	.weak	_ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE
	.section	.rodata._ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE,"aG",@progbits,_ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE,comdat
	.align 32
	.type	_ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE, @object
	.size	_ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE, 48
_ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE:
	.string	"NSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE"
	.weak	_ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE
	.section	.data.rel.ro._ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE,"awG",@progbits,_ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE,comdat
	.align 8
	.type	_ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE, @object
	.size	_ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE, 24
_ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE
	.quad	_ZTINSt8__format9_Buf_sinkIcEE
	.weak	_ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE
	.section	.rodata._ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE,"aG",@progbits,_ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE,comdat
	.align 32
	.type	_ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE, @object
	.size	_ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE, 57
_ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE:
	.string	"NSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE"
	.weak	_ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE
	.section	.data.rel.ro._ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE,"awG",@progbits,_ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE,comdat
	.align 8
	.type	_ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE, @object
	.size	_ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE, 24
_ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE
	.quad	_ZTINSt8__format8_ScannerIcEE
	.weak	_ZTVSt12format_error
	.section	.data.rel.ro._ZTVSt12format_error,"awG",@progbits,_ZTVSt12format_error,comdat
	.align 8
	.type	_ZTVSt12format_error, @object
	.size	_ZTVSt12format_error, 40
_ZTVSt12format_error:
	.quad	0
	.quad	_ZTISt12format_error
	.quad	_ZNSt12format_errorD1Ev
	.quad	_ZNSt12format_errorD0Ev
	.quad	_ZNKSt13runtime_error4whatEv
	.weak	_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE
	.section	.data.rel.ro.local._ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE,"awG",@progbits,_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE,comdat
	.align 8
	.type	_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE, @object
	.size	_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE, 24
_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE:
	.quad	0
	.quad	_ZTINSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE
	.quad	_ZNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE11_M_overflowEv
	.weak	_ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE
	.section	.data.rel.ro.local._ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE,"awG",@progbits,_ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE,comdat
	.align 8
	.type	_ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE, @object
	.size	_ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE, 24
_ZTVNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE:
	.quad	0
	.quad	_ZTINSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEEE
	.quad	_ZNSt8__format10_Iter_sinkIcNS_10_Sink_iterIcEEE11_M_overflowEv
	.weak	_ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE
	.section	.data.rel.ro.local._ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE,"awG",@progbits,_ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE,comdat
	.align 8
	.type	_ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE, @object
	.size	_ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE, 32
_ZTVNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE:
	.quad	0
	.quad	_ZTINSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcEE
	.quad	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE11_M_on_charsEPKc
	.quad	_ZNSt8__format19_Formatting_scannerINS_10_Sink_iterIcEEcE13_M_format_argEm
	.weak	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE
	.section	.rodata._ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE,"aG",@progbits,_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE,comdat
	.align 32
	.type	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE, @gnu_unique_object
	.size	_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE, 256
_ZNSt8__detail31__from_chars_alnum_to_val_tableILb0EE5valueE:
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	8
	.byte	9
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	10
	.byte	11
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	16
	.byte	17
	.byte	18
	.byte	19
	.byte	20
	.byte	21
	.byte	22
	.byte	23
	.byte	24
	.byte	25
	.byte	26
	.byte	27
	.byte	28
	.byte	29
	.byte	30
	.byte	31
	.byte	32
	.byte	33
	.byte	34
	.byte	35
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	10
	.byte	11
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	16
	.byte	17
	.byte	18
	.byte	19
	.byte	20
	.byte	21
	.byte	22
	.byte	23
	.byte	24
	.byte	25
	.byte	26
	.byte	27
	.byte	28
	.byte	29
	.byte	30
	.byte	31
	.byte	32
	.byte	33
	.byte	34
	.byte	35
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.byte	127
	.globl	efficient_cpu_weight
	.bss
	.align 8
	.type	efficient_cpu_weight, @object
	.size	efficient_cpu_weight, 8
efficient_cpu_weight:
	.zero	8
	.globl	performance_cpu_weight
	.data
	.align 8
	.type	performance_cpu_weight, @object
	.size	performance_cpu_weight, 8
performance_cpu_weight:
	.long	0
	.long	1073741824
	.set	.LC0,.LC1
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC1:
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC3:
	.long	0
	.long	1068498944
	.align 8
.LC8:
	.long	-1598689907
	.long	1051772663
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC25:
	.quad	3978425819141910832
	.quad	7378413942531504440
	.section	.rodata.cst32
	.align 32
.LC26:
	.quad	3688503277381496880
	.quad	3976738051646829616
	.quad	3544667369688283184
	.quad	3832902143785906737
	.align 32
.LC27:
	.quad	4121136918051239473
	.quad	3689066235924983858
	.quad	3977301010190316594
	.quad	3545230328231770162
	.align 32
.LC28:
	.quad	3833465102329393715
	.quad	4121699876594726451
	.quad	3689629194468470836
	.quad	3977863968733803572
	.align 32
.LC29:
	.quad	3545793286775257140
	.quad	3834028060872880693
	.quad	4122262835138213429
	.quad	3690192153011957814
	.align 32
.LC30:
	.quad	3978426927277290550
	.quad	3546356245318744118
	.quad	3834591019416367671
	.quad	4122825793681700407
	.align 32
.LC31:
	.quad	3690755111555444792
	.quad	3978989885820777528
	.quad	3546919203862231096
	.quad	3835153977959854649
	.section	.rodata.cst16
	.align 16
.LC32:
	.quad	4122263930388298034
	.quad	16106987313379638
	.align 16
.LC56:
	.quad	0
	.quad	-1
	.section	.data.rel.ro.local,"aw"
	.align 8
.LC58:
	.quad	_ZTVNSt8__format9_Seq_sinkINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE+16
	.section	.rodata.cst8
	.align 8
.LC61:
	.long	-755914244
	.long	1062232653
	.section	.rodata.cst32
	.align 32
.LC69:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
