??&
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
@
Softsign
features"T
activations"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8?? 
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
enc_outer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<*#
shared_nameenc_outer_0/kernel
z
&enc_outer_0/kernel/Read/ReadVariableOpReadVariableOpenc_outer_0/kernel*
_output_shapes
:	?<*
dtype0
x
enc_outer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_nameenc_outer_0/bias
q
$enc_outer_0/bias/Read/ReadVariableOpReadVariableOpenc_outer_0/bias*
_output_shapes
:<*
dtype0
?
enc_outer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<*#
shared_nameenc_outer_1/kernel
z
&enc_outer_1/kernel/Read/ReadVariableOpReadVariableOpenc_outer_1/kernel*
_output_shapes
:	?<*
dtype0
x
enc_outer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_nameenc_outer_1/bias
q
$enc_outer_1/bias/Read/ReadVariableOpReadVariableOpenc_outer_1/bias*
_output_shapes
:<*
dtype0
?
enc_outer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<*#
shared_nameenc_outer_2/kernel
z
&enc_outer_2/kernel/Read/ReadVariableOpReadVariableOpenc_outer_2/kernel*
_output_shapes
:	?<*
dtype0
x
enc_outer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_nameenc_outer_2/bias
q
$enc_outer_2/bias/Read/ReadVariableOpReadVariableOpenc_outer_2/bias*
_output_shapes
:<*
dtype0
?
enc_middle_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*$
shared_nameenc_middle_0/kernel
{
'enc_middle_0/kernel/Read/ReadVariableOpReadVariableOpenc_middle_0/kernel*
_output_shapes

:<2*
dtype0
z
enc_middle_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameenc_middle_0/bias
s
%enc_middle_0/bias/Read/ReadVariableOpReadVariableOpenc_middle_0/bias*
_output_shapes
:2*
dtype0
?
enc_middle_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*$
shared_nameenc_middle_1/kernel
{
'enc_middle_1/kernel/Read/ReadVariableOpReadVariableOpenc_middle_1/kernel*
_output_shapes

:<2*
dtype0
z
enc_middle_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameenc_middle_1/bias
s
%enc_middle_1/bias/Read/ReadVariableOpReadVariableOpenc_middle_1/bias*
_output_shapes
:2*
dtype0
?
enc_middle_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*$
shared_nameenc_middle_2/kernel
{
'enc_middle_2/kernel/Read/ReadVariableOpReadVariableOpenc_middle_2/kernel*
_output_shapes

:<2*
dtype0
z
enc_middle_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameenc_middle_2/bias
s
%enc_middle_2/bias/Read/ReadVariableOpReadVariableOpenc_middle_2/bias*
_output_shapes
:2*
dtype0
?
enc_inner_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*#
shared_nameenc_inner_0/kernel
y
&enc_inner_0/kernel/Read/ReadVariableOpReadVariableOpenc_inner_0/kernel*
_output_shapes

:2(*
dtype0
x
enc_inner_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameenc_inner_0/bias
q
$enc_inner_0/bias/Read/ReadVariableOpReadVariableOpenc_inner_0/bias*
_output_shapes
:(*
dtype0
?
enc_inner_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*#
shared_nameenc_inner_1/kernel
y
&enc_inner_1/kernel/Read/ReadVariableOpReadVariableOpenc_inner_1/kernel*
_output_shapes

:2(*
dtype0
x
enc_inner_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameenc_inner_1/bias
q
$enc_inner_1/bias/Read/ReadVariableOpReadVariableOpenc_inner_1/bias*
_output_shapes
:(*
dtype0
?
enc_inner_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*#
shared_nameenc_inner_2/kernel
y
&enc_inner_2/kernel/Read/ReadVariableOpReadVariableOpenc_inner_2/kernel*
_output_shapes

:2(*
dtype0
x
enc_inner_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameenc_inner_2/bias
q
$enc_inner_2/bias/Read/ReadVariableOpReadVariableOpenc_inner_2/bias*
_output_shapes
:(*
dtype0
|
channel_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namechannel_0/kernel
u
$channel_0/kernel/Read/ReadVariableOpReadVariableOpchannel_0/kernel*
_output_shapes

:(*
dtype0
t
channel_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechannel_0/bias
m
"channel_0/bias/Read/ReadVariableOpReadVariableOpchannel_0/bias*
_output_shapes
:*
dtype0
|
channel_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namechannel_1/kernel
u
$channel_1/kernel/Read/ReadVariableOpReadVariableOpchannel_1/kernel*
_output_shapes

:(*
dtype0
t
channel_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechannel_1/bias
m
"channel_1/bias/Read/ReadVariableOpReadVariableOpchannel_1/bias*
_output_shapes
:*
dtype0
|
channel_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namechannel_2/kernel
u
$channel_2/kernel/Read/ReadVariableOpReadVariableOpchannel_2/kernel*
_output_shapes

:(*
dtype0
t
channel_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechannel_2/bias
m
"channel_2/bias/Read/ReadVariableOpReadVariableOpchannel_2/bias*
_output_shapes
:*
dtype0
?
dec_inner_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedec_inner_0/kernel
y
&dec_inner_0/kernel/Read/ReadVariableOpReadVariableOpdec_inner_0/kernel*
_output_shapes

:(*
dtype0
x
dec_inner_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedec_inner_0/bias
q
$dec_inner_0/bias/Read/ReadVariableOpReadVariableOpdec_inner_0/bias*
_output_shapes
:(*
dtype0
?
dec_inner_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedec_inner_1/kernel
y
&dec_inner_1/kernel/Read/ReadVariableOpReadVariableOpdec_inner_1/kernel*
_output_shapes

:(*
dtype0
x
dec_inner_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedec_inner_1/bias
q
$dec_inner_1/bias/Read/ReadVariableOpReadVariableOpdec_inner_1/bias*
_output_shapes
:(*
dtype0
?
dec_inner_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedec_inner_2/kernel
y
&dec_inner_2/kernel/Read/ReadVariableOpReadVariableOpdec_inner_2/kernel*
_output_shapes

:(*
dtype0
x
dec_inner_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedec_inner_2/bias
q
$dec_inner_2/bias/Read/ReadVariableOpReadVariableOpdec_inner_2/bias*
_output_shapes
:(*
dtype0
?
dec_middle_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*$
shared_namedec_middle_0/kernel
{
'dec_middle_0/kernel/Read/ReadVariableOpReadVariableOpdec_middle_0/kernel*
_output_shapes

:(<*
dtype0
z
dec_middle_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*"
shared_namedec_middle_0/bias
s
%dec_middle_0/bias/Read/ReadVariableOpReadVariableOpdec_middle_0/bias*
_output_shapes
:<*
dtype0
?
dec_middle_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*$
shared_namedec_middle_1/kernel
{
'dec_middle_1/kernel/Read/ReadVariableOpReadVariableOpdec_middle_1/kernel*
_output_shapes

:(<*
dtype0
z
dec_middle_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*"
shared_namedec_middle_1/bias
s
%dec_middle_1/bias/Read/ReadVariableOpReadVariableOpdec_middle_1/bias*
_output_shapes
:<*
dtype0
?
dec_middle_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*$
shared_namedec_middle_2/kernel
{
'dec_middle_2/kernel/Read/ReadVariableOpReadVariableOpdec_middle_2/kernel*
_output_shapes

:(<*
dtype0
z
dec_middle_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*"
shared_namedec_middle_2/bias
s
%dec_middle_2/bias/Read/ReadVariableOpReadVariableOpdec_middle_2/bias*
_output_shapes
:<*
dtype0
?
dec_outer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*#
shared_namedec_outer_0/kernel
y
&dec_outer_0/kernel/Read/ReadVariableOpReadVariableOpdec_outer_0/kernel*
_output_shapes

:<<*
dtype0
x
dec_outer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_namedec_outer_0/bias
q
$dec_outer_0/bias/Read/ReadVariableOpReadVariableOpdec_outer_0/bias*
_output_shapes
:<*
dtype0
?
dec_outer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*#
shared_namedec_outer_1/kernel
y
&dec_outer_1/kernel/Read/ReadVariableOpReadVariableOpdec_outer_1/kernel*
_output_shapes

:<<*
dtype0
x
dec_outer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_namedec_outer_1/bias
q
$dec_outer_1/bias/Read/ReadVariableOpReadVariableOpdec_outer_1/bias*
_output_shapes
:<*
dtype0
?
dec_outer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*#
shared_namedec_outer_2/kernel
y
&dec_outer_2/kernel/Read/ReadVariableOpReadVariableOpdec_outer_2/kernel*
_output_shapes

:<<*
dtype0
x
dec_outer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_namedec_outer_2/bias
q
$dec_outer_2/bias/Read/ReadVariableOpReadVariableOpdec_outer_2/bias*
_output_shapes
:<*
dtype0
?
dec_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*"
shared_namedec_output/kernel
y
%dec_output/kernel/Read/ReadVariableOpReadVariableOpdec_output/kernel* 
_output_shapes
:
??*
dtype0
w
dec_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namedec_output/bias
p
#dec_output/bias/Read/ReadVariableOpReadVariableOpdec_output/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/enc_outer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_0/kernel/m
?
-Adam/enc_outer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_0/kernel/m*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_0/bias/m

+Adam/enc_outer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_0/bias/m*
_output_shapes
:<*
dtype0
?
Adam/enc_outer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_1/kernel/m
?
-Adam/enc_outer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_1/kernel/m*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_1/bias/m

+Adam/enc_outer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_1/bias/m*
_output_shapes
:<*
dtype0
?
Adam/enc_outer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_2/kernel/m
?
-Adam/enc_outer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_2/kernel/m*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_2/bias/m

+Adam/enc_outer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_2/bias/m*
_output_shapes
:<*
dtype0
?
Adam/enc_middle_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_0/kernel/m
?
.Adam/enc_middle_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_0/kernel/m*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_0/bias/m
?
,Adam/enc_middle_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_0/bias/m*
_output_shapes
:2*
dtype0
?
Adam/enc_middle_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_1/kernel/m
?
.Adam/enc_middle_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_1/kernel/m*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_1/bias/m
?
,Adam/enc_middle_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_1/bias/m*
_output_shapes
:2*
dtype0
?
Adam/enc_middle_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_2/kernel/m
?
.Adam/enc_middle_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_2/kernel/m*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_2/bias/m
?
,Adam/enc_middle_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_2/bias/m*
_output_shapes
:2*
dtype0
?
Adam/enc_inner_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_0/kernel/m
?
-Adam/enc_inner_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_0/kernel/m*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_0/bias/m

+Adam/enc_inner_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_0/bias/m*
_output_shapes
:(*
dtype0
?
Adam/enc_inner_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_1/kernel/m
?
-Adam/enc_inner_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_1/kernel/m*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_1/bias/m

+Adam/enc_inner_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_1/bias/m*
_output_shapes
:(*
dtype0
?
Adam/enc_inner_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_2/kernel/m
?
-Adam/enc_inner_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_2/kernel/m*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_2/bias/m

+Adam/enc_inner_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_2/bias/m*
_output_shapes
:(*
dtype0
?
Adam/channel_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_0/kernel/m
?
+Adam/channel_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/channel_0/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/channel_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_0/bias/m
{
)Adam/channel_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/channel_0/bias/m*
_output_shapes
:*
dtype0
?
Adam/channel_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_1/kernel/m
?
+Adam/channel_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/channel_1/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/channel_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_1/bias/m
{
)Adam/channel_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/channel_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/channel_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_2/kernel/m
?
+Adam/channel_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/channel_2/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/channel_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_2/bias/m
{
)Adam/channel_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/channel_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dec_inner_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_0/kernel/m
?
-Adam/dec_inner_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_0/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_0/bias/m

+Adam/dec_inner_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_0/bias/m*
_output_shapes
:(*
dtype0
?
Adam/dec_inner_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_1/kernel/m
?
-Adam/dec_inner_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_1/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_1/bias/m

+Adam/dec_inner_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_1/bias/m*
_output_shapes
:(*
dtype0
?
Adam/dec_inner_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_2/kernel/m
?
-Adam/dec_inner_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_2/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_2/bias/m

+Adam/dec_inner_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_2/bias/m*
_output_shapes
:(*
dtype0
?
Adam/dec_middle_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_0/kernel/m
?
.Adam/dec_middle_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_0/kernel/m*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_0/bias/m
?
,Adam/dec_middle_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_0/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dec_middle_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_1/kernel/m
?
.Adam/dec_middle_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_1/kernel/m*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_1/bias/m
?
,Adam/dec_middle_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_1/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dec_middle_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_2/kernel/m
?
.Adam/dec_middle_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_2/kernel/m*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_2/bias/m
?
,Adam/dec_middle_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_2/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dec_outer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_0/kernel/m
?
-Adam/dec_outer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_0/kernel/m*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_0/bias/m

+Adam/dec_outer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_0/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dec_outer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_1/kernel/m
?
-Adam/dec_outer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_1/kernel/m*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_1/bias/m

+Adam/dec_outer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_1/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dec_outer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_2/kernel/m
?
-Adam/dec_outer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_2/kernel/m*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_2/bias/m

+Adam/dec_outer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_2/bias/m*
_output_shapes
:<*
dtype0
?
Adam/dec_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/dec_output/kernel/m
?
,Adam/dec_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_output/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dec_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/dec_output/bias/m
~
*Adam/dec_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_output/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/enc_outer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_0/kernel/v
?
-Adam/enc_outer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_0/kernel/v*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_0/bias/v

+Adam/enc_outer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_0/bias/v*
_output_shapes
:<*
dtype0
?
Adam/enc_outer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_1/kernel/v
?
-Adam/enc_outer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_1/kernel/v*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_1/bias/v

+Adam/enc_outer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_1/bias/v*
_output_shapes
:<*
dtype0
?
Adam/enc_outer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_2/kernel/v
?
-Adam/enc_outer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_2/kernel/v*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_2/bias/v

+Adam/enc_outer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_2/bias/v*
_output_shapes
:<*
dtype0
?
Adam/enc_middle_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_0/kernel/v
?
.Adam/enc_middle_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_0/kernel/v*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_0/bias/v
?
,Adam/enc_middle_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_0/bias/v*
_output_shapes
:2*
dtype0
?
Adam/enc_middle_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_1/kernel/v
?
.Adam/enc_middle_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_1/kernel/v*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_1/bias/v
?
,Adam/enc_middle_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_1/bias/v*
_output_shapes
:2*
dtype0
?
Adam/enc_middle_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_2/kernel/v
?
.Adam/enc_middle_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_2/kernel/v*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_2/bias/v
?
,Adam/enc_middle_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_2/bias/v*
_output_shapes
:2*
dtype0
?
Adam/enc_inner_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_0/kernel/v
?
-Adam/enc_inner_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_0/kernel/v*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_0/bias/v

+Adam/enc_inner_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_0/bias/v*
_output_shapes
:(*
dtype0
?
Adam/enc_inner_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_1/kernel/v
?
-Adam/enc_inner_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_1/kernel/v*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_1/bias/v

+Adam/enc_inner_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_1/bias/v*
_output_shapes
:(*
dtype0
?
Adam/enc_inner_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_2/kernel/v
?
-Adam/enc_inner_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_2/kernel/v*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_2/bias/v

+Adam/enc_inner_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_2/bias/v*
_output_shapes
:(*
dtype0
?
Adam/channel_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_0/kernel/v
?
+Adam/channel_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/channel_0/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/channel_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_0/bias/v
{
)Adam/channel_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/channel_0/bias/v*
_output_shapes
:*
dtype0
?
Adam/channel_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_1/kernel/v
?
+Adam/channel_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/channel_1/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/channel_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_1/bias/v
{
)Adam/channel_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/channel_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/channel_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_2/kernel/v
?
+Adam/channel_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/channel_2/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/channel_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_2/bias/v
{
)Adam/channel_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/channel_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dec_inner_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_0/kernel/v
?
-Adam/dec_inner_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_0/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_0/bias/v

+Adam/dec_inner_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_0/bias/v*
_output_shapes
:(*
dtype0
?
Adam/dec_inner_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_1/kernel/v
?
-Adam/dec_inner_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_1/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_1/bias/v

+Adam/dec_inner_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_1/bias/v*
_output_shapes
:(*
dtype0
?
Adam/dec_inner_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_2/kernel/v
?
-Adam/dec_inner_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_2/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_2/bias/v

+Adam/dec_inner_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_2/bias/v*
_output_shapes
:(*
dtype0
?
Adam/dec_middle_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_0/kernel/v
?
.Adam/dec_middle_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_0/kernel/v*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_0/bias/v
?
,Adam/dec_middle_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_0/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dec_middle_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_1/kernel/v
?
.Adam/dec_middle_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_1/kernel/v*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_1/bias/v
?
,Adam/dec_middle_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_1/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dec_middle_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_2/kernel/v
?
.Adam/dec_middle_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_2/kernel/v*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_2/bias/v
?
,Adam/dec_middle_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_2/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dec_outer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_0/kernel/v
?
-Adam/dec_outer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_0/kernel/v*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_0/bias/v

+Adam/dec_outer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_0/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dec_outer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_1/kernel/v
?
-Adam/dec_outer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_1/kernel/v*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_1/bias/v

+Adam/dec_outer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_1/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dec_outer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_2/kernel/v
?
-Adam/dec_outer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_2/kernel/v*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_2/bias/v

+Adam/dec_outer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_2/bias/v*
_output_shapes
:<*
dtype0
?
Adam/dec_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/dec_output/kernel/v
?
,Adam/dec_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_output/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dec_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/dec_output/bias/v
~
*Adam/dec_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_output/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
encoder
decoder
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
layer_with_weights-7
layer-8
layer_with_weights-8
layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
 layer_with_weights-3
 layer-6
!layer_with_weights-4
!layer-7
"layer_with_weights-5
"layer-8
#layer_with_weights-6
#layer-9
$layer_with_weights-7
$layer-10
%layer_with_weights-8
%layer-11
&layer-12
'layer_with_weights-9
'layer-13
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?
,iter

-beta_1

.beta_2
	/decay
0learning_rate1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
I24
J25
K26
L27
M28
N29
O30
P31
Q32
R33
S34
T35
U36
V37
W38
X39
Y40
Z41
[42
\43
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
I24
J25
K26
L27
M28
N29
O30
P31
Q32
R33
S34
T35
U36
V37
W38
X39
Y40
Z41
[42
\43
 
?
	variables
]metrics
^non_trainable_variables
_layer_regularization_losses
trainable_variables
`layer_metrics
regularization_losses

alayers
 
 
h

1kernel
2bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
h

3kernel
4bias
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
h

5kernel
6bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
h

7kernel
8bias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
h

9kernel
:bias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
h

;kernel
<bias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
h

=kernel
>bias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
j

?kernel
@bias
~trainable_variables
	variables
?regularization_losses
?	keras_api
l

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ckernel
Dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Gkernel
Hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
 
?
	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
trainable_variables
?layer_metrics
regularization_losses
?layers
 
 
 
l

Ikernel
Jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Kkernel
Lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Mkernel
Nbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Qkernel
Rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Skernel
Tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ukernel
Vbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Wkernel
Xbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ykernel
Zbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api
l

[kernel
\bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
?
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
 
?
(	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
)trainable_variables
?layer_metrics
*regularization_losses
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_outer_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEenc_outer_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_outer_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEenc_outer_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_outer_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEenc_outer_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_middle_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_middle_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_middle_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_middle_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEenc_middle_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_middle_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_0/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_0/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_0/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_0/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_0/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_0/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_1/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_1/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_0/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_0/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_1/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_1/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_2/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_2/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_0/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_0/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_1/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_1/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_2/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_2/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_output/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdec_output/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE

?0
 
 
 

0
1

10
21

10
21
 
?
btrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
c	variables
?layer_metrics
dregularization_losses
?layers

30
41

30
41
 
?
ftrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
g	variables
?layer_metrics
hregularization_losses
?layers

50
61

50
61
 
?
jtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
k	variables
?layer_metrics
lregularization_losses
?layers

70
81

70
81
 
?
ntrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
o	variables
?layer_metrics
pregularization_losses
?layers

90
:1

90
:1
 
?
rtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
s	variables
?layer_metrics
tregularization_losses
?layers

;0
<1

;0
<1
 
?
vtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
w	variables
?layer_metrics
xregularization_losses
?layers

=0
>1

=0
>1
 
?
ztrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
{	variables
?layer_metrics
|regularization_losses
?layers

?0
@1

?0
@1
 
?
~trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
	variables
?layer_metrics
?regularization_losses
?layers

A0
B1

A0
B1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

C0
D1

C0
D1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

E0
F1

E0
F1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

G0
H1

G0
H1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
 
 
 
 
^
	0

1
2
3
4
5
6
7
8
9
10
11
12

I0
J1

I0
J1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

K0
L1

K0
L1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

M0
N1

M0
N1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

O0
P1

O0
P1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

Q0
R1

Q0
R1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

S0
T1

S0
T1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

U0
V1

U0
V1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

W0
X1

W0
X1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers

Y0
Z1

Y0
Z1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
 

[0
\1

[0
\1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
 
 
 
 
f
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
qo
VARIABLE_VALUEAdam/enc_outer_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_2/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_2/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_0/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_0/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_1/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_1/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_2/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_2/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_0/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_0/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_1/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_1/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_2/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_2/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_0/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_0/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_1/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_1/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_2/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_2/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_1/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_1/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_2/kernel/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_2/bias/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_1/kernel/mCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_1/bias/mCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_2/kernel/mCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_2/bias/mCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/mCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/mCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_2/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_2/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_0/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_0/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_1/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_1/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_2/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_2/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_0/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_0/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_1/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_1/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_2/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_2/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_0/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_0/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_1/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_1/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_2/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_2/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_1/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_1/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_2/kernel/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_2/bias/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_1/kernel/vCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_1/bias/vCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_2/kernel/vCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_2/bias/vCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/vCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/vCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1enc_outer_2/kernelenc_outer_2/biasenc_outer_1/kernelenc_outer_1/biasenc_outer_0/kernelenc_outer_0/biasenc_middle_2/kernelenc_middle_2/biasenc_middle_1/kernelenc_middle_1/biasenc_middle_0/kernelenc_middle_0/biasenc_inner_2/kernelenc_inner_2/biasenc_inner_1/kernelenc_inner_1/biasenc_inner_0/kernelenc_inner_0/biaschannel_2/kernelchannel_2/biaschannel_1/kernelchannel_1/biaschannel_0/kernelchannel_0/biasdec_inner_2/kerneldec_inner_2/biasdec_inner_1/kerneldec_inner_1/biasdec_inner_0/kerneldec_inner_0/biasdec_middle_2/kerneldec_middle_2/biasdec_middle_1/kerneldec_middle_1/biasdec_middle_0/kerneldec_middle_0/biasdec_outer_0/kerneldec_outer_0/biasdec_outer_1/kerneldec_outer_1/biasdec_outer_2/kerneldec_outer_2/biasdec_output/kerneldec_output/bias*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_238508
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?2
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp&enc_outer_0/kernel/Read/ReadVariableOp$enc_outer_0/bias/Read/ReadVariableOp&enc_outer_1/kernel/Read/ReadVariableOp$enc_outer_1/bias/Read/ReadVariableOp&enc_outer_2/kernel/Read/ReadVariableOp$enc_outer_2/bias/Read/ReadVariableOp'enc_middle_0/kernel/Read/ReadVariableOp%enc_middle_0/bias/Read/ReadVariableOp'enc_middle_1/kernel/Read/ReadVariableOp%enc_middle_1/bias/Read/ReadVariableOp'enc_middle_2/kernel/Read/ReadVariableOp%enc_middle_2/bias/Read/ReadVariableOp&enc_inner_0/kernel/Read/ReadVariableOp$enc_inner_0/bias/Read/ReadVariableOp&enc_inner_1/kernel/Read/ReadVariableOp$enc_inner_1/bias/Read/ReadVariableOp&enc_inner_2/kernel/Read/ReadVariableOp$enc_inner_2/bias/Read/ReadVariableOp$channel_0/kernel/Read/ReadVariableOp"channel_0/bias/Read/ReadVariableOp$channel_1/kernel/Read/ReadVariableOp"channel_1/bias/Read/ReadVariableOp$channel_2/kernel/Read/ReadVariableOp"channel_2/bias/Read/ReadVariableOp&dec_inner_0/kernel/Read/ReadVariableOp$dec_inner_0/bias/Read/ReadVariableOp&dec_inner_1/kernel/Read/ReadVariableOp$dec_inner_1/bias/Read/ReadVariableOp&dec_inner_2/kernel/Read/ReadVariableOp$dec_inner_2/bias/Read/ReadVariableOp'dec_middle_0/kernel/Read/ReadVariableOp%dec_middle_0/bias/Read/ReadVariableOp'dec_middle_1/kernel/Read/ReadVariableOp%dec_middle_1/bias/Read/ReadVariableOp'dec_middle_2/kernel/Read/ReadVariableOp%dec_middle_2/bias/Read/ReadVariableOp&dec_outer_0/kernel/Read/ReadVariableOp$dec_outer_0/bias/Read/ReadVariableOp&dec_outer_1/kernel/Read/ReadVariableOp$dec_outer_1/bias/Read/ReadVariableOp&dec_outer_2/kernel/Read/ReadVariableOp$dec_outer_2/bias/Read/ReadVariableOp%dec_output/kernel/Read/ReadVariableOp#dec_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/enc_outer_0/kernel/m/Read/ReadVariableOp+Adam/enc_outer_0/bias/m/Read/ReadVariableOp-Adam/enc_outer_1/kernel/m/Read/ReadVariableOp+Adam/enc_outer_1/bias/m/Read/ReadVariableOp-Adam/enc_outer_2/kernel/m/Read/ReadVariableOp+Adam/enc_outer_2/bias/m/Read/ReadVariableOp.Adam/enc_middle_0/kernel/m/Read/ReadVariableOp,Adam/enc_middle_0/bias/m/Read/ReadVariableOp.Adam/enc_middle_1/kernel/m/Read/ReadVariableOp,Adam/enc_middle_1/bias/m/Read/ReadVariableOp.Adam/enc_middle_2/kernel/m/Read/ReadVariableOp,Adam/enc_middle_2/bias/m/Read/ReadVariableOp-Adam/enc_inner_0/kernel/m/Read/ReadVariableOp+Adam/enc_inner_0/bias/m/Read/ReadVariableOp-Adam/enc_inner_1/kernel/m/Read/ReadVariableOp+Adam/enc_inner_1/bias/m/Read/ReadVariableOp-Adam/enc_inner_2/kernel/m/Read/ReadVariableOp+Adam/enc_inner_2/bias/m/Read/ReadVariableOp+Adam/channel_0/kernel/m/Read/ReadVariableOp)Adam/channel_0/bias/m/Read/ReadVariableOp+Adam/channel_1/kernel/m/Read/ReadVariableOp)Adam/channel_1/bias/m/Read/ReadVariableOp+Adam/channel_2/kernel/m/Read/ReadVariableOp)Adam/channel_2/bias/m/Read/ReadVariableOp-Adam/dec_inner_0/kernel/m/Read/ReadVariableOp+Adam/dec_inner_0/bias/m/Read/ReadVariableOp-Adam/dec_inner_1/kernel/m/Read/ReadVariableOp+Adam/dec_inner_1/bias/m/Read/ReadVariableOp-Adam/dec_inner_2/kernel/m/Read/ReadVariableOp+Adam/dec_inner_2/bias/m/Read/ReadVariableOp.Adam/dec_middle_0/kernel/m/Read/ReadVariableOp,Adam/dec_middle_0/bias/m/Read/ReadVariableOp.Adam/dec_middle_1/kernel/m/Read/ReadVariableOp,Adam/dec_middle_1/bias/m/Read/ReadVariableOp.Adam/dec_middle_2/kernel/m/Read/ReadVariableOp,Adam/dec_middle_2/bias/m/Read/ReadVariableOp-Adam/dec_outer_0/kernel/m/Read/ReadVariableOp+Adam/dec_outer_0/bias/m/Read/ReadVariableOp-Adam/dec_outer_1/kernel/m/Read/ReadVariableOp+Adam/dec_outer_1/bias/m/Read/ReadVariableOp-Adam/dec_outer_2/kernel/m/Read/ReadVariableOp+Adam/dec_outer_2/bias/m/Read/ReadVariableOp,Adam/dec_output/kernel/m/Read/ReadVariableOp*Adam/dec_output/bias/m/Read/ReadVariableOp-Adam/enc_outer_0/kernel/v/Read/ReadVariableOp+Adam/enc_outer_0/bias/v/Read/ReadVariableOp-Adam/enc_outer_1/kernel/v/Read/ReadVariableOp+Adam/enc_outer_1/bias/v/Read/ReadVariableOp-Adam/enc_outer_2/kernel/v/Read/ReadVariableOp+Adam/enc_outer_2/bias/v/Read/ReadVariableOp.Adam/enc_middle_0/kernel/v/Read/ReadVariableOp,Adam/enc_middle_0/bias/v/Read/ReadVariableOp.Adam/enc_middle_1/kernel/v/Read/ReadVariableOp,Adam/enc_middle_1/bias/v/Read/ReadVariableOp.Adam/enc_middle_2/kernel/v/Read/ReadVariableOp,Adam/enc_middle_2/bias/v/Read/ReadVariableOp-Adam/enc_inner_0/kernel/v/Read/ReadVariableOp+Adam/enc_inner_0/bias/v/Read/ReadVariableOp-Adam/enc_inner_1/kernel/v/Read/ReadVariableOp+Adam/enc_inner_1/bias/v/Read/ReadVariableOp-Adam/enc_inner_2/kernel/v/Read/ReadVariableOp+Adam/enc_inner_2/bias/v/Read/ReadVariableOp+Adam/channel_0/kernel/v/Read/ReadVariableOp)Adam/channel_0/bias/v/Read/ReadVariableOp+Adam/channel_1/kernel/v/Read/ReadVariableOp)Adam/channel_1/bias/v/Read/ReadVariableOp+Adam/channel_2/kernel/v/Read/ReadVariableOp)Adam/channel_2/bias/v/Read/ReadVariableOp-Adam/dec_inner_0/kernel/v/Read/ReadVariableOp+Adam/dec_inner_0/bias/v/Read/ReadVariableOp-Adam/dec_inner_1/kernel/v/Read/ReadVariableOp+Adam/dec_inner_1/bias/v/Read/ReadVariableOp-Adam/dec_inner_2/kernel/v/Read/ReadVariableOp+Adam/dec_inner_2/bias/v/Read/ReadVariableOp.Adam/dec_middle_0/kernel/v/Read/ReadVariableOp,Adam/dec_middle_0/bias/v/Read/ReadVariableOp.Adam/dec_middle_1/kernel/v/Read/ReadVariableOp,Adam/dec_middle_1/bias/v/Read/ReadVariableOp.Adam/dec_middle_2/kernel/v/Read/ReadVariableOp,Adam/dec_middle_2/bias/v/Read/ReadVariableOp-Adam/dec_outer_0/kernel/v/Read/ReadVariableOp+Adam/dec_outer_0/bias/v/Read/ReadVariableOp-Adam/dec_outer_1/kernel/v/Read/ReadVariableOp+Adam/dec_outer_1/bias/v/Read/ReadVariableOp-Adam/dec_outer_2/kernel/v/Read/ReadVariableOp+Adam/dec_outer_2/bias/v/Read/ReadVariableOp,Adam/dec_output/kernel/v/Read/ReadVariableOp*Adam/dec_output/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *(
f#R!
__inference__traced_save_240438
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateenc_outer_0/kernelenc_outer_0/biasenc_outer_1/kernelenc_outer_1/biasenc_outer_2/kernelenc_outer_2/biasenc_middle_0/kernelenc_middle_0/biasenc_middle_1/kernelenc_middle_1/biasenc_middle_2/kernelenc_middle_2/biasenc_inner_0/kernelenc_inner_0/biasenc_inner_1/kernelenc_inner_1/biasenc_inner_2/kernelenc_inner_2/biaschannel_0/kernelchannel_0/biaschannel_1/kernelchannel_1/biaschannel_2/kernelchannel_2/biasdec_inner_0/kerneldec_inner_0/biasdec_inner_1/kerneldec_inner_1/biasdec_inner_2/kerneldec_inner_2/biasdec_middle_0/kerneldec_middle_0/biasdec_middle_1/kerneldec_middle_1/biasdec_middle_2/kerneldec_middle_2/biasdec_outer_0/kerneldec_outer_0/biasdec_outer_1/kerneldec_outer_1/biasdec_outer_2/kerneldec_outer_2/biasdec_output/kerneldec_output/biastotalcountAdam/enc_outer_0/kernel/mAdam/enc_outer_0/bias/mAdam/enc_outer_1/kernel/mAdam/enc_outer_1/bias/mAdam/enc_outer_2/kernel/mAdam/enc_outer_2/bias/mAdam/enc_middle_0/kernel/mAdam/enc_middle_0/bias/mAdam/enc_middle_1/kernel/mAdam/enc_middle_1/bias/mAdam/enc_middle_2/kernel/mAdam/enc_middle_2/bias/mAdam/enc_inner_0/kernel/mAdam/enc_inner_0/bias/mAdam/enc_inner_1/kernel/mAdam/enc_inner_1/bias/mAdam/enc_inner_2/kernel/mAdam/enc_inner_2/bias/mAdam/channel_0/kernel/mAdam/channel_0/bias/mAdam/channel_1/kernel/mAdam/channel_1/bias/mAdam/channel_2/kernel/mAdam/channel_2/bias/mAdam/dec_inner_0/kernel/mAdam/dec_inner_0/bias/mAdam/dec_inner_1/kernel/mAdam/dec_inner_1/bias/mAdam/dec_inner_2/kernel/mAdam/dec_inner_2/bias/mAdam/dec_middle_0/kernel/mAdam/dec_middle_0/bias/mAdam/dec_middle_1/kernel/mAdam/dec_middle_1/bias/mAdam/dec_middle_2/kernel/mAdam/dec_middle_2/bias/mAdam/dec_outer_0/kernel/mAdam/dec_outer_0/bias/mAdam/dec_outer_1/kernel/mAdam/dec_outer_1/bias/mAdam/dec_outer_2/kernel/mAdam/dec_outer_2/bias/mAdam/dec_output/kernel/mAdam/dec_output/bias/mAdam/enc_outer_0/kernel/vAdam/enc_outer_0/bias/vAdam/enc_outer_1/kernel/vAdam/enc_outer_1/bias/vAdam/enc_outer_2/kernel/vAdam/enc_outer_2/bias/vAdam/enc_middle_0/kernel/vAdam/enc_middle_0/bias/vAdam/enc_middle_1/kernel/vAdam/enc_middle_1/bias/vAdam/enc_middle_2/kernel/vAdam/enc_middle_2/bias/vAdam/enc_inner_0/kernel/vAdam/enc_inner_0/bias/vAdam/enc_inner_1/kernel/vAdam/enc_inner_1/bias/vAdam/enc_inner_2/kernel/vAdam/enc_inner_2/bias/vAdam/channel_0/kernel/vAdam/channel_0/bias/vAdam/channel_1/kernel/vAdam/channel_1/bias/vAdam/channel_2/kernel/vAdam/channel_2/bias/vAdam/dec_inner_0/kernel/vAdam/dec_inner_0/bias/vAdam/dec_inner_1/kernel/vAdam/dec_inner_1/bias/vAdam/dec_inner_2/kernel/vAdam/dec_inner_2/bias/vAdam/dec_middle_0/kernel/vAdam/dec_middle_0/bias/vAdam/dec_middle_1/kernel/vAdam/dec_middle_1/bias/vAdam/dec_middle_2/kernel/vAdam/dec_middle_2/bias/vAdam/dec_outer_0/kernel/vAdam/dec_outer_0/bias/vAdam/dec_outer_1/kernel/vAdam/dec_outer_1/bias/vAdam/dec_outer_2/kernel/vAdam/dec_outer_2/bias/vAdam/dec_output/kernel/vAdam/dec_output/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *+
f&R$
"__inference__traced_restore_240865??
?	
?
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_236448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_enc_inner_0_layer_call_fn_239698

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_2366642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
,__inference_dec_outer_1_layer_call_fn_239958

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_2372832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_236556

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
??
?,
!__inference__wrapped_model_236433
input_1D
@autoencoder_2_model_4_enc_outer_2_matmul_readvariableop_resourceE
Aautoencoder_2_model_4_enc_outer_2_biasadd_readvariableop_resourceD
@autoencoder_2_model_4_enc_outer_1_matmul_readvariableop_resourceE
Aautoencoder_2_model_4_enc_outer_1_biasadd_readvariableop_resourceD
@autoencoder_2_model_4_enc_outer_0_matmul_readvariableop_resourceE
Aautoencoder_2_model_4_enc_outer_0_biasadd_readvariableop_resourceE
Aautoencoder_2_model_4_enc_middle_2_matmul_readvariableop_resourceF
Bautoencoder_2_model_4_enc_middle_2_biasadd_readvariableop_resourceE
Aautoencoder_2_model_4_enc_middle_1_matmul_readvariableop_resourceF
Bautoencoder_2_model_4_enc_middle_1_biasadd_readvariableop_resourceE
Aautoencoder_2_model_4_enc_middle_0_matmul_readvariableop_resourceF
Bautoencoder_2_model_4_enc_middle_0_biasadd_readvariableop_resourceD
@autoencoder_2_model_4_enc_inner_2_matmul_readvariableop_resourceE
Aautoencoder_2_model_4_enc_inner_2_biasadd_readvariableop_resourceD
@autoencoder_2_model_4_enc_inner_1_matmul_readvariableop_resourceE
Aautoencoder_2_model_4_enc_inner_1_biasadd_readvariableop_resourceD
@autoencoder_2_model_4_enc_inner_0_matmul_readvariableop_resourceE
Aautoencoder_2_model_4_enc_inner_0_biasadd_readvariableop_resourceB
>autoencoder_2_model_4_channel_2_matmul_readvariableop_resourceC
?autoencoder_2_model_4_channel_2_biasadd_readvariableop_resourceB
>autoencoder_2_model_4_channel_1_matmul_readvariableop_resourceC
?autoencoder_2_model_4_channel_1_biasadd_readvariableop_resourceB
>autoencoder_2_model_4_channel_0_matmul_readvariableop_resourceC
?autoencoder_2_model_4_channel_0_biasadd_readvariableop_resourceD
@autoencoder_2_model_5_dec_inner_2_matmul_readvariableop_resourceE
Aautoencoder_2_model_5_dec_inner_2_biasadd_readvariableop_resourceD
@autoencoder_2_model_5_dec_inner_1_matmul_readvariableop_resourceE
Aautoencoder_2_model_5_dec_inner_1_biasadd_readvariableop_resourceD
@autoencoder_2_model_5_dec_inner_0_matmul_readvariableop_resourceE
Aautoencoder_2_model_5_dec_inner_0_biasadd_readvariableop_resourceE
Aautoencoder_2_model_5_dec_middle_2_matmul_readvariableop_resourceF
Bautoencoder_2_model_5_dec_middle_2_biasadd_readvariableop_resourceE
Aautoencoder_2_model_5_dec_middle_1_matmul_readvariableop_resourceF
Bautoencoder_2_model_5_dec_middle_1_biasadd_readvariableop_resourceE
Aautoencoder_2_model_5_dec_middle_0_matmul_readvariableop_resourceF
Bautoencoder_2_model_5_dec_middle_0_biasadd_readvariableop_resourceD
@autoencoder_2_model_5_dec_outer_0_matmul_readvariableop_resourceE
Aautoencoder_2_model_5_dec_outer_0_biasadd_readvariableop_resourceD
@autoencoder_2_model_5_dec_outer_1_matmul_readvariableop_resourceE
Aautoencoder_2_model_5_dec_outer_1_biasadd_readvariableop_resourceD
@autoencoder_2_model_5_dec_outer_2_matmul_readvariableop_resourceE
Aautoencoder_2_model_5_dec_outer_2_biasadd_readvariableop_resourceC
?autoencoder_2_model_5_dec_output_matmul_readvariableop_resourceD
@autoencoder_2_model_5_dec_output_biasadd_readvariableop_resource
identity??6autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOp?5autoencoder_2/model_4/channel_0/MatMul/ReadVariableOp?6autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOp?5autoencoder_2/model_4/channel_1/MatMul/ReadVariableOp?6autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOp?5autoencoder_2/model_4/channel_2/MatMul/ReadVariableOp?8autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOp?7autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOp?8autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOp?7autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOp?8autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOp?7autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOp?9autoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOp?8autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOp?9autoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOp?8autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOp?9autoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOp?8autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOp?8autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOp?7autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOp?8autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOp?7autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOp?8autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOp?7autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOp?8autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOp?7autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOp?8autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOp?7autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOp?8autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOp?7autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOp?9autoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOp?8autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOp?9autoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOp?8autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOp?9autoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOp?8autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOp?8autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOp?7autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOp?8autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOp?7autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOp?8autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOp?7autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOp?7autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOp?6autoencoder_2/model_5/dec_output/MatMul/ReadVariableOp?
7autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_4_enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOp?
(autoencoder_2/model_4/enc_outer_2/MatMulMatMulinput_1?autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_2/model_4/enc_outer_2/MatMul?
8autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOp?
)autoencoder_2/model_4/enc_outer_2/BiasAddBiasAdd2autoencoder_2/model_4/enc_outer_2/MatMul:product:0@autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_4/enc_outer_2/BiasAdd?
&autoencoder_2/model_4/enc_outer_2/ReluRelu2autoencoder_2/model_4/enc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_2/model_4/enc_outer_2/Relu?
7autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_4_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOp?
(autoencoder_2/model_4/enc_outer_1/MatMulMatMulinput_1?autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_2/model_4/enc_outer_1/MatMul?
8autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOp?
)autoencoder_2/model_4/enc_outer_1/BiasAddBiasAdd2autoencoder_2/model_4/enc_outer_1/MatMul:product:0@autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_4/enc_outer_1/BiasAdd?
&autoencoder_2/model_4/enc_outer_1/ReluRelu2autoencoder_2/model_4/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_2/model_4/enc_outer_1/Relu?
7autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_4_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOp?
(autoencoder_2/model_4/enc_outer_0/MatMulMatMulinput_1?autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_2/model_4/enc_outer_0/MatMul?
8autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOp?
)autoencoder_2/model_4/enc_outer_0/BiasAddBiasAdd2autoencoder_2/model_4/enc_outer_0/MatMul:product:0@autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_4/enc_outer_0/BiasAdd?
&autoencoder_2/model_4/enc_outer_0/ReluRelu2autoencoder_2/model_4/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_2/model_4/enc_outer_0/Relu?
8autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOp?
)autoencoder_2/model_4/enc_middle_2/MatMulMatMul4autoencoder_2/model_4/enc_outer_2/Relu:activations:0@autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_2/model_4/enc_middle_2/MatMul?
9autoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_model_4_enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOp?
*autoencoder_2/model_4/enc_middle_2/BiasAddBiasAdd3autoencoder_2/model_4/enc_middle_2/MatMul:product:0Aautoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_2/model_4/enc_middle_2/BiasAdd?
'autoencoder_2/model_4/enc_middle_2/ReluRelu3autoencoder_2/model_4/enc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_2/model_4/enc_middle_2/Relu?
8autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOp?
)autoencoder_2/model_4/enc_middle_1/MatMulMatMul4autoencoder_2/model_4/enc_outer_1/Relu:activations:0@autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_2/model_4/enc_middle_1/MatMul?
9autoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_model_4_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOp?
*autoencoder_2/model_4/enc_middle_1/BiasAddBiasAdd3autoencoder_2/model_4/enc_middle_1/MatMul:product:0Aautoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_2/model_4/enc_middle_1/BiasAdd?
'autoencoder_2/model_4/enc_middle_1/ReluRelu3autoencoder_2/model_4/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_2/model_4/enc_middle_1/Relu?
8autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOp?
)autoencoder_2/model_4/enc_middle_0/MatMulMatMul4autoencoder_2/model_4/enc_outer_0/Relu:activations:0@autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_2/model_4/enc_middle_0/MatMul?
9autoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_model_4_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOp?
*autoencoder_2/model_4/enc_middle_0/BiasAddBiasAdd3autoencoder_2/model_4/enc_middle_0/MatMul:product:0Aautoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_2/model_4/enc_middle_0/BiasAdd?
'autoencoder_2/model_4/enc_middle_0/ReluRelu3autoencoder_2/model_4/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_2/model_4/enc_middle_0/Relu?
7autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_4_enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOp?
(autoencoder_2/model_4/enc_inner_2/MatMulMatMul5autoencoder_2/model_4/enc_middle_2/Relu:activations:0?autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_2/model_4/enc_inner_2/MatMul?
8autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOp?
)autoencoder_2/model_4/enc_inner_2/BiasAddBiasAdd2autoencoder_2/model_4/enc_inner_2/MatMul:product:0@autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_2/model_4/enc_inner_2/BiasAdd?
&autoencoder_2/model_4/enc_inner_2/ReluRelu2autoencoder_2/model_4/enc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_2/model_4/enc_inner_2/Relu?
7autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_4_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOp?
(autoencoder_2/model_4/enc_inner_1/MatMulMatMul5autoencoder_2/model_4/enc_middle_1/Relu:activations:0?autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_2/model_4/enc_inner_1/MatMul?
8autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOp?
)autoencoder_2/model_4/enc_inner_1/BiasAddBiasAdd2autoencoder_2/model_4/enc_inner_1/MatMul:product:0@autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_2/model_4/enc_inner_1/BiasAdd?
&autoencoder_2/model_4/enc_inner_1/ReluRelu2autoencoder_2/model_4/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_2/model_4/enc_inner_1/Relu?
7autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_4_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOp?
(autoencoder_2/model_4/enc_inner_0/MatMulMatMul5autoencoder_2/model_4/enc_middle_0/Relu:activations:0?autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_2/model_4/enc_inner_0/MatMul?
8autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_4_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOp?
)autoencoder_2/model_4/enc_inner_0/BiasAddBiasAdd2autoencoder_2/model_4/enc_inner_0/MatMul:product:0@autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_2/model_4/enc_inner_0/BiasAdd?
&autoencoder_2/model_4/enc_inner_0/ReluRelu2autoencoder_2/model_4/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_2/model_4/enc_inner_0/Relu?
5autoencoder_2/model_4/channel_2/MatMul/ReadVariableOpReadVariableOp>autoencoder_2_model_4_channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_2/model_4/channel_2/MatMul/ReadVariableOp?
&autoencoder_2/model_4/channel_2/MatMulMatMul4autoencoder_2/model_4/enc_inner_2/Relu:activations:0=autoencoder_2/model_4/channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_2/model_4/channel_2/MatMul?
6autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_2_model_4_channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOp?
'autoencoder_2/model_4/channel_2/BiasAddBiasAdd0autoencoder_2/model_4/channel_2/MatMul:product:0>autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_2/model_4/channel_2/BiasAdd?
(autoencoder_2/model_4/channel_2/SoftsignSoftsign0autoencoder_2/model_4/channel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_2/model_4/channel_2/Softsign?
5autoencoder_2/model_4/channel_1/MatMul/ReadVariableOpReadVariableOp>autoencoder_2_model_4_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_2/model_4/channel_1/MatMul/ReadVariableOp?
&autoencoder_2/model_4/channel_1/MatMulMatMul4autoencoder_2/model_4/enc_inner_1/Relu:activations:0=autoencoder_2/model_4/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_2/model_4/channel_1/MatMul?
6autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_2_model_4_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOp?
'autoencoder_2/model_4/channel_1/BiasAddBiasAdd0autoencoder_2/model_4/channel_1/MatMul:product:0>autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_2/model_4/channel_1/BiasAdd?
(autoencoder_2/model_4/channel_1/SoftsignSoftsign0autoencoder_2/model_4/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_2/model_4/channel_1/Softsign?
5autoencoder_2/model_4/channel_0/MatMul/ReadVariableOpReadVariableOp>autoencoder_2_model_4_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_2/model_4/channel_0/MatMul/ReadVariableOp?
&autoencoder_2/model_4/channel_0/MatMulMatMul4autoencoder_2/model_4/enc_inner_0/Relu:activations:0=autoencoder_2/model_4/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_2/model_4/channel_0/MatMul?
6autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_2_model_4_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOp?
'autoencoder_2/model_4/channel_0/BiasAddBiasAdd0autoencoder_2/model_4/channel_0/MatMul:product:0>autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_2/model_4/channel_0/BiasAdd?
(autoencoder_2/model_4/channel_0/SoftsignSoftsign0autoencoder_2/model_4/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_2/model_4/channel_0/Softsign?
7autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOp?
(autoencoder_2/model_5/dec_inner_2/MatMulMatMul6autoencoder_2/model_4/channel_2/Softsign:activations:0?autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_2/model_5/dec_inner_2/MatMul?
8autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOp?
)autoencoder_2/model_5/dec_inner_2/BiasAddBiasAdd2autoencoder_2/model_5/dec_inner_2/MatMul:product:0@autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_2/model_5/dec_inner_2/BiasAdd?
&autoencoder_2/model_5/dec_inner_2/ReluRelu2autoencoder_2/model_5/dec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_2/model_5/dec_inner_2/Relu?
7autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOp?
(autoencoder_2/model_5/dec_inner_1/MatMulMatMul6autoencoder_2/model_4/channel_1/Softsign:activations:0?autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_2/model_5/dec_inner_1/MatMul?
8autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOp?
)autoencoder_2/model_5/dec_inner_1/BiasAddBiasAdd2autoencoder_2/model_5/dec_inner_1/MatMul:product:0@autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_2/model_5/dec_inner_1/BiasAdd?
&autoencoder_2/model_5/dec_inner_1/ReluRelu2autoencoder_2/model_5/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_2/model_5/dec_inner_1/Relu?
7autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOp?
(autoencoder_2/model_5/dec_inner_0/MatMulMatMul6autoencoder_2/model_4/channel_0/Softsign:activations:0?autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_2/model_5/dec_inner_0/MatMul?
8autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOp?
)autoencoder_2/model_5/dec_inner_0/BiasAddBiasAdd2autoencoder_2/model_5/dec_inner_0/MatMul:product:0@autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_2/model_5/dec_inner_0/BiasAdd?
&autoencoder_2/model_5/dec_inner_0/ReluRelu2autoencoder_2/model_5/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_2/model_5/dec_inner_0/Relu?
8autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOp?
)autoencoder_2/model_5/dec_middle_2/MatMulMatMul4autoencoder_2/model_5/dec_inner_2/Relu:activations:0@autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_5/dec_middle_2/MatMul?
9autoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_model_5_dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOp?
*autoencoder_2/model_5/dec_middle_2/BiasAddBiasAdd3autoencoder_2/model_5/dec_middle_2/MatMul:product:0Aautoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_2/model_5/dec_middle_2/BiasAdd?
'autoencoder_2/model_5/dec_middle_2/ReluRelu3autoencoder_2/model_5/dec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_2/model_5/dec_middle_2/Relu?
8autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOp?
)autoencoder_2/model_5/dec_middle_1/MatMulMatMul4autoencoder_2/model_5/dec_inner_1/Relu:activations:0@autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_5/dec_middle_1/MatMul?
9autoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_model_5_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOp?
*autoencoder_2/model_5/dec_middle_1/BiasAddBiasAdd3autoencoder_2/model_5/dec_middle_1/MatMul:product:0Aautoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_2/model_5/dec_middle_1/BiasAdd?
'autoencoder_2/model_5/dec_middle_1/ReluRelu3autoencoder_2/model_5/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_2/model_5/dec_middle_1/Relu?
8autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOp?
)autoencoder_2/model_5/dec_middle_0/MatMulMatMul4autoencoder_2/model_5/dec_inner_0/Relu:activations:0@autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_5/dec_middle_0/MatMul?
9autoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_model_5_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOp?
*autoencoder_2/model_5/dec_middle_0/BiasAddBiasAdd3autoencoder_2/model_5/dec_middle_0/MatMul:product:0Aautoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_2/model_5/dec_middle_0/BiasAdd?
'autoencoder_2/model_5/dec_middle_0/ReluRelu3autoencoder_2/model_5/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_2/model_5/dec_middle_0/Relu?
7autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOp?
(autoencoder_2/model_5/dec_outer_0/MatMulMatMul5autoencoder_2/model_5/dec_middle_0/Relu:activations:0?autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_2/model_5/dec_outer_0/MatMul?
8autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOp?
)autoencoder_2/model_5/dec_outer_0/BiasAddBiasAdd2autoencoder_2/model_5/dec_outer_0/MatMul:product:0@autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_5/dec_outer_0/BiasAdd?
&autoencoder_2/model_5/dec_outer_0/ReluRelu2autoencoder_2/model_5/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_2/model_5/dec_outer_0/Relu?
7autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOp?
(autoencoder_2/model_5/dec_outer_1/MatMulMatMul5autoencoder_2/model_5/dec_middle_1/Relu:activations:0?autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_2/model_5/dec_outer_1/MatMul?
8autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOp?
)autoencoder_2/model_5/dec_outer_1/BiasAddBiasAdd2autoencoder_2/model_5/dec_outer_1/MatMul:product:0@autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_5/dec_outer_1/BiasAdd?
&autoencoder_2/model_5/dec_outer_1/ReluRelu2autoencoder_2/model_5/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_2/model_5/dec_outer_1/Relu?
7autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOp?
(autoencoder_2/model_5/dec_outer_2/MatMulMatMul5autoencoder_2/model_5/dec_middle_2/Relu:activations:0?autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_2/model_5/dec_outer_2/MatMul?
8autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_2_model_5_dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOp?
)autoencoder_2/model_5/dec_outer_2/BiasAddBiasAdd2autoencoder_2/model_5/dec_outer_2/MatMul:product:0@autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_2/model_5/dec_outer_2/BiasAdd?
&autoencoder_2/model_5/dec_outer_2/ReluRelu2autoencoder_2/model_5/dec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_2/model_5/dec_outer_2/Relu?
-autoencoder_2/model_5/tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder_2/model_5/tf.concat_1/concat/axis?
(autoencoder_2/model_5/tf.concat_1/concatConcatV24autoencoder_2/model_5/dec_outer_0/Relu:activations:04autoencoder_2/model_5/dec_outer_1/Relu:activations:04autoencoder_2/model_5/dec_outer_2/Relu:activations:06autoencoder_2/model_5/tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2*
(autoencoder_2/model_5/tf.concat_1/concat?
6autoencoder_2/model_5/dec_output/MatMul/ReadVariableOpReadVariableOp?autoencoder_2_model_5_dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6autoencoder_2/model_5/dec_output/MatMul/ReadVariableOp?
'autoencoder_2/model_5/dec_output/MatMulMatMul1autoencoder_2/model_5/tf.concat_1/concat:output:0>autoencoder_2/model_5/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'autoencoder_2/model_5/dec_output/MatMul?
7autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_2_model_5_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOp?
(autoencoder_2/model_5/dec_output/BiasAddBiasAdd1autoencoder_2/model_5/dec_output/MatMul:product:0?autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(autoencoder_2/model_5/dec_output/BiasAdd?
(autoencoder_2/model_5/dec_output/SigmoidSigmoid1autoencoder_2/model_5/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2*
(autoencoder_2/model_5/dec_output/Sigmoid?
IdentityIdentity,autoencoder_2/model_5/dec_output/Sigmoid:y:07^autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOp6^autoencoder_2/model_4/channel_0/MatMul/ReadVariableOp7^autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOp6^autoencoder_2/model_4/channel_1/MatMul/ReadVariableOp7^autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOp6^autoencoder_2/model_4/channel_2/MatMul/ReadVariableOp9^autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOp8^autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOp9^autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOp8^autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOp9^autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOp8^autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOp:^autoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOp9^autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOp:^autoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOp9^autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOp:^autoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOp9^autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOp9^autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOp8^autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOp9^autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOp8^autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOp9^autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOp8^autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOp9^autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOp8^autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOp9^autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOp8^autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOp9^autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOp8^autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOp:^autoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOp9^autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOp:^autoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOp9^autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOp:^autoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOp9^autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOp9^autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOp8^autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOp9^autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOp8^autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOp9^autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOp8^autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOp8^autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOp7^autoencoder_2/model_5/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2p
6autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOp6autoencoder_2/model_4/channel_0/BiasAdd/ReadVariableOp2n
5autoencoder_2/model_4/channel_0/MatMul/ReadVariableOp5autoencoder_2/model_4/channel_0/MatMul/ReadVariableOp2p
6autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOp6autoencoder_2/model_4/channel_1/BiasAdd/ReadVariableOp2n
5autoencoder_2/model_4/channel_1/MatMul/ReadVariableOp5autoencoder_2/model_4/channel_1/MatMul/ReadVariableOp2p
6autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOp6autoencoder_2/model_4/channel_2/BiasAdd/ReadVariableOp2n
5autoencoder_2/model_4/channel_2/MatMul/ReadVariableOp5autoencoder_2/model_4/channel_2/MatMul/ReadVariableOp2t
8autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOp8autoencoder_2/model_4/enc_inner_0/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOp7autoencoder_2/model_4/enc_inner_0/MatMul/ReadVariableOp2t
8autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOp8autoencoder_2/model_4/enc_inner_1/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOp7autoencoder_2/model_4/enc_inner_1/MatMul/ReadVariableOp2t
8autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOp8autoencoder_2/model_4/enc_inner_2/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOp7autoencoder_2/model_4/enc_inner_2/MatMul/ReadVariableOp2v
9autoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOp9autoencoder_2/model_4/enc_middle_0/BiasAdd/ReadVariableOp2t
8autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOp8autoencoder_2/model_4/enc_middle_0/MatMul/ReadVariableOp2v
9autoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOp9autoencoder_2/model_4/enc_middle_1/BiasAdd/ReadVariableOp2t
8autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOp8autoencoder_2/model_4/enc_middle_1/MatMul/ReadVariableOp2v
9autoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOp9autoencoder_2/model_4/enc_middle_2/BiasAdd/ReadVariableOp2t
8autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOp8autoencoder_2/model_4/enc_middle_2/MatMul/ReadVariableOp2t
8autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOp8autoencoder_2/model_4/enc_outer_0/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOp7autoencoder_2/model_4/enc_outer_0/MatMul/ReadVariableOp2t
8autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOp8autoencoder_2/model_4/enc_outer_1/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOp7autoencoder_2/model_4/enc_outer_1/MatMul/ReadVariableOp2t
8autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOp8autoencoder_2/model_4/enc_outer_2/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOp7autoencoder_2/model_4/enc_outer_2/MatMul/ReadVariableOp2t
8autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOp8autoencoder_2/model_5/dec_inner_0/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOp7autoencoder_2/model_5/dec_inner_0/MatMul/ReadVariableOp2t
8autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOp8autoencoder_2/model_5/dec_inner_1/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOp7autoencoder_2/model_5/dec_inner_1/MatMul/ReadVariableOp2t
8autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOp8autoencoder_2/model_5/dec_inner_2/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOp7autoencoder_2/model_5/dec_inner_2/MatMul/ReadVariableOp2v
9autoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOp9autoencoder_2/model_5/dec_middle_0/BiasAdd/ReadVariableOp2t
8autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOp8autoencoder_2/model_5/dec_middle_0/MatMul/ReadVariableOp2v
9autoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOp9autoencoder_2/model_5/dec_middle_1/BiasAdd/ReadVariableOp2t
8autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOp8autoencoder_2/model_5/dec_middle_1/MatMul/ReadVariableOp2v
9autoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOp9autoencoder_2/model_5/dec_middle_2/BiasAdd/ReadVariableOp2t
8autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOp8autoencoder_2/model_5/dec_middle_2/MatMul/ReadVariableOp2t
8autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOp8autoencoder_2/model_5/dec_outer_0/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOp7autoencoder_2/model_5/dec_outer_0/MatMul/ReadVariableOp2t
8autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOp8autoencoder_2/model_5/dec_outer_1/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOp7autoencoder_2/model_5/dec_outer_1/MatMul/ReadVariableOp2t
8autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOp8autoencoder_2/model_5/dec_outer_2/BiasAdd/ReadVariableOp2r
7autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOp7autoencoder_2/model_5/dec_outer_2/MatMul/ReadVariableOp2r
7autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOp7autoencoder_2/model_5/dec_output/BiasAdd/ReadVariableOp2p
6autoencoder_2/model_5/dec_output/MatMul/ReadVariableOp6autoencoder_2/model_5/dec_output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?;
__inference__traced_save_240438
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop1
-savev2_enc_outer_0_kernel_read_readvariableop/
+savev2_enc_outer_0_bias_read_readvariableop1
-savev2_enc_outer_1_kernel_read_readvariableop/
+savev2_enc_outer_1_bias_read_readvariableop1
-savev2_enc_outer_2_kernel_read_readvariableop/
+savev2_enc_outer_2_bias_read_readvariableop2
.savev2_enc_middle_0_kernel_read_readvariableop0
,savev2_enc_middle_0_bias_read_readvariableop2
.savev2_enc_middle_1_kernel_read_readvariableop0
,savev2_enc_middle_1_bias_read_readvariableop2
.savev2_enc_middle_2_kernel_read_readvariableop0
,savev2_enc_middle_2_bias_read_readvariableop1
-savev2_enc_inner_0_kernel_read_readvariableop/
+savev2_enc_inner_0_bias_read_readvariableop1
-savev2_enc_inner_1_kernel_read_readvariableop/
+savev2_enc_inner_1_bias_read_readvariableop1
-savev2_enc_inner_2_kernel_read_readvariableop/
+savev2_enc_inner_2_bias_read_readvariableop/
+savev2_channel_0_kernel_read_readvariableop-
)savev2_channel_0_bias_read_readvariableop/
+savev2_channel_1_kernel_read_readvariableop-
)savev2_channel_1_bias_read_readvariableop/
+savev2_channel_2_kernel_read_readvariableop-
)savev2_channel_2_bias_read_readvariableop1
-savev2_dec_inner_0_kernel_read_readvariableop/
+savev2_dec_inner_0_bias_read_readvariableop1
-savev2_dec_inner_1_kernel_read_readvariableop/
+savev2_dec_inner_1_bias_read_readvariableop1
-savev2_dec_inner_2_kernel_read_readvariableop/
+savev2_dec_inner_2_bias_read_readvariableop2
.savev2_dec_middle_0_kernel_read_readvariableop0
,savev2_dec_middle_0_bias_read_readvariableop2
.savev2_dec_middle_1_kernel_read_readvariableop0
,savev2_dec_middle_1_bias_read_readvariableop2
.savev2_dec_middle_2_kernel_read_readvariableop0
,savev2_dec_middle_2_bias_read_readvariableop1
-savev2_dec_outer_0_kernel_read_readvariableop/
+savev2_dec_outer_0_bias_read_readvariableop1
-savev2_dec_outer_1_kernel_read_readvariableop/
+savev2_dec_outer_1_bias_read_readvariableop1
-savev2_dec_outer_2_kernel_read_readvariableop/
+savev2_dec_outer_2_bias_read_readvariableop0
,savev2_dec_output_kernel_read_readvariableop.
*savev2_dec_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_0_bias_m_read_readvariableop8
4savev2_adam_enc_outer_1_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_1_bias_m_read_readvariableop8
4savev2_adam_enc_outer_2_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_2_bias_m_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_0_bias_m_read_readvariableop9
5savev2_adam_enc_middle_1_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_1_bias_m_read_readvariableop9
5savev2_adam_enc_middle_2_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_2_bias_m_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_0_bias_m_read_readvariableop8
4savev2_adam_enc_inner_1_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_1_bias_m_read_readvariableop8
4savev2_adam_enc_inner_2_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_2_bias_m_read_readvariableop6
2savev2_adam_channel_0_kernel_m_read_readvariableop4
0savev2_adam_channel_0_bias_m_read_readvariableop6
2savev2_adam_channel_1_kernel_m_read_readvariableop4
0savev2_adam_channel_1_bias_m_read_readvariableop6
2savev2_adam_channel_2_kernel_m_read_readvariableop4
0savev2_adam_channel_2_bias_m_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_0_bias_m_read_readvariableop8
4savev2_adam_dec_inner_1_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_1_bias_m_read_readvariableop8
4savev2_adam_dec_inner_2_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_2_bias_m_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_0_bias_m_read_readvariableop9
5savev2_adam_dec_middle_1_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_1_bias_m_read_readvariableop9
5savev2_adam_dec_middle_2_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_2_bias_m_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_0_bias_m_read_readvariableop8
4savev2_adam_dec_outer_1_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_1_bias_m_read_readvariableop8
4savev2_adam_dec_outer_2_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_2_bias_m_read_readvariableop7
3savev2_adam_dec_output_kernel_m_read_readvariableop5
1savev2_adam_dec_output_bias_m_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_0_bias_v_read_readvariableop8
4savev2_adam_enc_outer_1_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_1_bias_v_read_readvariableop8
4savev2_adam_enc_outer_2_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_2_bias_v_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_0_bias_v_read_readvariableop9
5savev2_adam_enc_middle_1_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_1_bias_v_read_readvariableop9
5savev2_adam_enc_middle_2_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_2_bias_v_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_0_bias_v_read_readvariableop8
4savev2_adam_enc_inner_1_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_1_bias_v_read_readvariableop8
4savev2_adam_enc_inner_2_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_2_bias_v_read_readvariableop6
2savev2_adam_channel_0_kernel_v_read_readvariableop4
0savev2_adam_channel_0_bias_v_read_readvariableop6
2savev2_adam_channel_1_kernel_v_read_readvariableop4
0savev2_adam_channel_1_bias_v_read_readvariableop6
2savev2_adam_channel_2_kernel_v_read_readvariableop4
0savev2_adam_channel_2_bias_v_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_0_bias_v_read_readvariableop8
4savev2_adam_dec_inner_1_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_1_bias_v_read_readvariableop8
4savev2_adam_dec_inner_2_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_2_bias_v_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_0_bias_v_read_readvariableop9
5savev2_adam_dec_middle_1_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_1_bias_v_read_readvariableop9
5savev2_adam_dec_middle_2_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_2_bias_v_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_0_bias_v_read_readvariableop8
4savev2_adam_dec_outer_1_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_1_bias_v_read_readvariableop8
4savev2_adam_dec_outer_2_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_2_bias_v_read_readvariableop7
3savev2_adam_dec_output_kernel_v_read_readvariableop5
1savev2_adam_dec_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?A
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?@
value?@B?@?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?8
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop-savev2_enc_outer_0_kernel_read_readvariableop+savev2_enc_outer_0_bias_read_readvariableop-savev2_enc_outer_1_kernel_read_readvariableop+savev2_enc_outer_1_bias_read_readvariableop-savev2_enc_outer_2_kernel_read_readvariableop+savev2_enc_outer_2_bias_read_readvariableop.savev2_enc_middle_0_kernel_read_readvariableop,savev2_enc_middle_0_bias_read_readvariableop.savev2_enc_middle_1_kernel_read_readvariableop,savev2_enc_middle_1_bias_read_readvariableop.savev2_enc_middle_2_kernel_read_readvariableop,savev2_enc_middle_2_bias_read_readvariableop-savev2_enc_inner_0_kernel_read_readvariableop+savev2_enc_inner_0_bias_read_readvariableop-savev2_enc_inner_1_kernel_read_readvariableop+savev2_enc_inner_1_bias_read_readvariableop-savev2_enc_inner_2_kernel_read_readvariableop+savev2_enc_inner_2_bias_read_readvariableop+savev2_channel_0_kernel_read_readvariableop)savev2_channel_0_bias_read_readvariableop+savev2_channel_1_kernel_read_readvariableop)savev2_channel_1_bias_read_readvariableop+savev2_channel_2_kernel_read_readvariableop)savev2_channel_2_bias_read_readvariableop-savev2_dec_inner_0_kernel_read_readvariableop+savev2_dec_inner_0_bias_read_readvariableop-savev2_dec_inner_1_kernel_read_readvariableop+savev2_dec_inner_1_bias_read_readvariableop-savev2_dec_inner_2_kernel_read_readvariableop+savev2_dec_inner_2_bias_read_readvariableop.savev2_dec_middle_0_kernel_read_readvariableop,savev2_dec_middle_0_bias_read_readvariableop.savev2_dec_middle_1_kernel_read_readvariableop,savev2_dec_middle_1_bias_read_readvariableop.savev2_dec_middle_2_kernel_read_readvariableop,savev2_dec_middle_2_bias_read_readvariableop-savev2_dec_outer_0_kernel_read_readvariableop+savev2_dec_outer_0_bias_read_readvariableop-savev2_dec_outer_1_kernel_read_readvariableop+savev2_dec_outer_1_bias_read_readvariableop-savev2_dec_outer_2_kernel_read_readvariableop+savev2_dec_outer_2_bias_read_readvariableop,savev2_dec_output_kernel_read_readvariableop*savev2_dec_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_enc_outer_0_kernel_m_read_readvariableop2savev2_adam_enc_outer_0_bias_m_read_readvariableop4savev2_adam_enc_outer_1_kernel_m_read_readvariableop2savev2_adam_enc_outer_1_bias_m_read_readvariableop4savev2_adam_enc_outer_2_kernel_m_read_readvariableop2savev2_adam_enc_outer_2_bias_m_read_readvariableop5savev2_adam_enc_middle_0_kernel_m_read_readvariableop3savev2_adam_enc_middle_0_bias_m_read_readvariableop5savev2_adam_enc_middle_1_kernel_m_read_readvariableop3savev2_adam_enc_middle_1_bias_m_read_readvariableop5savev2_adam_enc_middle_2_kernel_m_read_readvariableop3savev2_adam_enc_middle_2_bias_m_read_readvariableop4savev2_adam_enc_inner_0_kernel_m_read_readvariableop2savev2_adam_enc_inner_0_bias_m_read_readvariableop4savev2_adam_enc_inner_1_kernel_m_read_readvariableop2savev2_adam_enc_inner_1_bias_m_read_readvariableop4savev2_adam_enc_inner_2_kernel_m_read_readvariableop2savev2_adam_enc_inner_2_bias_m_read_readvariableop2savev2_adam_channel_0_kernel_m_read_readvariableop0savev2_adam_channel_0_bias_m_read_readvariableop2savev2_adam_channel_1_kernel_m_read_readvariableop0savev2_adam_channel_1_bias_m_read_readvariableop2savev2_adam_channel_2_kernel_m_read_readvariableop0savev2_adam_channel_2_bias_m_read_readvariableop4savev2_adam_dec_inner_0_kernel_m_read_readvariableop2savev2_adam_dec_inner_0_bias_m_read_readvariableop4savev2_adam_dec_inner_1_kernel_m_read_readvariableop2savev2_adam_dec_inner_1_bias_m_read_readvariableop4savev2_adam_dec_inner_2_kernel_m_read_readvariableop2savev2_adam_dec_inner_2_bias_m_read_readvariableop5savev2_adam_dec_middle_0_kernel_m_read_readvariableop3savev2_adam_dec_middle_0_bias_m_read_readvariableop5savev2_adam_dec_middle_1_kernel_m_read_readvariableop3savev2_adam_dec_middle_1_bias_m_read_readvariableop5savev2_adam_dec_middle_2_kernel_m_read_readvariableop3savev2_adam_dec_middle_2_bias_m_read_readvariableop4savev2_adam_dec_outer_0_kernel_m_read_readvariableop2savev2_adam_dec_outer_0_bias_m_read_readvariableop4savev2_adam_dec_outer_1_kernel_m_read_readvariableop2savev2_adam_dec_outer_1_bias_m_read_readvariableop4savev2_adam_dec_outer_2_kernel_m_read_readvariableop2savev2_adam_dec_outer_2_bias_m_read_readvariableop3savev2_adam_dec_output_kernel_m_read_readvariableop1savev2_adam_dec_output_bias_m_read_readvariableop4savev2_adam_enc_outer_0_kernel_v_read_readvariableop2savev2_adam_enc_outer_0_bias_v_read_readvariableop4savev2_adam_enc_outer_1_kernel_v_read_readvariableop2savev2_adam_enc_outer_1_bias_v_read_readvariableop4savev2_adam_enc_outer_2_kernel_v_read_readvariableop2savev2_adam_enc_outer_2_bias_v_read_readvariableop5savev2_adam_enc_middle_0_kernel_v_read_readvariableop3savev2_adam_enc_middle_0_bias_v_read_readvariableop5savev2_adam_enc_middle_1_kernel_v_read_readvariableop3savev2_adam_enc_middle_1_bias_v_read_readvariableop5savev2_adam_enc_middle_2_kernel_v_read_readvariableop3savev2_adam_enc_middle_2_bias_v_read_readvariableop4savev2_adam_enc_inner_0_kernel_v_read_readvariableop2savev2_adam_enc_inner_0_bias_v_read_readvariableop4savev2_adam_enc_inner_1_kernel_v_read_readvariableop2savev2_adam_enc_inner_1_bias_v_read_readvariableop4savev2_adam_enc_inner_2_kernel_v_read_readvariableop2savev2_adam_enc_inner_2_bias_v_read_readvariableop2savev2_adam_channel_0_kernel_v_read_readvariableop0savev2_adam_channel_0_bias_v_read_readvariableop2savev2_adam_channel_1_kernel_v_read_readvariableop0savev2_adam_channel_1_bias_v_read_readvariableop2savev2_adam_channel_2_kernel_v_read_readvariableop0savev2_adam_channel_2_bias_v_read_readvariableop4savev2_adam_dec_inner_0_kernel_v_read_readvariableop2savev2_adam_dec_inner_0_bias_v_read_readvariableop4savev2_adam_dec_inner_1_kernel_v_read_readvariableop2savev2_adam_dec_inner_1_bias_v_read_readvariableop4savev2_adam_dec_inner_2_kernel_v_read_readvariableop2savev2_adam_dec_inner_2_bias_v_read_readvariableop5savev2_adam_dec_middle_0_kernel_v_read_readvariableop3savev2_adam_dec_middle_0_bias_v_read_readvariableop5savev2_adam_dec_middle_1_kernel_v_read_readvariableop3savev2_adam_dec_middle_1_bias_v_read_readvariableop5savev2_adam_dec_middle_2_kernel_v_read_readvariableop3savev2_adam_dec_middle_2_bias_v_read_readvariableop4savev2_adam_dec_outer_0_kernel_v_read_readvariableop2savev2_adam_dec_outer_0_bias_v_read_readvariableop4savev2_adam_dec_outer_1_kernel_v_read_readvariableop2savev2_adam_dec_outer_1_bias_v_read_readvariableop4savev2_adam_dec_outer_2_kernel_v_read_readvariableop2savev2_adam_dec_outer_2_bias_v_read_readvariableop3savev2_adam_dec_output_kernel_v_read_readvariableop1savev2_adam_dec_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	?<:<:	?<:<:	?<:<:<2:2:<2:2:<2:2:2(:(:2(:(:2(:(:(::(::(::(:(:(:(:(:(:(<:<:(<:<:(<:<:<<:<:<<:<:<<:<:
??:?: : :	?<:<:	?<:<:	?<:<:<2:2:<2:2:<2:2:2(:(:2(:(:2(:(:(::(::(::(:(:(:(:(:(:(<:<:(<:<:(<:<:<<:<:<<:<:<<:<:
??:?:	?<:<:	?<:<:	?<:<:<2:2:<2:2:<2:2:2(:(:2(:(:2(:(:(::(::(::(:(:(:(:(:(:(<:<:(<:<:(<:<:<<:<:<<:<:<<:<:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?<: 

_output_shapes
:<:%!

_output_shapes
:	?<: 	

_output_shapes
:<:%
!

_output_shapes
:	?<: 

_output_shapes
:<:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:$  

_output_shapes

:(: !

_output_shapes
:(:$" 

_output_shapes

:(: #

_output_shapes
:(:$$ 

_output_shapes

:(<: %

_output_shapes
:<:$& 

_output_shapes

:(<: '

_output_shapes
:<:$( 

_output_shapes

:(<: )

_output_shapes
:<:$* 

_output_shapes

:<<: +

_output_shapes
:<:$, 

_output_shapes

:<<: -

_output_shapes
:<:$. 

_output_shapes

:<<: /

_output_shapes
:<:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:2

_output_shapes
: :3

_output_shapes
: :%4!

_output_shapes
:	?<: 5

_output_shapes
:<:%6!

_output_shapes
:	?<: 7

_output_shapes
:<:%8!

_output_shapes
:	?<: 9

_output_shapes
:<:$: 

_output_shapes

:<2: ;

_output_shapes
:2:$< 

_output_shapes

:<2: =

_output_shapes
:2:$> 

_output_shapes

:<2: ?

_output_shapes
:2:$@ 

_output_shapes

:2(: A

_output_shapes
:(:$B 

_output_shapes

:2(: C

_output_shapes
:(:$D 

_output_shapes

:2(: E

_output_shapes
:(:$F 

_output_shapes

:(: G

_output_shapes
::$H 

_output_shapes

:(: I

_output_shapes
::$J 

_output_shapes

:(: K

_output_shapes
::$L 

_output_shapes

:(: M

_output_shapes
:(:$N 

_output_shapes

:(: O

_output_shapes
:(:$P 

_output_shapes

:(: Q

_output_shapes
:(:$R 

_output_shapes

:(<: S

_output_shapes
:<:$T 

_output_shapes

:(<: U

_output_shapes
:<:$V 

_output_shapes

:(<: W

_output_shapes
:<:$X 

_output_shapes

:<<: Y

_output_shapes
:<:$Z 

_output_shapes

:<<: [

_output_shapes
:<:$\ 

_output_shapes

:<<: ]

_output_shapes
:<:&^"
 
_output_shapes
:
??:!_

_output_shapes	
:?:%`!

_output_shapes
:	?<: a

_output_shapes
:<:%b!

_output_shapes
:	?<: c

_output_shapes
:<:%d!

_output_shapes
:	?<: e

_output_shapes
:<:$f 

_output_shapes

:<2: g

_output_shapes
:2:$h 

_output_shapes

:<2: i

_output_shapes
:2:$j 

_output_shapes

:<2: k

_output_shapes
:2:$l 

_output_shapes

:2(: m

_output_shapes
:(:$n 

_output_shapes

:2(: o

_output_shapes
:(:$p 

_output_shapes

:2(: q

_output_shapes
:(:$r 

_output_shapes

:(: s

_output_shapes
::$t 

_output_shapes

:(: u

_output_shapes
::$v 

_output_shapes

:(: w

_output_shapes
::$x 

_output_shapes

:(: y

_output_shapes
:(:$z 

_output_shapes

:(: {

_output_shapes
:(:$| 

_output_shapes

:(: }

_output_shapes
:(:$~ 

_output_shapes

:(<: 

_output_shapes
:<:%? 

_output_shapes

:(<:!?

_output_shapes
:<:%? 

_output_shapes

:(<:!?

_output_shapes
:<:%? 

_output_shapes

:<<:!?

_output_shapes
:<:%? 

_output_shapes

:<<:!?

_output_shapes
:<:%? 

_output_shapes

:<<:!?

_output_shapes
:<:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:?

_output_shapes
: 
?	
?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_239929

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_2_layer_call_fn_239014
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_2383142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
.__inference_autoencoder_2_layer_call_fn_238921
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_2381252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238125
x
model_4_238032
model_4_238034
model_4_238036
model_4_238038
model_4_238040
model_4_238042
model_4_238044
model_4_238046
model_4_238048
model_4_238050
model_4_238052
model_4_238054
model_4_238056
model_4_238058
model_4_238060
model_4_238062
model_4_238064
model_4_238066
model_4_238068
model_4_238070
model_4_238072
model_4_238074
model_4_238076
model_4_238078
model_5_238083
model_5_238085
model_5_238087
model_5_238089
model_5_238091
model_5_238093
model_5_238095
model_5_238097
model_5_238099
model_5_238101
model_5_238103
model_5_238105
model_5_238107
model_5_238109
model_5_238111
model_5_238113
model_5_238115
model_5_238117
model_5_238119
model_5_238121
identity??model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?
model_4/StatefulPartitionedCallStatefulPartitionedCallxmodel_4_238032model_4_238034model_4_238036model_4_238038model_4_238040model_4_238042model_4_238044model_4_238046model_4_238048model_4_238050model_4_238052model_4_238054model_4_238056model_4_238058model_4_238060model_4_238062model_4_238064model_4_238066model_4_238068model_4_238070model_4_238072model_4_238074model_4_238076model_4_238078*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2368992!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0(model_4/StatefulPartitionedCall:output:1(model_4/StatefulPartitionedCall:output:2model_5_238083model_5_238085model_5_238087model_5_238089model_5_238091model_5_238093model_5_238095model_5_238097model_5_238099model_5_238101model_5_238103model_5_238105model_5_238107model_5_238109model_5_238111model_5_238113model_5_238115model_5_238117model_5_238119model_5_238121*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2374772!
model_5/StatefulPartitionedCall?
IdentityIdentity(model_5/StatefulPartitionedCall:output:0 ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
-__inference_enc_middle_0_layer_call_fn_239638

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_2365832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
,__inference_dec_inner_2_layer_call_fn_239858

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_2370942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_239569

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_2_layer_call_fn_238216
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_2381252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?J
?	
C__inference_model_4_layer_call_and_return_conditional_losses_236899

inputs
enc_outer_2_236836
enc_outer_2_236838
enc_outer_1_236841
enc_outer_1_236843
enc_outer_0_236846
enc_outer_0_236848
enc_middle_2_236851
enc_middle_2_236853
enc_middle_1_236856
enc_middle_1_236858
enc_middle_0_236861
enc_middle_0_236863
enc_inner_2_236866
enc_inner_2_236868
enc_inner_1_236871
enc_inner_1_236873
enc_inner_0_236876
enc_inner_0_236878
channel_2_236881
channel_2_236883
channel_1_236886
channel_1_236888
channel_0_236891
channel_0_236893
identity

identity_1

identity_2??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_2_236836enc_outer_2_236838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_2364482%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_1_236841enc_outer_1_236843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_2364752%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_236846enc_outer_0_236848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_2365022%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_236851enc_middle_2_236853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_2365292&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_236856enc_middle_1_236858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_2365562&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_236861enc_middle_0_236863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_2365832&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_236866enc_inner_2_236868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_2366102%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_236871enc_inner_1_236873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_2366372%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_236876enc_inner_0_236878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_2366642%
#enc_inner_0/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_236881channel_2_236883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_2_layer_call_and_return_conditional_losses_2366912#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_236886channel_1_236888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_1_layer_call_and_return_conditional_losses_2367182#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_236891channel_0_236893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_0_layer_call_and_return_conditional_losses_2367452#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?K
?	
C__inference_model_4_layer_call_and_return_conditional_losses_236764
encoder_input
enc_outer_2_236459
enc_outer_2_236461
enc_outer_1_236486
enc_outer_1_236488
enc_outer_0_236513
enc_outer_0_236515
enc_middle_2_236540
enc_middle_2_236542
enc_middle_1_236567
enc_middle_1_236569
enc_middle_0_236594
enc_middle_0_236596
enc_inner_2_236621
enc_inner_2_236623
enc_inner_1_236648
enc_inner_1_236650
enc_inner_0_236675
enc_inner_0_236677
channel_2_236702
channel_2_236704
channel_1_236729
channel_1_236731
channel_0_236756
channel_0_236758
identity

identity_1

identity_2??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_2_236459enc_outer_2_236461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_2364482%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_1_236486enc_outer_1_236488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_2364752%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_236513enc_outer_0_236515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_2365022%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_236540enc_middle_2_236542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_2365292&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_236567enc_middle_1_236569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_2365562&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_236594enc_middle_0_236596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_2365832&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_236621enc_inner_2_236623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_2366102%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_236648enc_inner_1_236650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_2366372%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_236675enc_inner_0_236677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_2366642%
#enc_inner_0/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_236702channel_2_236704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_2_layer_call_and_return_conditional_losses_2366912#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_236729channel_1_236731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_1_layer_call_and_return_conditional_losses_2367182#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_236756channel_0_236758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_0_layer_call_and_return_conditional_losses_2367452#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?
?
(__inference_model_5_layer_call_fn_239511
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2374772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?	
?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_239629

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_239969

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_236502

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_237202

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_236637

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
ֆ
?
C__inference_model_4_layer_call_and_return_conditional_losses_239194

inputs.
*enc_outer_2_matmul_readvariableop_resource/
+enc_outer_2_biasadd_readvariableop_resource.
*enc_outer_1_matmul_readvariableop_resource/
+enc_outer_1_biasadd_readvariableop_resource.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_2_matmul_readvariableop_resource0
,enc_middle_2_biasadd_readvariableop_resource/
+enc_middle_1_matmul_readvariableop_resource0
,enc_middle_1_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_2_matmul_readvariableop_resource/
+enc_inner_2_biasadd_readvariableop_resource.
*enc_inner_1_matmul_readvariableop_resource/
+enc_inner_1_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_2_matmul_readvariableop_resource-
)channel_2_biasadd_readvariableop_resource,
(channel_1_matmul_readvariableop_resource-
)channel_1_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity

identity_1

identity_2?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp? channel_1/BiasAdd/ReadVariableOp?channel_1/MatMul/ReadVariableOp? channel_2/BiasAdd/ReadVariableOp?channel_2/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?"enc_inner_1/BiasAdd/ReadVariableOp?!enc_inner_1/MatMul/ReadVariableOp?"enc_inner_2/BiasAdd/ReadVariableOp?!enc_inner_2/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?#enc_middle_1/BiasAdd/ReadVariableOp?"enc_middle_1/MatMul/ReadVariableOp?#enc_middle_2/BiasAdd/ReadVariableOp?"enc_middle_2/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?"enc_outer_1/BiasAdd/ReadVariableOp?!enc_outer_1/MatMul/ReadVariableOp?"enc_outer_2/BiasAdd/ReadVariableOp?!enc_outer_2/MatMul/ReadVariableOp?
!enc_outer_2/MatMul/ReadVariableOpReadVariableOp*enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_2/MatMul/ReadVariableOp?
enc_outer_2/MatMulMatMulinputs)enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_2/MatMul?
"enc_outer_2/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_2/BiasAdd/ReadVariableOp?
enc_outer_2/BiasAddBiasAddenc_outer_2/MatMul:product:0*enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_2/BiasAdd|
enc_outer_2/ReluReluenc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_2/Relu?
!enc_outer_1/MatMul/ReadVariableOpReadVariableOp*enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_1/MatMul/ReadVariableOp?
enc_outer_1/MatMulMatMulinputs)enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_1/MatMul?
"enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_1/BiasAdd/ReadVariableOp?
enc_outer_1/BiasAddBiasAddenc_outer_1/MatMul:product:0*enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_1/BiasAdd|
enc_outer_1/ReluReluenc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_1/Relu?
!enc_outer_0/MatMul/ReadVariableOpReadVariableOp*enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_0/MatMul/ReadVariableOp?
enc_outer_0/MatMulMatMulinputs)enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_0/MatMul?
"enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_0/BiasAdd/ReadVariableOp?
enc_outer_0/BiasAddBiasAddenc_outer_0/MatMul:product:0*enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_0/BiasAdd|
enc_outer_0/ReluReluenc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_0/Relu?
"enc_middle_2/MatMul/ReadVariableOpReadVariableOp+enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_2/MatMul/ReadVariableOp?
enc_middle_2/MatMulMatMulenc_outer_2/Relu:activations:0*enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_2/MatMul?
#enc_middle_2/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_2/BiasAdd/ReadVariableOp?
enc_middle_2/BiasAddBiasAddenc_middle_2/MatMul:product:0+enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_2/BiasAdd
enc_middle_2/ReluReluenc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_2/Relu?
"enc_middle_1/MatMul/ReadVariableOpReadVariableOp+enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_1/MatMul/ReadVariableOp?
enc_middle_1/MatMulMatMulenc_outer_1/Relu:activations:0*enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_1/MatMul?
#enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_1/BiasAdd/ReadVariableOp?
enc_middle_1/BiasAddBiasAddenc_middle_1/MatMul:product:0+enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_1/BiasAdd
enc_middle_1/ReluReluenc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_1/Relu?
"enc_middle_0/MatMul/ReadVariableOpReadVariableOp+enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_0/MatMul/ReadVariableOp?
enc_middle_0/MatMulMatMulenc_outer_0/Relu:activations:0*enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_0/MatMul?
#enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_0/BiasAdd/ReadVariableOp?
enc_middle_0/BiasAddBiasAddenc_middle_0/MatMul:product:0+enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_0/BiasAdd
enc_middle_0/ReluReluenc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_0/Relu?
!enc_inner_2/MatMul/ReadVariableOpReadVariableOp*enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_2/MatMul/ReadVariableOp?
enc_inner_2/MatMulMatMulenc_middle_2/Relu:activations:0)enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_2/MatMul?
"enc_inner_2/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_2/BiasAdd/ReadVariableOp?
enc_inner_2/BiasAddBiasAddenc_inner_2/MatMul:product:0*enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_2/BiasAdd|
enc_inner_2/ReluReluenc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_2/Relu?
!enc_inner_1/MatMul/ReadVariableOpReadVariableOp*enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_1/MatMul/ReadVariableOp?
enc_inner_1/MatMulMatMulenc_middle_1/Relu:activations:0)enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_1/MatMul?
"enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_1/BiasAdd/ReadVariableOp?
enc_inner_1/BiasAddBiasAddenc_inner_1/MatMul:product:0*enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_1/BiasAdd|
enc_inner_1/ReluReluenc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_1/Relu?
!enc_inner_0/MatMul/ReadVariableOpReadVariableOp*enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_0/MatMul/ReadVariableOp?
enc_inner_0/MatMulMatMulenc_middle_0/Relu:activations:0)enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_0/MatMul?
"enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_0/BiasAdd/ReadVariableOp?
enc_inner_0/BiasAddBiasAddenc_inner_0/MatMul:product:0*enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_0/BiasAdd|
enc_inner_0/ReluReluenc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_0/Relu?
channel_2/MatMul/ReadVariableOpReadVariableOp(channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_2/MatMul/ReadVariableOp?
channel_2/MatMulMatMulenc_inner_2/Relu:activations:0'channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_2/MatMul?
 channel_2/BiasAdd/ReadVariableOpReadVariableOp)channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_2/BiasAdd/ReadVariableOp?
channel_2/BiasAddBiasAddchannel_2/MatMul:product:0(channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_2/BiasAdd?
channel_2/SoftsignSoftsignchannel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_2/Softsign?
channel_1/MatMul/ReadVariableOpReadVariableOp(channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_1/MatMul/ReadVariableOp?
channel_1/MatMulMatMulenc_inner_1/Relu:activations:0'channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_1/MatMul?
 channel_1/BiasAdd/ReadVariableOpReadVariableOp)channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_1/BiasAdd/ReadVariableOp?
channel_1/BiasAddBiasAddchannel_1/MatMul:product:0(channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_1/BiasAdd?
channel_1/SoftsignSoftsignchannel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_1/Softsign?
channel_0/MatMul/ReadVariableOpReadVariableOp(channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_0/MatMul/ReadVariableOp?
channel_0/MatMulMatMulenc_inner_0/Relu:activations:0'channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_0/MatMul?
 channel_0/BiasAdd/ReadVariableOpReadVariableOp)channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_0/BiasAdd/ReadVariableOp?
channel_0/BiasAddBiasAddchannel_0/MatMul:product:0(channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_0/BiasAdd?
channel_0/SoftsignSoftsignchannel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_0/Softsign?
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity channel_1/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity channel_2/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2D
 channel_1/BiasAdd/ReadVariableOp channel_1/BiasAdd/ReadVariableOp2B
channel_1/MatMul/ReadVariableOpchannel_1/MatMul/ReadVariableOp2D
 channel_2/BiasAdd/ReadVariableOp channel_2/BiasAdd/ReadVariableOp2B
channel_2/MatMul/ReadVariableOpchannel_2/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2H
"enc_inner_1/BiasAdd/ReadVariableOp"enc_inner_1/BiasAdd/ReadVariableOp2F
!enc_inner_1/MatMul/ReadVariableOp!enc_inner_1/MatMul/ReadVariableOp2H
"enc_inner_2/BiasAdd/ReadVariableOp"enc_inner_2/BiasAdd/ReadVariableOp2F
!enc_inner_2/MatMul/ReadVariableOp!enc_inner_2/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2J
#enc_middle_1/BiasAdd/ReadVariableOp#enc_middle_1/BiasAdd/ReadVariableOp2H
"enc_middle_1/MatMul/ReadVariableOp"enc_middle_1/MatMul/ReadVariableOp2J
#enc_middle_2/BiasAdd/ReadVariableOp#enc_middle_2/BiasAdd/ReadVariableOp2H
"enc_middle_2/MatMul/ReadVariableOp"enc_middle_2/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp2H
"enc_outer_1/BiasAdd/ReadVariableOp"enc_outer_1/BiasAdd/ReadVariableOp2F
!enc_outer_1/MatMul/ReadVariableOp!enc_outer_1/MatMul/ReadVariableOp2H
"enc_outer_2/BiasAdd/ReadVariableOp"enc_outer_2/BiasAdd/ReadVariableOp2F
!enc_outer_2/MatMul/ReadVariableOp!enc_outer_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_dec_middle_1_layer_call_fn_239898

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_2372022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_237148

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_237094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_239809

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
C__inference_model_5_layer_call_and_return_conditional_losses_237356
decoder_input_0
decoder_input_1
decoder_input_2
dec_inner_2_237105
dec_inner_2_237107
dec_inner_1_237132
dec_inner_1_237134
dec_inner_0_237159
dec_inner_0_237161
dec_middle_2_237186
dec_middle_2_237188
dec_middle_1_237213
dec_middle_1_237215
dec_middle_0_237240
dec_middle_0_237242
dec_outer_0_237267
dec_outer_0_237269
dec_outer_1_237294
dec_outer_1_237296
dec_outer_2_237321
dec_outer_2_237323
dec_output_237350
dec_output_237352
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_2dec_inner_2_237105dec_inner_2_237107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_2370942%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_1dec_inner_1_237132dec_inner_1_237134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_2371212%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_237159dec_inner_0_237161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_2371482%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_237186dec_middle_2_237188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_2371752&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_237213dec_middle_1_237215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_2372022&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_237240dec_middle_0_237242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_2372292&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_237267dec_outer_0_237269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_2372562%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_237294dec_outer_1_237296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_2372832%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_237321dec_outer_2_237323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_2373102%
#dec_outer_2/StatefulPartitionedCallt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_1/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dec_output_237350dec_output_237352*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dec_output_layer_call_and_return_conditional_losses_2373392$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_1:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_2
?	
?
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_239909

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
??
?K
"__inference__traced_restore_240865
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate)
%assignvariableop_5_enc_outer_0_kernel'
#assignvariableop_6_enc_outer_0_bias)
%assignvariableop_7_enc_outer_1_kernel'
#assignvariableop_8_enc_outer_1_bias)
%assignvariableop_9_enc_outer_2_kernel(
$assignvariableop_10_enc_outer_2_bias+
'assignvariableop_11_enc_middle_0_kernel)
%assignvariableop_12_enc_middle_0_bias+
'assignvariableop_13_enc_middle_1_kernel)
%assignvariableop_14_enc_middle_1_bias+
'assignvariableop_15_enc_middle_2_kernel)
%assignvariableop_16_enc_middle_2_bias*
&assignvariableop_17_enc_inner_0_kernel(
$assignvariableop_18_enc_inner_0_bias*
&assignvariableop_19_enc_inner_1_kernel(
$assignvariableop_20_enc_inner_1_bias*
&assignvariableop_21_enc_inner_2_kernel(
$assignvariableop_22_enc_inner_2_bias(
$assignvariableop_23_channel_0_kernel&
"assignvariableop_24_channel_0_bias(
$assignvariableop_25_channel_1_kernel&
"assignvariableop_26_channel_1_bias(
$assignvariableop_27_channel_2_kernel&
"assignvariableop_28_channel_2_bias*
&assignvariableop_29_dec_inner_0_kernel(
$assignvariableop_30_dec_inner_0_bias*
&assignvariableop_31_dec_inner_1_kernel(
$assignvariableop_32_dec_inner_1_bias*
&assignvariableop_33_dec_inner_2_kernel(
$assignvariableop_34_dec_inner_2_bias+
'assignvariableop_35_dec_middle_0_kernel)
%assignvariableop_36_dec_middle_0_bias+
'assignvariableop_37_dec_middle_1_kernel)
%assignvariableop_38_dec_middle_1_bias+
'assignvariableop_39_dec_middle_2_kernel)
%assignvariableop_40_dec_middle_2_bias*
&assignvariableop_41_dec_outer_0_kernel(
$assignvariableop_42_dec_outer_0_bias*
&assignvariableop_43_dec_outer_1_kernel(
$assignvariableop_44_dec_outer_1_bias*
&assignvariableop_45_dec_outer_2_kernel(
$assignvariableop_46_dec_outer_2_bias)
%assignvariableop_47_dec_output_kernel'
#assignvariableop_48_dec_output_bias
assignvariableop_49_total
assignvariableop_50_count1
-assignvariableop_51_adam_enc_outer_0_kernel_m/
+assignvariableop_52_adam_enc_outer_0_bias_m1
-assignvariableop_53_adam_enc_outer_1_kernel_m/
+assignvariableop_54_adam_enc_outer_1_bias_m1
-assignvariableop_55_adam_enc_outer_2_kernel_m/
+assignvariableop_56_adam_enc_outer_2_bias_m2
.assignvariableop_57_adam_enc_middle_0_kernel_m0
,assignvariableop_58_adam_enc_middle_0_bias_m2
.assignvariableop_59_adam_enc_middle_1_kernel_m0
,assignvariableop_60_adam_enc_middle_1_bias_m2
.assignvariableop_61_adam_enc_middle_2_kernel_m0
,assignvariableop_62_adam_enc_middle_2_bias_m1
-assignvariableop_63_adam_enc_inner_0_kernel_m/
+assignvariableop_64_adam_enc_inner_0_bias_m1
-assignvariableop_65_adam_enc_inner_1_kernel_m/
+assignvariableop_66_adam_enc_inner_1_bias_m1
-assignvariableop_67_adam_enc_inner_2_kernel_m/
+assignvariableop_68_adam_enc_inner_2_bias_m/
+assignvariableop_69_adam_channel_0_kernel_m-
)assignvariableop_70_adam_channel_0_bias_m/
+assignvariableop_71_adam_channel_1_kernel_m-
)assignvariableop_72_adam_channel_1_bias_m/
+assignvariableop_73_adam_channel_2_kernel_m-
)assignvariableop_74_adam_channel_2_bias_m1
-assignvariableop_75_adam_dec_inner_0_kernel_m/
+assignvariableop_76_adam_dec_inner_0_bias_m1
-assignvariableop_77_adam_dec_inner_1_kernel_m/
+assignvariableop_78_adam_dec_inner_1_bias_m1
-assignvariableop_79_adam_dec_inner_2_kernel_m/
+assignvariableop_80_adam_dec_inner_2_bias_m2
.assignvariableop_81_adam_dec_middle_0_kernel_m0
,assignvariableop_82_adam_dec_middle_0_bias_m2
.assignvariableop_83_adam_dec_middle_1_kernel_m0
,assignvariableop_84_adam_dec_middle_1_bias_m2
.assignvariableop_85_adam_dec_middle_2_kernel_m0
,assignvariableop_86_adam_dec_middle_2_bias_m1
-assignvariableop_87_adam_dec_outer_0_kernel_m/
+assignvariableop_88_adam_dec_outer_0_bias_m1
-assignvariableop_89_adam_dec_outer_1_kernel_m/
+assignvariableop_90_adam_dec_outer_1_bias_m1
-assignvariableop_91_adam_dec_outer_2_kernel_m/
+assignvariableop_92_adam_dec_outer_2_bias_m0
,assignvariableop_93_adam_dec_output_kernel_m.
*assignvariableop_94_adam_dec_output_bias_m1
-assignvariableop_95_adam_enc_outer_0_kernel_v/
+assignvariableop_96_adam_enc_outer_0_bias_v1
-assignvariableop_97_adam_enc_outer_1_kernel_v/
+assignvariableop_98_adam_enc_outer_1_bias_v1
-assignvariableop_99_adam_enc_outer_2_kernel_v0
,assignvariableop_100_adam_enc_outer_2_bias_v3
/assignvariableop_101_adam_enc_middle_0_kernel_v1
-assignvariableop_102_adam_enc_middle_0_bias_v3
/assignvariableop_103_adam_enc_middle_1_kernel_v1
-assignvariableop_104_adam_enc_middle_1_bias_v3
/assignvariableop_105_adam_enc_middle_2_kernel_v1
-assignvariableop_106_adam_enc_middle_2_bias_v2
.assignvariableop_107_adam_enc_inner_0_kernel_v0
,assignvariableop_108_adam_enc_inner_0_bias_v2
.assignvariableop_109_adam_enc_inner_1_kernel_v0
,assignvariableop_110_adam_enc_inner_1_bias_v2
.assignvariableop_111_adam_enc_inner_2_kernel_v0
,assignvariableop_112_adam_enc_inner_2_bias_v0
,assignvariableop_113_adam_channel_0_kernel_v.
*assignvariableop_114_adam_channel_0_bias_v0
,assignvariableop_115_adam_channel_1_kernel_v.
*assignvariableop_116_adam_channel_1_bias_v0
,assignvariableop_117_adam_channel_2_kernel_v.
*assignvariableop_118_adam_channel_2_bias_v2
.assignvariableop_119_adam_dec_inner_0_kernel_v0
,assignvariableop_120_adam_dec_inner_0_bias_v2
.assignvariableop_121_adam_dec_inner_1_kernel_v0
,assignvariableop_122_adam_dec_inner_1_bias_v2
.assignvariableop_123_adam_dec_inner_2_kernel_v0
,assignvariableop_124_adam_dec_inner_2_bias_v3
/assignvariableop_125_adam_dec_middle_0_kernel_v1
-assignvariableop_126_adam_dec_middle_0_bias_v3
/assignvariableop_127_adam_dec_middle_1_kernel_v1
-assignvariableop_128_adam_dec_middle_1_bias_v3
/assignvariableop_129_adam_dec_middle_2_kernel_v1
-assignvariableop_130_adam_dec_middle_2_bias_v2
.assignvariableop_131_adam_dec_outer_0_kernel_v0
,assignvariableop_132_adam_dec_outer_0_bias_v2
.assignvariableop_133_adam_dec_outer_1_kernel_v0
,assignvariableop_134_adam_dec_outer_1_bias_v2
.assignvariableop_135_adam_dec_outer_2_kernel_v0
,assignvariableop_136_adam_dec_outer_2_bias_v1
-assignvariableop_137_adam_dec_output_kernel_v/
+assignvariableop_138_adam_dec_output_bias_v
identity_140??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?A
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?@
value?@B?@?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp%assignvariableop_5_enc_outer_0_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_enc_outer_0_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_enc_outer_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_enc_outer_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_enc_outer_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_enc_outer_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_enc_middle_0_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_enc_middle_0_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_enc_middle_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_enc_middle_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_enc_middle_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_enc_middle_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_enc_inner_0_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_enc_inner_0_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp&assignvariableop_19_enc_inner_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_enc_inner_1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_enc_inner_2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_enc_inner_2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_channel_0_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_channel_0_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_channel_1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_channel_1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_channel_2_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_channel_2_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_dec_inner_0_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_dec_inner_0_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_dec_inner_1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_dec_inner_1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp&assignvariableop_33_dec_inner_2_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_dec_inner_2_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_dec_middle_0_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_dec_middle_0_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_dec_middle_1_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_dec_middle_1_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp'assignvariableop_39_dec_middle_2_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp%assignvariableop_40_dec_middle_2_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp&assignvariableop_41_dec_outer_0_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_dec_outer_0_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp&assignvariableop_43_dec_outer_1_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp$assignvariableop_44_dec_outer_1_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp&assignvariableop_45_dec_outer_2_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp$assignvariableop_46_dec_outer_2_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp%assignvariableop_47_dec_output_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp#assignvariableop_48_dec_output_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_enc_outer_0_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_enc_outer_0_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_enc_outer_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_enc_outer_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_enc_outer_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_enc_outer_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_enc_middle_0_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp,assignvariableop_58_adam_enc_middle_0_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp.assignvariableop_59_adam_enc_middle_1_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_enc_middle_1_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp.assignvariableop_61_adam_enc_middle_2_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_enc_middle_2_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp-assignvariableop_63_adam_enc_inner_0_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp+assignvariableop_64_adam_enc_inner_0_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp-assignvariableop_65_adam_enc_inner_1_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_enc_inner_1_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp-assignvariableop_67_adam_enc_inner_2_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_enc_inner_2_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_channel_0_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_channel_0_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_channel_1_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_channel_1_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_channel_2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_channel_2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp-assignvariableop_75_adam_dec_inner_0_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp+assignvariableop_76_adam_dec_inner_0_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp-assignvariableop_77_adam_dec_inner_1_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_dec_inner_1_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp-assignvariableop_79_adam_dec_inner_2_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_dec_inner_2_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp.assignvariableop_81_adam_dec_middle_0_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp,assignvariableop_82_adam_dec_middle_0_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp.assignvariableop_83_adam_dec_middle_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp,assignvariableop_84_adam_dec_middle_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp.assignvariableop_85_adam_dec_middle_2_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp,assignvariableop_86_adam_dec_middle_2_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp-assignvariableop_87_adam_dec_outer_0_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp+assignvariableop_88_adam_dec_outer_0_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp-assignvariableop_89_adam_dec_outer_1_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp+assignvariableop_90_adam_dec_outer_1_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp-assignvariableop_91_adam_dec_outer_2_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp+assignvariableop_92_adam_dec_outer_2_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_dec_output_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_dec_output_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp-assignvariableop_95_adam_enc_outer_0_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp+assignvariableop_96_adam_enc_outer_0_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp-assignvariableop_97_adam_enc_outer_1_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp+assignvariableop_98_adam_enc_outer_1_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp-assignvariableop_99_adam_enc_outer_2_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp,assignvariableop_100_adam_enc_outer_2_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp/assignvariableop_101_adam_enc_middle_0_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp-assignvariableop_102_adam_enc_middle_0_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp/assignvariableop_103_adam_enc_middle_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp-assignvariableop_104_adam_enc_middle_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp/assignvariableop_105_adam_enc_middle_2_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp-assignvariableop_106_adam_enc_middle_2_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp.assignvariableop_107_adam_enc_inner_0_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp,assignvariableop_108_adam_enc_inner_0_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp.assignvariableop_109_adam_enc_inner_1_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp,assignvariableop_110_adam_enc_inner_1_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp.assignvariableop_111_adam_enc_inner_2_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp,assignvariableop_112_adam_enc_inner_2_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_channel_0_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_channel_0_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_channel_1_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_channel_1_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_channel_2_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_channel_2_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp.assignvariableop_119_adam_dec_inner_0_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp,assignvariableop_120_adam_dec_inner_0_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp.assignvariableop_121_adam_dec_inner_1_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp,assignvariableop_122_adam_dec_inner_1_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp.assignvariableop_123_adam_dec_inner_2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp,assignvariableop_124_adam_dec_inner_2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp/assignvariableop_125_adam_dec_middle_0_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp-assignvariableop_126_adam_dec_middle_0_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp/assignvariableop_127_adam_dec_middle_1_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp-assignvariableop_128_adam_dec_middle_1_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp/assignvariableop_129_adam_dec_middle_2_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp-assignvariableop_130_adam_dec_middle_2_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp.assignvariableop_131_adam_dec_outer_0_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp,assignvariableop_132_adam_dec_outer_0_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp.assignvariableop_133_adam_dec_outer_1_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp,assignvariableop_134_adam_dec_outer_1_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOp.assignvariableop_135_adam_dec_outer_2_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOp,assignvariableop_136_adam_dec_outer_2_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOp-assignvariableop_137_adam_dec_output_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138?
AssignVariableOp_138AssignVariableOp+assignvariableop_138_adam_dec_output_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_139Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_139?
Identity_140IdentityIdentity_139:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_140"%
identity_140Identity_140:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_enc_inner_1_layer_call_fn_239718

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_2366372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
F__inference_dec_output_layer_call_and_return_conditional_losses_237339

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?#
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238668
x6
2model_4_enc_outer_2_matmul_readvariableop_resource7
3model_4_enc_outer_2_biasadd_readvariableop_resource6
2model_4_enc_outer_1_matmul_readvariableop_resource7
3model_4_enc_outer_1_biasadd_readvariableop_resource6
2model_4_enc_outer_0_matmul_readvariableop_resource7
3model_4_enc_outer_0_biasadd_readvariableop_resource7
3model_4_enc_middle_2_matmul_readvariableop_resource8
4model_4_enc_middle_2_biasadd_readvariableop_resource7
3model_4_enc_middle_1_matmul_readvariableop_resource8
4model_4_enc_middle_1_biasadd_readvariableop_resource7
3model_4_enc_middle_0_matmul_readvariableop_resource8
4model_4_enc_middle_0_biasadd_readvariableop_resource6
2model_4_enc_inner_2_matmul_readvariableop_resource7
3model_4_enc_inner_2_biasadd_readvariableop_resource6
2model_4_enc_inner_1_matmul_readvariableop_resource7
3model_4_enc_inner_1_biasadd_readvariableop_resource6
2model_4_enc_inner_0_matmul_readvariableop_resource7
3model_4_enc_inner_0_biasadd_readvariableop_resource4
0model_4_channel_2_matmul_readvariableop_resource5
1model_4_channel_2_biasadd_readvariableop_resource4
0model_4_channel_1_matmul_readvariableop_resource5
1model_4_channel_1_biasadd_readvariableop_resource4
0model_4_channel_0_matmul_readvariableop_resource5
1model_4_channel_0_biasadd_readvariableop_resource6
2model_5_dec_inner_2_matmul_readvariableop_resource7
3model_5_dec_inner_2_biasadd_readvariableop_resource6
2model_5_dec_inner_1_matmul_readvariableop_resource7
3model_5_dec_inner_1_biasadd_readvariableop_resource6
2model_5_dec_inner_0_matmul_readvariableop_resource7
3model_5_dec_inner_0_biasadd_readvariableop_resource7
3model_5_dec_middle_2_matmul_readvariableop_resource8
4model_5_dec_middle_2_biasadd_readvariableop_resource7
3model_5_dec_middle_1_matmul_readvariableop_resource8
4model_5_dec_middle_1_biasadd_readvariableop_resource7
3model_5_dec_middle_0_matmul_readvariableop_resource8
4model_5_dec_middle_0_biasadd_readvariableop_resource6
2model_5_dec_outer_0_matmul_readvariableop_resource7
3model_5_dec_outer_0_biasadd_readvariableop_resource6
2model_5_dec_outer_1_matmul_readvariableop_resource7
3model_5_dec_outer_1_biasadd_readvariableop_resource6
2model_5_dec_outer_2_matmul_readvariableop_resource7
3model_5_dec_outer_2_biasadd_readvariableop_resource5
1model_5_dec_output_matmul_readvariableop_resource6
2model_5_dec_output_biasadd_readvariableop_resource
identity??(model_4/channel_0/BiasAdd/ReadVariableOp?'model_4/channel_0/MatMul/ReadVariableOp?(model_4/channel_1/BiasAdd/ReadVariableOp?'model_4/channel_1/MatMul/ReadVariableOp?(model_4/channel_2/BiasAdd/ReadVariableOp?'model_4/channel_2/MatMul/ReadVariableOp?*model_4/enc_inner_0/BiasAdd/ReadVariableOp?)model_4/enc_inner_0/MatMul/ReadVariableOp?*model_4/enc_inner_1/BiasAdd/ReadVariableOp?)model_4/enc_inner_1/MatMul/ReadVariableOp?*model_4/enc_inner_2/BiasAdd/ReadVariableOp?)model_4/enc_inner_2/MatMul/ReadVariableOp?+model_4/enc_middle_0/BiasAdd/ReadVariableOp?*model_4/enc_middle_0/MatMul/ReadVariableOp?+model_4/enc_middle_1/BiasAdd/ReadVariableOp?*model_4/enc_middle_1/MatMul/ReadVariableOp?+model_4/enc_middle_2/BiasAdd/ReadVariableOp?*model_4/enc_middle_2/MatMul/ReadVariableOp?*model_4/enc_outer_0/BiasAdd/ReadVariableOp?)model_4/enc_outer_0/MatMul/ReadVariableOp?*model_4/enc_outer_1/BiasAdd/ReadVariableOp?)model_4/enc_outer_1/MatMul/ReadVariableOp?*model_4/enc_outer_2/BiasAdd/ReadVariableOp?)model_4/enc_outer_2/MatMul/ReadVariableOp?*model_5/dec_inner_0/BiasAdd/ReadVariableOp?)model_5/dec_inner_0/MatMul/ReadVariableOp?*model_5/dec_inner_1/BiasAdd/ReadVariableOp?)model_5/dec_inner_1/MatMul/ReadVariableOp?*model_5/dec_inner_2/BiasAdd/ReadVariableOp?)model_5/dec_inner_2/MatMul/ReadVariableOp?+model_5/dec_middle_0/BiasAdd/ReadVariableOp?*model_5/dec_middle_0/MatMul/ReadVariableOp?+model_5/dec_middle_1/BiasAdd/ReadVariableOp?*model_5/dec_middle_1/MatMul/ReadVariableOp?+model_5/dec_middle_2/BiasAdd/ReadVariableOp?*model_5/dec_middle_2/MatMul/ReadVariableOp?*model_5/dec_outer_0/BiasAdd/ReadVariableOp?)model_5/dec_outer_0/MatMul/ReadVariableOp?*model_5/dec_outer_1/BiasAdd/ReadVariableOp?)model_5/dec_outer_1/MatMul/ReadVariableOp?*model_5/dec_outer_2/BiasAdd/ReadVariableOp?)model_5/dec_outer_2/MatMul/ReadVariableOp?)model_5/dec_output/BiasAdd/ReadVariableOp?(model_5/dec_output/MatMul/ReadVariableOp?
)model_4/enc_outer_2/MatMul/ReadVariableOpReadVariableOp2model_4_enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_4/enc_outer_2/MatMul/ReadVariableOp?
model_4/enc_outer_2/MatMulMatMulx1model_4/enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_2/MatMul?
*model_4/enc_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_4/enc_outer_2/BiasAdd/ReadVariableOp?
model_4/enc_outer_2/BiasAddBiasAdd$model_4/enc_outer_2/MatMul:product:02model_4/enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_2/BiasAdd?
model_4/enc_outer_2/ReluRelu$model_4/enc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_2/Relu?
)model_4/enc_outer_1/MatMul/ReadVariableOpReadVariableOp2model_4_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_4/enc_outer_1/MatMul/ReadVariableOp?
model_4/enc_outer_1/MatMulMatMulx1model_4/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_1/MatMul?
*model_4/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_4/enc_outer_1/BiasAdd/ReadVariableOp?
model_4/enc_outer_1/BiasAddBiasAdd$model_4/enc_outer_1/MatMul:product:02model_4/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_1/BiasAdd?
model_4/enc_outer_1/ReluRelu$model_4/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_1/Relu?
)model_4/enc_outer_0/MatMul/ReadVariableOpReadVariableOp2model_4_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_4/enc_outer_0/MatMul/ReadVariableOp?
model_4/enc_outer_0/MatMulMatMulx1model_4/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_0/MatMul?
*model_4/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_4/enc_outer_0/BiasAdd/ReadVariableOp?
model_4/enc_outer_0/BiasAddBiasAdd$model_4/enc_outer_0/MatMul:product:02model_4/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_0/BiasAdd?
model_4/enc_outer_0/ReluRelu$model_4/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_0/Relu?
*model_4/enc_middle_2/MatMul/ReadVariableOpReadVariableOp3model_4_enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_4/enc_middle_2/MatMul/ReadVariableOp?
model_4/enc_middle_2/MatMulMatMul&model_4/enc_outer_2/Relu:activations:02model_4/enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_2/MatMul?
+model_4/enc_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_4_enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_4/enc_middle_2/BiasAdd/ReadVariableOp?
model_4/enc_middle_2/BiasAddBiasAdd%model_4/enc_middle_2/MatMul:product:03model_4/enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_2/BiasAdd?
model_4/enc_middle_2/ReluRelu%model_4/enc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_2/Relu?
*model_4/enc_middle_1/MatMul/ReadVariableOpReadVariableOp3model_4_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_4/enc_middle_1/MatMul/ReadVariableOp?
model_4/enc_middle_1/MatMulMatMul&model_4/enc_outer_1/Relu:activations:02model_4/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_1/MatMul?
+model_4/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_4_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_4/enc_middle_1/BiasAdd/ReadVariableOp?
model_4/enc_middle_1/BiasAddBiasAdd%model_4/enc_middle_1/MatMul:product:03model_4/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_1/BiasAdd?
model_4/enc_middle_1/ReluRelu%model_4/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_1/Relu?
*model_4/enc_middle_0/MatMul/ReadVariableOpReadVariableOp3model_4_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_4/enc_middle_0/MatMul/ReadVariableOp?
model_4/enc_middle_0/MatMulMatMul&model_4/enc_outer_0/Relu:activations:02model_4/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_0/MatMul?
+model_4/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_4_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_4/enc_middle_0/BiasAdd/ReadVariableOp?
model_4/enc_middle_0/BiasAddBiasAdd%model_4/enc_middle_0/MatMul:product:03model_4/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_0/BiasAdd?
model_4/enc_middle_0/ReluRelu%model_4/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_0/Relu?
)model_4/enc_inner_2/MatMul/ReadVariableOpReadVariableOp2model_4_enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_4/enc_inner_2/MatMul/ReadVariableOp?
model_4/enc_inner_2/MatMulMatMul'model_4/enc_middle_2/Relu:activations:01model_4/enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_2/MatMul?
*model_4/enc_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_4/enc_inner_2/BiasAdd/ReadVariableOp?
model_4/enc_inner_2/BiasAddBiasAdd$model_4/enc_inner_2/MatMul:product:02model_4/enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_2/BiasAdd?
model_4/enc_inner_2/ReluRelu$model_4/enc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_2/Relu?
)model_4/enc_inner_1/MatMul/ReadVariableOpReadVariableOp2model_4_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_4/enc_inner_1/MatMul/ReadVariableOp?
model_4/enc_inner_1/MatMulMatMul'model_4/enc_middle_1/Relu:activations:01model_4/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_1/MatMul?
*model_4/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_4/enc_inner_1/BiasAdd/ReadVariableOp?
model_4/enc_inner_1/BiasAddBiasAdd$model_4/enc_inner_1/MatMul:product:02model_4/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_1/BiasAdd?
model_4/enc_inner_1/ReluRelu$model_4/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_1/Relu?
)model_4/enc_inner_0/MatMul/ReadVariableOpReadVariableOp2model_4_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_4/enc_inner_0/MatMul/ReadVariableOp?
model_4/enc_inner_0/MatMulMatMul'model_4/enc_middle_0/Relu:activations:01model_4/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_0/MatMul?
*model_4/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_4/enc_inner_0/BiasAdd/ReadVariableOp?
model_4/enc_inner_0/BiasAddBiasAdd$model_4/enc_inner_0/MatMul:product:02model_4/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_0/BiasAdd?
model_4/enc_inner_0/ReluRelu$model_4/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_0/Relu?
'model_4/channel_2/MatMul/ReadVariableOpReadVariableOp0model_4_channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_4/channel_2/MatMul/ReadVariableOp?
model_4/channel_2/MatMulMatMul&model_4/enc_inner_2/Relu:activations:0/model_4/channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_2/MatMul?
(model_4/channel_2/BiasAdd/ReadVariableOpReadVariableOp1model_4_channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_4/channel_2/BiasAdd/ReadVariableOp?
model_4/channel_2/BiasAddBiasAdd"model_4/channel_2/MatMul:product:00model_4/channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_2/BiasAdd?
model_4/channel_2/SoftsignSoftsign"model_4/channel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/channel_2/Softsign?
'model_4/channel_1/MatMul/ReadVariableOpReadVariableOp0model_4_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_4/channel_1/MatMul/ReadVariableOp?
model_4/channel_1/MatMulMatMul&model_4/enc_inner_1/Relu:activations:0/model_4/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_1/MatMul?
(model_4/channel_1/BiasAdd/ReadVariableOpReadVariableOp1model_4_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_4/channel_1/BiasAdd/ReadVariableOp?
model_4/channel_1/BiasAddBiasAdd"model_4/channel_1/MatMul:product:00model_4/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_1/BiasAdd?
model_4/channel_1/SoftsignSoftsign"model_4/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/channel_1/Softsign?
'model_4/channel_0/MatMul/ReadVariableOpReadVariableOp0model_4_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_4/channel_0/MatMul/ReadVariableOp?
model_4/channel_0/MatMulMatMul&model_4/enc_inner_0/Relu:activations:0/model_4/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_0/MatMul?
(model_4/channel_0/BiasAdd/ReadVariableOpReadVariableOp1model_4_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_4/channel_0/BiasAdd/ReadVariableOp?
model_4/channel_0/BiasAddBiasAdd"model_4/channel_0/MatMul:product:00model_4/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_0/BiasAdd?
model_4/channel_0/SoftsignSoftsign"model_4/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/channel_0/Softsign?
)model_5/dec_inner_2/MatMul/ReadVariableOpReadVariableOp2model_5_dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_5/dec_inner_2/MatMul/ReadVariableOp?
model_5/dec_inner_2/MatMulMatMul(model_4/channel_2/Softsign:activations:01model_5/dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_2/MatMul?
*model_5/dec_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_5/dec_inner_2/BiasAdd/ReadVariableOp?
model_5/dec_inner_2/BiasAddBiasAdd$model_5/dec_inner_2/MatMul:product:02model_5/dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_2/BiasAdd?
model_5/dec_inner_2/ReluRelu$model_5/dec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_2/Relu?
)model_5/dec_inner_1/MatMul/ReadVariableOpReadVariableOp2model_5_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_5/dec_inner_1/MatMul/ReadVariableOp?
model_5/dec_inner_1/MatMulMatMul(model_4/channel_1/Softsign:activations:01model_5/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_1/MatMul?
*model_5/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_5/dec_inner_1/BiasAdd/ReadVariableOp?
model_5/dec_inner_1/BiasAddBiasAdd$model_5/dec_inner_1/MatMul:product:02model_5/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_1/BiasAdd?
model_5/dec_inner_1/ReluRelu$model_5/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_1/Relu?
)model_5/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_5_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_5/dec_inner_0/MatMul/ReadVariableOp?
model_5/dec_inner_0/MatMulMatMul(model_4/channel_0/Softsign:activations:01model_5/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_0/MatMul?
*model_5/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_5/dec_inner_0/BiasAdd/ReadVariableOp?
model_5/dec_inner_0/BiasAddBiasAdd$model_5/dec_inner_0/MatMul:product:02model_5/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_0/BiasAdd?
model_5/dec_inner_0/ReluRelu$model_5/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_0/Relu?
*model_5/dec_middle_2/MatMul/ReadVariableOpReadVariableOp3model_5_dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_5/dec_middle_2/MatMul/ReadVariableOp?
model_5/dec_middle_2/MatMulMatMul&model_5/dec_inner_2/Relu:activations:02model_5/dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_2/MatMul?
+model_5/dec_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_5_dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_5/dec_middle_2/BiasAdd/ReadVariableOp?
model_5/dec_middle_2/BiasAddBiasAdd%model_5/dec_middle_2/MatMul:product:03model_5/dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_2/BiasAdd?
model_5/dec_middle_2/ReluRelu%model_5/dec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_2/Relu?
*model_5/dec_middle_1/MatMul/ReadVariableOpReadVariableOp3model_5_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_5/dec_middle_1/MatMul/ReadVariableOp?
model_5/dec_middle_1/MatMulMatMul&model_5/dec_inner_1/Relu:activations:02model_5/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_1/MatMul?
+model_5/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_5_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_5/dec_middle_1/BiasAdd/ReadVariableOp?
model_5/dec_middle_1/BiasAddBiasAdd%model_5/dec_middle_1/MatMul:product:03model_5/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_1/BiasAdd?
model_5/dec_middle_1/ReluRelu%model_5/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_1/Relu?
*model_5/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_5_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_5/dec_middle_0/MatMul/ReadVariableOp?
model_5/dec_middle_0/MatMulMatMul&model_5/dec_inner_0/Relu:activations:02model_5/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_0/MatMul?
+model_5/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_5_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_5/dec_middle_0/BiasAdd/ReadVariableOp?
model_5/dec_middle_0/BiasAddBiasAdd%model_5/dec_middle_0/MatMul:product:03model_5/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_0/BiasAdd?
model_5/dec_middle_0/ReluRelu%model_5/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_0/Relu?
)model_5/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_5_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_5/dec_outer_0/MatMul/ReadVariableOp?
model_5/dec_outer_0/MatMulMatMul'model_5/dec_middle_0/Relu:activations:01model_5/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_0/MatMul?
*model_5/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_5/dec_outer_0/BiasAdd/ReadVariableOp?
model_5/dec_outer_0/BiasAddBiasAdd$model_5/dec_outer_0/MatMul:product:02model_5/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_0/BiasAdd?
model_5/dec_outer_0/ReluRelu$model_5/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_0/Relu?
)model_5/dec_outer_1/MatMul/ReadVariableOpReadVariableOp2model_5_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_5/dec_outer_1/MatMul/ReadVariableOp?
model_5/dec_outer_1/MatMulMatMul'model_5/dec_middle_1/Relu:activations:01model_5/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_1/MatMul?
*model_5/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_5/dec_outer_1/BiasAdd/ReadVariableOp?
model_5/dec_outer_1/BiasAddBiasAdd$model_5/dec_outer_1/MatMul:product:02model_5/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_1/BiasAdd?
model_5/dec_outer_1/ReluRelu$model_5/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_1/Relu?
)model_5/dec_outer_2/MatMul/ReadVariableOpReadVariableOp2model_5_dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_5/dec_outer_2/MatMul/ReadVariableOp?
model_5/dec_outer_2/MatMulMatMul'model_5/dec_middle_2/Relu:activations:01model_5/dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_2/MatMul?
*model_5/dec_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_5/dec_outer_2/BiasAdd/ReadVariableOp?
model_5/dec_outer_2/BiasAddBiasAdd$model_5/dec_outer_2/MatMul:product:02model_5/dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_2/BiasAdd?
model_5/dec_outer_2/ReluRelu$model_5/dec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_2/Relu?
model_5/tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model_5/tf.concat_1/concat/axis?
model_5/tf.concat_1/concatConcatV2&model_5/dec_outer_0/Relu:activations:0&model_5/dec_outer_1/Relu:activations:0&model_5/dec_outer_2/Relu:activations:0(model_5/tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_5/tf.concat_1/concat?
(model_5/dec_output/MatMul/ReadVariableOpReadVariableOp1model_5_dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_5/dec_output/MatMul/ReadVariableOp?
model_5/dec_output/MatMulMatMul#model_5/tf.concat_1/concat:output:00model_5/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_5/dec_output/MatMul?
)model_5/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_5_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_5/dec_output/BiasAdd/ReadVariableOp?
model_5/dec_output/BiasAddBiasAdd#model_5/dec_output/MatMul:product:01model_5/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_5/dec_output/BiasAdd?
model_5/dec_output/SigmoidSigmoid#model_5/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_5/dec_output/Sigmoid?
IdentityIdentitymodel_5/dec_output/Sigmoid:y:0)^model_4/channel_0/BiasAdd/ReadVariableOp(^model_4/channel_0/MatMul/ReadVariableOp)^model_4/channel_1/BiasAdd/ReadVariableOp(^model_4/channel_1/MatMul/ReadVariableOp)^model_4/channel_2/BiasAdd/ReadVariableOp(^model_4/channel_2/MatMul/ReadVariableOp+^model_4/enc_inner_0/BiasAdd/ReadVariableOp*^model_4/enc_inner_0/MatMul/ReadVariableOp+^model_4/enc_inner_1/BiasAdd/ReadVariableOp*^model_4/enc_inner_1/MatMul/ReadVariableOp+^model_4/enc_inner_2/BiasAdd/ReadVariableOp*^model_4/enc_inner_2/MatMul/ReadVariableOp,^model_4/enc_middle_0/BiasAdd/ReadVariableOp+^model_4/enc_middle_0/MatMul/ReadVariableOp,^model_4/enc_middle_1/BiasAdd/ReadVariableOp+^model_4/enc_middle_1/MatMul/ReadVariableOp,^model_4/enc_middle_2/BiasAdd/ReadVariableOp+^model_4/enc_middle_2/MatMul/ReadVariableOp+^model_4/enc_outer_0/BiasAdd/ReadVariableOp*^model_4/enc_outer_0/MatMul/ReadVariableOp+^model_4/enc_outer_1/BiasAdd/ReadVariableOp*^model_4/enc_outer_1/MatMul/ReadVariableOp+^model_4/enc_outer_2/BiasAdd/ReadVariableOp*^model_4/enc_outer_2/MatMul/ReadVariableOp+^model_5/dec_inner_0/BiasAdd/ReadVariableOp*^model_5/dec_inner_0/MatMul/ReadVariableOp+^model_5/dec_inner_1/BiasAdd/ReadVariableOp*^model_5/dec_inner_1/MatMul/ReadVariableOp+^model_5/dec_inner_2/BiasAdd/ReadVariableOp*^model_5/dec_inner_2/MatMul/ReadVariableOp,^model_5/dec_middle_0/BiasAdd/ReadVariableOp+^model_5/dec_middle_0/MatMul/ReadVariableOp,^model_5/dec_middle_1/BiasAdd/ReadVariableOp+^model_5/dec_middle_1/MatMul/ReadVariableOp,^model_5/dec_middle_2/BiasAdd/ReadVariableOp+^model_5/dec_middle_2/MatMul/ReadVariableOp+^model_5/dec_outer_0/BiasAdd/ReadVariableOp*^model_5/dec_outer_0/MatMul/ReadVariableOp+^model_5/dec_outer_1/BiasAdd/ReadVariableOp*^model_5/dec_outer_1/MatMul/ReadVariableOp+^model_5/dec_outer_2/BiasAdd/ReadVariableOp*^model_5/dec_outer_2/MatMul/ReadVariableOp*^model_5/dec_output/BiasAdd/ReadVariableOp)^model_5/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2T
(model_4/channel_0/BiasAdd/ReadVariableOp(model_4/channel_0/BiasAdd/ReadVariableOp2R
'model_4/channel_0/MatMul/ReadVariableOp'model_4/channel_0/MatMul/ReadVariableOp2T
(model_4/channel_1/BiasAdd/ReadVariableOp(model_4/channel_1/BiasAdd/ReadVariableOp2R
'model_4/channel_1/MatMul/ReadVariableOp'model_4/channel_1/MatMul/ReadVariableOp2T
(model_4/channel_2/BiasAdd/ReadVariableOp(model_4/channel_2/BiasAdd/ReadVariableOp2R
'model_4/channel_2/MatMul/ReadVariableOp'model_4/channel_2/MatMul/ReadVariableOp2X
*model_4/enc_inner_0/BiasAdd/ReadVariableOp*model_4/enc_inner_0/BiasAdd/ReadVariableOp2V
)model_4/enc_inner_0/MatMul/ReadVariableOp)model_4/enc_inner_0/MatMul/ReadVariableOp2X
*model_4/enc_inner_1/BiasAdd/ReadVariableOp*model_4/enc_inner_1/BiasAdd/ReadVariableOp2V
)model_4/enc_inner_1/MatMul/ReadVariableOp)model_4/enc_inner_1/MatMul/ReadVariableOp2X
*model_4/enc_inner_2/BiasAdd/ReadVariableOp*model_4/enc_inner_2/BiasAdd/ReadVariableOp2V
)model_4/enc_inner_2/MatMul/ReadVariableOp)model_4/enc_inner_2/MatMul/ReadVariableOp2Z
+model_4/enc_middle_0/BiasAdd/ReadVariableOp+model_4/enc_middle_0/BiasAdd/ReadVariableOp2X
*model_4/enc_middle_0/MatMul/ReadVariableOp*model_4/enc_middle_0/MatMul/ReadVariableOp2Z
+model_4/enc_middle_1/BiasAdd/ReadVariableOp+model_4/enc_middle_1/BiasAdd/ReadVariableOp2X
*model_4/enc_middle_1/MatMul/ReadVariableOp*model_4/enc_middle_1/MatMul/ReadVariableOp2Z
+model_4/enc_middle_2/BiasAdd/ReadVariableOp+model_4/enc_middle_2/BiasAdd/ReadVariableOp2X
*model_4/enc_middle_2/MatMul/ReadVariableOp*model_4/enc_middle_2/MatMul/ReadVariableOp2X
*model_4/enc_outer_0/BiasAdd/ReadVariableOp*model_4/enc_outer_0/BiasAdd/ReadVariableOp2V
)model_4/enc_outer_0/MatMul/ReadVariableOp)model_4/enc_outer_0/MatMul/ReadVariableOp2X
*model_4/enc_outer_1/BiasAdd/ReadVariableOp*model_4/enc_outer_1/BiasAdd/ReadVariableOp2V
)model_4/enc_outer_1/MatMul/ReadVariableOp)model_4/enc_outer_1/MatMul/ReadVariableOp2X
*model_4/enc_outer_2/BiasAdd/ReadVariableOp*model_4/enc_outer_2/BiasAdd/ReadVariableOp2V
)model_4/enc_outer_2/MatMul/ReadVariableOp)model_4/enc_outer_2/MatMul/ReadVariableOp2X
*model_5/dec_inner_0/BiasAdd/ReadVariableOp*model_5/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_5/dec_inner_0/MatMul/ReadVariableOp)model_5/dec_inner_0/MatMul/ReadVariableOp2X
*model_5/dec_inner_1/BiasAdd/ReadVariableOp*model_5/dec_inner_1/BiasAdd/ReadVariableOp2V
)model_5/dec_inner_1/MatMul/ReadVariableOp)model_5/dec_inner_1/MatMul/ReadVariableOp2X
*model_5/dec_inner_2/BiasAdd/ReadVariableOp*model_5/dec_inner_2/BiasAdd/ReadVariableOp2V
)model_5/dec_inner_2/MatMul/ReadVariableOp)model_5/dec_inner_2/MatMul/ReadVariableOp2Z
+model_5/dec_middle_0/BiasAdd/ReadVariableOp+model_5/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_5/dec_middle_0/MatMul/ReadVariableOp*model_5/dec_middle_0/MatMul/ReadVariableOp2Z
+model_5/dec_middle_1/BiasAdd/ReadVariableOp+model_5/dec_middle_1/BiasAdd/ReadVariableOp2X
*model_5/dec_middle_1/MatMul/ReadVariableOp*model_5/dec_middle_1/MatMul/ReadVariableOp2Z
+model_5/dec_middle_2/BiasAdd/ReadVariableOp+model_5/dec_middle_2/BiasAdd/ReadVariableOp2X
*model_5/dec_middle_2/MatMul/ReadVariableOp*model_5/dec_middle_2/MatMul/ReadVariableOp2X
*model_5/dec_outer_0/BiasAdd/ReadVariableOp*model_5/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_5/dec_outer_0/MatMul/ReadVariableOp)model_5/dec_outer_0/MatMul/ReadVariableOp2X
*model_5/dec_outer_1/BiasAdd/ReadVariableOp*model_5/dec_outer_1/BiasAdd/ReadVariableOp2V
)model_5/dec_outer_1/MatMul/ReadVariableOp)model_5/dec_outer_1/MatMul/ReadVariableOp2X
*model_5/dec_outer_2/BiasAdd/ReadVariableOp*model_5/dec_outer_2/BiasAdd/ReadVariableOp2V
)model_5/dec_outer_2/MatMul/ReadVariableOp)model_5/dec_outer_2/MatMul/ReadVariableOp2V
)model_5/dec_output/BiasAdd/ReadVariableOp)model_5/dec_output/BiasAdd/ReadVariableOp2T
(model_5/dec_output/MatMul/ReadVariableOp(model_5/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?h
?
C__inference_model_5_layer_call_and_return_conditional_losses_239386
inputs_0
inputs_1
inputs_2.
*dec_inner_2_matmul_readvariableop_resource/
+dec_inner_2_biasadd_readvariableop_resource.
*dec_inner_1_matmul_readvariableop_resource/
+dec_inner_1_biasadd_readvariableop_resource.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_2_matmul_readvariableop_resource0
,dec_middle_2_biasadd_readvariableop_resource/
+dec_middle_1_matmul_readvariableop_resource0
,dec_middle_1_biasadd_readvariableop_resource/
+dec_middle_0_matmul_readvariableop_resource0
,dec_middle_0_biasadd_readvariableop_resource.
*dec_outer_0_matmul_readvariableop_resource/
+dec_outer_0_biasadd_readvariableop_resource.
*dec_outer_1_matmul_readvariableop_resource/
+dec_outer_1_biasadd_readvariableop_resource.
*dec_outer_2_matmul_readvariableop_resource/
+dec_outer_2_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?"dec_inner_1/BiasAdd/ReadVariableOp?!dec_inner_1/MatMul/ReadVariableOp?"dec_inner_2/BiasAdd/ReadVariableOp?!dec_inner_2/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?#dec_middle_1/BiasAdd/ReadVariableOp?"dec_middle_1/MatMul/ReadVariableOp?#dec_middle_2/BiasAdd/ReadVariableOp?"dec_middle_2/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?"dec_outer_1/BiasAdd/ReadVariableOp?!dec_outer_1/MatMul/ReadVariableOp?"dec_outer_2/BiasAdd/ReadVariableOp?!dec_outer_2/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
!dec_inner_2/MatMul/ReadVariableOpReadVariableOp*dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_2/MatMul/ReadVariableOp?
dec_inner_2/MatMulMatMulinputs_2)dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_2/MatMul?
"dec_inner_2/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_2/BiasAdd/ReadVariableOp?
dec_inner_2/BiasAddBiasAdddec_inner_2/MatMul:product:0*dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_2/BiasAdd|
dec_inner_2/ReluReludec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_2/Relu?
!dec_inner_1/MatMul/ReadVariableOpReadVariableOp*dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_1/MatMul/ReadVariableOp?
dec_inner_1/MatMulMatMulinputs_1)dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_1/MatMul?
"dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_1/BiasAdd/ReadVariableOp?
dec_inner_1/BiasAddBiasAdddec_inner_1/MatMul:product:0*dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_1/BiasAdd|
dec_inner_1/ReluReludec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_1/Relu?
!dec_inner_0/MatMul/ReadVariableOpReadVariableOp*dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_0/MatMul/ReadVariableOp?
dec_inner_0/MatMulMatMulinputs_0)dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_0/MatMul?
"dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_0/BiasAdd/ReadVariableOp?
dec_inner_0/BiasAddBiasAdddec_inner_0/MatMul:product:0*dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_0/BiasAdd|
dec_inner_0/ReluReludec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_0/Relu?
"dec_middle_2/MatMul/ReadVariableOpReadVariableOp+dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_2/MatMul/ReadVariableOp?
dec_middle_2/MatMulMatMuldec_inner_2/Relu:activations:0*dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_2/MatMul?
#dec_middle_2/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_2/BiasAdd/ReadVariableOp?
dec_middle_2/BiasAddBiasAdddec_middle_2/MatMul:product:0+dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_2/BiasAdd
dec_middle_2/ReluReludec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_2/Relu?
"dec_middle_1/MatMul/ReadVariableOpReadVariableOp+dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_1/MatMul/ReadVariableOp?
dec_middle_1/MatMulMatMuldec_inner_1/Relu:activations:0*dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_1/MatMul?
#dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_1/BiasAdd/ReadVariableOp?
dec_middle_1/BiasAddBiasAdddec_middle_1/MatMul:product:0+dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_1/BiasAdd
dec_middle_1/ReluReludec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_1/Relu?
"dec_middle_0/MatMul/ReadVariableOpReadVariableOp+dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_0/MatMul/ReadVariableOp?
dec_middle_0/MatMulMatMuldec_inner_0/Relu:activations:0*dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_0/MatMul?
#dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_0/BiasAdd/ReadVariableOp?
dec_middle_0/BiasAddBiasAdddec_middle_0/MatMul:product:0+dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_0/BiasAdd
dec_middle_0/ReluReludec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_0/Relu?
!dec_outer_0/MatMul/ReadVariableOpReadVariableOp*dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_0/MatMul/ReadVariableOp?
dec_outer_0/MatMulMatMuldec_middle_0/Relu:activations:0)dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_0/MatMul?
"dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_0/BiasAdd/ReadVariableOp?
dec_outer_0/BiasAddBiasAdddec_outer_0/MatMul:product:0*dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_0/BiasAdd|
dec_outer_0/ReluReludec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_0/Relu?
!dec_outer_1/MatMul/ReadVariableOpReadVariableOp*dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_1/MatMul/ReadVariableOp?
dec_outer_1/MatMulMatMuldec_middle_1/Relu:activations:0)dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_1/MatMul?
"dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_1/BiasAdd/ReadVariableOp?
dec_outer_1/BiasAddBiasAdddec_outer_1/MatMul:product:0*dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_1/BiasAdd|
dec_outer_1/ReluReludec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_1/Relu?
!dec_outer_2/MatMul/ReadVariableOpReadVariableOp*dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_2/MatMul/ReadVariableOp?
dec_outer_2/MatMulMatMuldec_middle_2/Relu:activations:0)dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_2/MatMul?
"dec_outer_2/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_2/BiasAdd/ReadVariableOp?
dec_outer_2/BiasAddBiasAdddec_outer_2/MatMul:product:0*dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_2/BiasAdd|
dec_outer_2/ReluReludec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_2/Relut
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2dec_outer_0/Relu:activations:0dec_outer_1/Relu:activations:0dec_outer_2/Relu:activations:0 tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_1/concat?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.concat_1/concat:output:0(dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_output/MatMul?
!dec_output/BiasAdd/ReadVariableOpReadVariableOp*dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!dec_output/BiasAdd/ReadVariableOp?
dec_output/BiasAddBiasAdddec_output/MatMul:product:0)dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_output/BiasAdd?
dec_output/SigmoidSigmoiddec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp#^dec_inner_1/BiasAdd/ReadVariableOp"^dec_inner_1/MatMul/ReadVariableOp#^dec_inner_2/BiasAdd/ReadVariableOp"^dec_inner_2/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp$^dec_middle_1/BiasAdd/ReadVariableOp#^dec_middle_1/MatMul/ReadVariableOp$^dec_middle_2/BiasAdd/ReadVariableOp#^dec_middle_2/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp#^dec_outer_1/BiasAdd/ReadVariableOp"^dec_outer_1/MatMul/ReadVariableOp#^dec_outer_2/BiasAdd/ReadVariableOp"^dec_outer_2/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2H
"dec_inner_1/BiasAdd/ReadVariableOp"dec_inner_1/BiasAdd/ReadVariableOp2F
!dec_inner_1/MatMul/ReadVariableOp!dec_inner_1/MatMul/ReadVariableOp2H
"dec_inner_2/BiasAdd/ReadVariableOp"dec_inner_2/BiasAdd/ReadVariableOp2F
!dec_inner_2/MatMul/ReadVariableOp!dec_inner_2/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2J
#dec_middle_1/BiasAdd/ReadVariableOp#dec_middle_1/BiasAdd/ReadVariableOp2H
"dec_middle_1/MatMul/ReadVariableOp"dec_middle_1/MatMul/ReadVariableOp2J
#dec_middle_2/BiasAdd/ReadVariableOp#dec_middle_2/BiasAdd/ReadVariableOp2H
"dec_middle_2/MatMul/ReadVariableOp"dec_middle_2/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2H
"dec_outer_1/BiasAdd/ReadVariableOp"dec_outer_1/BiasAdd/ReadVariableOp2F
!dec_outer_1/MatMul/ReadVariableOp!dec_outer_1/MatMul/ReadVariableOp2H
"dec_outer_2/BiasAdd/ReadVariableOp"dec_outer_2/BiasAdd/ReadVariableOp2F
!dec_outer_2/MatMul/ReadVariableOp!dec_outer_2/MatMul/ReadVariableOp2F
!dec_output/BiasAdd/ReadVariableOp!dec_output/BiasAdd/ReadVariableOp2D
 dec_output/MatMul/ReadVariableOp dec_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
,__inference_enc_outer_0_layer_call_fn_239578

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_2365022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_enc_outer_2_layer_call_fn_239618

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_2364482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_enc_inner_2_layer_call_fn_239738

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_2366102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
+__inference_dec_output_layer_call_fn_239998

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dec_output_layer_call_and_return_conditional_losses_2373392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_237930
input_1
model_4_237743
model_4_237745
model_4_237747
model_4_237749
model_4_237751
model_4_237753
model_4_237755
model_4_237757
model_4_237759
model_4_237761
model_4_237763
model_4_237765
model_4_237767
model_4_237769
model_4_237771
model_4_237773
model_4_237775
model_4_237777
model_4_237779
model_4_237781
model_4_237783
model_4_237785
model_4_237787
model_4_237789
model_5_237888
model_5_237890
model_5_237892
model_5_237894
model_5_237896
model_5_237898
model_5_237900
model_5_237902
model_5_237904
model_5_237906
model_5_237908
model_5_237910
model_5_237912
model_5_237914
model_5_237916
model_5_237918
model_5_237920
model_5_237922
model_5_237924
model_5_237926
identity??model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?
model_4/StatefulPartitionedCallStatefulPartitionedCallinput_1model_4_237743model_4_237745model_4_237747model_4_237749model_4_237751model_4_237753model_4_237755model_4_237757model_4_237759model_4_237761model_4_237763model_4_237765model_4_237767model_4_237769model_4_237771model_4_237773model_4_237775model_4_237777model_4_237779model_4_237781model_4_237783model_4_237785model_4_237787model_4_237789*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2368992!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0(model_4/StatefulPartitionedCall:output:1(model_4/StatefulPartitionedCall:output:2model_5_237888model_5_237890model_5_237892model_5_237894model_5_237896model_5_237898model_5_237900model_5_237902model_5_237904model_5_237906model_5_237908model_5_237910model_5_237912model_5_237914model_5_237916model_5_237918model_5_237920model_5_237922model_5_237924model_5_237926*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2374772!
model_5/StatefulPartitionedCall?
IdentityIdentity(model_5/StatefulPartitionedCall:output:0 ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?#
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238828
x6
2model_4_enc_outer_2_matmul_readvariableop_resource7
3model_4_enc_outer_2_biasadd_readvariableop_resource6
2model_4_enc_outer_1_matmul_readvariableop_resource7
3model_4_enc_outer_1_biasadd_readvariableop_resource6
2model_4_enc_outer_0_matmul_readvariableop_resource7
3model_4_enc_outer_0_biasadd_readvariableop_resource7
3model_4_enc_middle_2_matmul_readvariableop_resource8
4model_4_enc_middle_2_biasadd_readvariableop_resource7
3model_4_enc_middle_1_matmul_readvariableop_resource8
4model_4_enc_middle_1_biasadd_readvariableop_resource7
3model_4_enc_middle_0_matmul_readvariableop_resource8
4model_4_enc_middle_0_biasadd_readvariableop_resource6
2model_4_enc_inner_2_matmul_readvariableop_resource7
3model_4_enc_inner_2_biasadd_readvariableop_resource6
2model_4_enc_inner_1_matmul_readvariableop_resource7
3model_4_enc_inner_1_biasadd_readvariableop_resource6
2model_4_enc_inner_0_matmul_readvariableop_resource7
3model_4_enc_inner_0_biasadd_readvariableop_resource4
0model_4_channel_2_matmul_readvariableop_resource5
1model_4_channel_2_biasadd_readvariableop_resource4
0model_4_channel_1_matmul_readvariableop_resource5
1model_4_channel_1_biasadd_readvariableop_resource4
0model_4_channel_0_matmul_readvariableop_resource5
1model_4_channel_0_biasadd_readvariableop_resource6
2model_5_dec_inner_2_matmul_readvariableop_resource7
3model_5_dec_inner_2_biasadd_readvariableop_resource6
2model_5_dec_inner_1_matmul_readvariableop_resource7
3model_5_dec_inner_1_biasadd_readvariableop_resource6
2model_5_dec_inner_0_matmul_readvariableop_resource7
3model_5_dec_inner_0_biasadd_readvariableop_resource7
3model_5_dec_middle_2_matmul_readvariableop_resource8
4model_5_dec_middle_2_biasadd_readvariableop_resource7
3model_5_dec_middle_1_matmul_readvariableop_resource8
4model_5_dec_middle_1_biasadd_readvariableop_resource7
3model_5_dec_middle_0_matmul_readvariableop_resource8
4model_5_dec_middle_0_biasadd_readvariableop_resource6
2model_5_dec_outer_0_matmul_readvariableop_resource7
3model_5_dec_outer_0_biasadd_readvariableop_resource6
2model_5_dec_outer_1_matmul_readvariableop_resource7
3model_5_dec_outer_1_biasadd_readvariableop_resource6
2model_5_dec_outer_2_matmul_readvariableop_resource7
3model_5_dec_outer_2_biasadd_readvariableop_resource5
1model_5_dec_output_matmul_readvariableop_resource6
2model_5_dec_output_biasadd_readvariableop_resource
identity??(model_4/channel_0/BiasAdd/ReadVariableOp?'model_4/channel_0/MatMul/ReadVariableOp?(model_4/channel_1/BiasAdd/ReadVariableOp?'model_4/channel_1/MatMul/ReadVariableOp?(model_4/channel_2/BiasAdd/ReadVariableOp?'model_4/channel_2/MatMul/ReadVariableOp?*model_4/enc_inner_0/BiasAdd/ReadVariableOp?)model_4/enc_inner_0/MatMul/ReadVariableOp?*model_4/enc_inner_1/BiasAdd/ReadVariableOp?)model_4/enc_inner_1/MatMul/ReadVariableOp?*model_4/enc_inner_2/BiasAdd/ReadVariableOp?)model_4/enc_inner_2/MatMul/ReadVariableOp?+model_4/enc_middle_0/BiasAdd/ReadVariableOp?*model_4/enc_middle_0/MatMul/ReadVariableOp?+model_4/enc_middle_1/BiasAdd/ReadVariableOp?*model_4/enc_middle_1/MatMul/ReadVariableOp?+model_4/enc_middle_2/BiasAdd/ReadVariableOp?*model_4/enc_middle_2/MatMul/ReadVariableOp?*model_4/enc_outer_0/BiasAdd/ReadVariableOp?)model_4/enc_outer_0/MatMul/ReadVariableOp?*model_4/enc_outer_1/BiasAdd/ReadVariableOp?)model_4/enc_outer_1/MatMul/ReadVariableOp?*model_4/enc_outer_2/BiasAdd/ReadVariableOp?)model_4/enc_outer_2/MatMul/ReadVariableOp?*model_5/dec_inner_0/BiasAdd/ReadVariableOp?)model_5/dec_inner_0/MatMul/ReadVariableOp?*model_5/dec_inner_1/BiasAdd/ReadVariableOp?)model_5/dec_inner_1/MatMul/ReadVariableOp?*model_5/dec_inner_2/BiasAdd/ReadVariableOp?)model_5/dec_inner_2/MatMul/ReadVariableOp?+model_5/dec_middle_0/BiasAdd/ReadVariableOp?*model_5/dec_middle_0/MatMul/ReadVariableOp?+model_5/dec_middle_1/BiasAdd/ReadVariableOp?*model_5/dec_middle_1/MatMul/ReadVariableOp?+model_5/dec_middle_2/BiasAdd/ReadVariableOp?*model_5/dec_middle_2/MatMul/ReadVariableOp?*model_5/dec_outer_0/BiasAdd/ReadVariableOp?)model_5/dec_outer_0/MatMul/ReadVariableOp?*model_5/dec_outer_1/BiasAdd/ReadVariableOp?)model_5/dec_outer_1/MatMul/ReadVariableOp?*model_5/dec_outer_2/BiasAdd/ReadVariableOp?)model_5/dec_outer_2/MatMul/ReadVariableOp?)model_5/dec_output/BiasAdd/ReadVariableOp?(model_5/dec_output/MatMul/ReadVariableOp?
)model_4/enc_outer_2/MatMul/ReadVariableOpReadVariableOp2model_4_enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_4/enc_outer_2/MatMul/ReadVariableOp?
model_4/enc_outer_2/MatMulMatMulx1model_4/enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_2/MatMul?
*model_4/enc_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_4/enc_outer_2/BiasAdd/ReadVariableOp?
model_4/enc_outer_2/BiasAddBiasAdd$model_4/enc_outer_2/MatMul:product:02model_4/enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_2/BiasAdd?
model_4/enc_outer_2/ReluRelu$model_4/enc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_2/Relu?
)model_4/enc_outer_1/MatMul/ReadVariableOpReadVariableOp2model_4_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_4/enc_outer_1/MatMul/ReadVariableOp?
model_4/enc_outer_1/MatMulMatMulx1model_4/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_1/MatMul?
*model_4/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_4/enc_outer_1/BiasAdd/ReadVariableOp?
model_4/enc_outer_1/BiasAddBiasAdd$model_4/enc_outer_1/MatMul:product:02model_4/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_1/BiasAdd?
model_4/enc_outer_1/ReluRelu$model_4/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_1/Relu?
)model_4/enc_outer_0/MatMul/ReadVariableOpReadVariableOp2model_4_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_4/enc_outer_0/MatMul/ReadVariableOp?
model_4/enc_outer_0/MatMulMatMulx1model_4/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_0/MatMul?
*model_4/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_4/enc_outer_0/BiasAdd/ReadVariableOp?
model_4/enc_outer_0/BiasAddBiasAdd$model_4/enc_outer_0/MatMul:product:02model_4/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_0/BiasAdd?
model_4/enc_outer_0/ReluRelu$model_4/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_4/enc_outer_0/Relu?
*model_4/enc_middle_2/MatMul/ReadVariableOpReadVariableOp3model_4_enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_4/enc_middle_2/MatMul/ReadVariableOp?
model_4/enc_middle_2/MatMulMatMul&model_4/enc_outer_2/Relu:activations:02model_4/enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_2/MatMul?
+model_4/enc_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_4_enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_4/enc_middle_2/BiasAdd/ReadVariableOp?
model_4/enc_middle_2/BiasAddBiasAdd%model_4/enc_middle_2/MatMul:product:03model_4/enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_2/BiasAdd?
model_4/enc_middle_2/ReluRelu%model_4/enc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_2/Relu?
*model_4/enc_middle_1/MatMul/ReadVariableOpReadVariableOp3model_4_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_4/enc_middle_1/MatMul/ReadVariableOp?
model_4/enc_middle_1/MatMulMatMul&model_4/enc_outer_1/Relu:activations:02model_4/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_1/MatMul?
+model_4/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_4_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_4/enc_middle_1/BiasAdd/ReadVariableOp?
model_4/enc_middle_1/BiasAddBiasAdd%model_4/enc_middle_1/MatMul:product:03model_4/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_1/BiasAdd?
model_4/enc_middle_1/ReluRelu%model_4/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_1/Relu?
*model_4/enc_middle_0/MatMul/ReadVariableOpReadVariableOp3model_4_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_4/enc_middle_0/MatMul/ReadVariableOp?
model_4/enc_middle_0/MatMulMatMul&model_4/enc_outer_0/Relu:activations:02model_4/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_0/MatMul?
+model_4/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_4_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_4/enc_middle_0/BiasAdd/ReadVariableOp?
model_4/enc_middle_0/BiasAddBiasAdd%model_4/enc_middle_0/MatMul:product:03model_4/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_0/BiasAdd?
model_4/enc_middle_0/ReluRelu%model_4/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/enc_middle_0/Relu?
)model_4/enc_inner_2/MatMul/ReadVariableOpReadVariableOp2model_4_enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_4/enc_inner_2/MatMul/ReadVariableOp?
model_4/enc_inner_2/MatMulMatMul'model_4/enc_middle_2/Relu:activations:01model_4/enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_2/MatMul?
*model_4/enc_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_4/enc_inner_2/BiasAdd/ReadVariableOp?
model_4/enc_inner_2/BiasAddBiasAdd$model_4/enc_inner_2/MatMul:product:02model_4/enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_2/BiasAdd?
model_4/enc_inner_2/ReluRelu$model_4/enc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_2/Relu?
)model_4/enc_inner_1/MatMul/ReadVariableOpReadVariableOp2model_4_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_4/enc_inner_1/MatMul/ReadVariableOp?
model_4/enc_inner_1/MatMulMatMul'model_4/enc_middle_1/Relu:activations:01model_4/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_1/MatMul?
*model_4/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_4/enc_inner_1/BiasAdd/ReadVariableOp?
model_4/enc_inner_1/BiasAddBiasAdd$model_4/enc_inner_1/MatMul:product:02model_4/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_1/BiasAdd?
model_4/enc_inner_1/ReluRelu$model_4/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_1/Relu?
)model_4/enc_inner_0/MatMul/ReadVariableOpReadVariableOp2model_4_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_4/enc_inner_0/MatMul/ReadVariableOp?
model_4/enc_inner_0/MatMulMatMul'model_4/enc_middle_0/Relu:activations:01model_4/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_0/MatMul?
*model_4/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_4_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_4/enc_inner_0/BiasAdd/ReadVariableOp?
model_4/enc_inner_0/BiasAddBiasAdd$model_4/enc_inner_0/MatMul:product:02model_4/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_0/BiasAdd?
model_4/enc_inner_0/ReluRelu$model_4/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_4/enc_inner_0/Relu?
'model_4/channel_2/MatMul/ReadVariableOpReadVariableOp0model_4_channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_4/channel_2/MatMul/ReadVariableOp?
model_4/channel_2/MatMulMatMul&model_4/enc_inner_2/Relu:activations:0/model_4/channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_2/MatMul?
(model_4/channel_2/BiasAdd/ReadVariableOpReadVariableOp1model_4_channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_4/channel_2/BiasAdd/ReadVariableOp?
model_4/channel_2/BiasAddBiasAdd"model_4/channel_2/MatMul:product:00model_4/channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_2/BiasAdd?
model_4/channel_2/SoftsignSoftsign"model_4/channel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/channel_2/Softsign?
'model_4/channel_1/MatMul/ReadVariableOpReadVariableOp0model_4_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_4/channel_1/MatMul/ReadVariableOp?
model_4/channel_1/MatMulMatMul&model_4/enc_inner_1/Relu:activations:0/model_4/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_1/MatMul?
(model_4/channel_1/BiasAdd/ReadVariableOpReadVariableOp1model_4_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_4/channel_1/BiasAdd/ReadVariableOp?
model_4/channel_1/BiasAddBiasAdd"model_4/channel_1/MatMul:product:00model_4/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_1/BiasAdd?
model_4/channel_1/SoftsignSoftsign"model_4/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/channel_1/Softsign?
'model_4/channel_0/MatMul/ReadVariableOpReadVariableOp0model_4_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_4/channel_0/MatMul/ReadVariableOp?
model_4/channel_0/MatMulMatMul&model_4/enc_inner_0/Relu:activations:0/model_4/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_0/MatMul?
(model_4/channel_0/BiasAdd/ReadVariableOpReadVariableOp1model_4_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_4/channel_0/BiasAdd/ReadVariableOp?
model_4/channel_0/BiasAddBiasAdd"model_4/channel_0/MatMul:product:00model_4/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/channel_0/BiasAdd?
model_4/channel_0/SoftsignSoftsign"model_4/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/channel_0/Softsign?
)model_5/dec_inner_2/MatMul/ReadVariableOpReadVariableOp2model_5_dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_5/dec_inner_2/MatMul/ReadVariableOp?
model_5/dec_inner_2/MatMulMatMul(model_4/channel_2/Softsign:activations:01model_5/dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_2/MatMul?
*model_5/dec_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_5/dec_inner_2/BiasAdd/ReadVariableOp?
model_5/dec_inner_2/BiasAddBiasAdd$model_5/dec_inner_2/MatMul:product:02model_5/dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_2/BiasAdd?
model_5/dec_inner_2/ReluRelu$model_5/dec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_2/Relu?
)model_5/dec_inner_1/MatMul/ReadVariableOpReadVariableOp2model_5_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_5/dec_inner_1/MatMul/ReadVariableOp?
model_5/dec_inner_1/MatMulMatMul(model_4/channel_1/Softsign:activations:01model_5/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_1/MatMul?
*model_5/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_5/dec_inner_1/BiasAdd/ReadVariableOp?
model_5/dec_inner_1/BiasAddBiasAdd$model_5/dec_inner_1/MatMul:product:02model_5/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_1/BiasAdd?
model_5/dec_inner_1/ReluRelu$model_5/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_1/Relu?
)model_5/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_5_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_5/dec_inner_0/MatMul/ReadVariableOp?
model_5/dec_inner_0/MatMulMatMul(model_4/channel_0/Softsign:activations:01model_5/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_0/MatMul?
*model_5/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_5/dec_inner_0/BiasAdd/ReadVariableOp?
model_5/dec_inner_0/BiasAddBiasAdd$model_5/dec_inner_0/MatMul:product:02model_5/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_0/BiasAdd?
model_5/dec_inner_0/ReluRelu$model_5/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_5/dec_inner_0/Relu?
*model_5/dec_middle_2/MatMul/ReadVariableOpReadVariableOp3model_5_dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_5/dec_middle_2/MatMul/ReadVariableOp?
model_5/dec_middle_2/MatMulMatMul&model_5/dec_inner_2/Relu:activations:02model_5/dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_2/MatMul?
+model_5/dec_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_5_dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_5/dec_middle_2/BiasAdd/ReadVariableOp?
model_5/dec_middle_2/BiasAddBiasAdd%model_5/dec_middle_2/MatMul:product:03model_5/dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_2/BiasAdd?
model_5/dec_middle_2/ReluRelu%model_5/dec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_2/Relu?
*model_5/dec_middle_1/MatMul/ReadVariableOpReadVariableOp3model_5_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_5/dec_middle_1/MatMul/ReadVariableOp?
model_5/dec_middle_1/MatMulMatMul&model_5/dec_inner_1/Relu:activations:02model_5/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_1/MatMul?
+model_5/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_5_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_5/dec_middle_1/BiasAdd/ReadVariableOp?
model_5/dec_middle_1/BiasAddBiasAdd%model_5/dec_middle_1/MatMul:product:03model_5/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_1/BiasAdd?
model_5/dec_middle_1/ReluRelu%model_5/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_1/Relu?
*model_5/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_5_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_5/dec_middle_0/MatMul/ReadVariableOp?
model_5/dec_middle_0/MatMulMatMul&model_5/dec_inner_0/Relu:activations:02model_5/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_0/MatMul?
+model_5/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_5_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_5/dec_middle_0/BiasAdd/ReadVariableOp?
model_5/dec_middle_0/BiasAddBiasAdd%model_5/dec_middle_0/MatMul:product:03model_5/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_0/BiasAdd?
model_5/dec_middle_0/ReluRelu%model_5/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_middle_0/Relu?
)model_5/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_5_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_5/dec_outer_0/MatMul/ReadVariableOp?
model_5/dec_outer_0/MatMulMatMul'model_5/dec_middle_0/Relu:activations:01model_5/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_0/MatMul?
*model_5/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_5/dec_outer_0/BiasAdd/ReadVariableOp?
model_5/dec_outer_0/BiasAddBiasAdd$model_5/dec_outer_0/MatMul:product:02model_5/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_0/BiasAdd?
model_5/dec_outer_0/ReluRelu$model_5/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_0/Relu?
)model_5/dec_outer_1/MatMul/ReadVariableOpReadVariableOp2model_5_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_5/dec_outer_1/MatMul/ReadVariableOp?
model_5/dec_outer_1/MatMulMatMul'model_5/dec_middle_1/Relu:activations:01model_5/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_1/MatMul?
*model_5/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_5/dec_outer_1/BiasAdd/ReadVariableOp?
model_5/dec_outer_1/BiasAddBiasAdd$model_5/dec_outer_1/MatMul:product:02model_5/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_1/BiasAdd?
model_5/dec_outer_1/ReluRelu$model_5/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_1/Relu?
)model_5/dec_outer_2/MatMul/ReadVariableOpReadVariableOp2model_5_dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_5/dec_outer_2/MatMul/ReadVariableOp?
model_5/dec_outer_2/MatMulMatMul'model_5/dec_middle_2/Relu:activations:01model_5/dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_2/MatMul?
*model_5/dec_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_5_dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_5/dec_outer_2/BiasAdd/ReadVariableOp?
model_5/dec_outer_2/BiasAddBiasAdd$model_5/dec_outer_2/MatMul:product:02model_5/dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_2/BiasAdd?
model_5/dec_outer_2/ReluRelu$model_5/dec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_5/dec_outer_2/Relu?
model_5/tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model_5/tf.concat_1/concat/axis?
model_5/tf.concat_1/concatConcatV2&model_5/dec_outer_0/Relu:activations:0&model_5/dec_outer_1/Relu:activations:0&model_5/dec_outer_2/Relu:activations:0(model_5/tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_5/tf.concat_1/concat?
(model_5/dec_output/MatMul/ReadVariableOpReadVariableOp1model_5_dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_5/dec_output/MatMul/ReadVariableOp?
model_5/dec_output/MatMulMatMul#model_5/tf.concat_1/concat:output:00model_5/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_5/dec_output/MatMul?
)model_5/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_5_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_5/dec_output/BiasAdd/ReadVariableOp?
model_5/dec_output/BiasAddBiasAdd#model_5/dec_output/MatMul:product:01model_5/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_5/dec_output/BiasAdd?
model_5/dec_output/SigmoidSigmoid#model_5/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_5/dec_output/Sigmoid?
IdentityIdentitymodel_5/dec_output/Sigmoid:y:0)^model_4/channel_0/BiasAdd/ReadVariableOp(^model_4/channel_0/MatMul/ReadVariableOp)^model_4/channel_1/BiasAdd/ReadVariableOp(^model_4/channel_1/MatMul/ReadVariableOp)^model_4/channel_2/BiasAdd/ReadVariableOp(^model_4/channel_2/MatMul/ReadVariableOp+^model_4/enc_inner_0/BiasAdd/ReadVariableOp*^model_4/enc_inner_0/MatMul/ReadVariableOp+^model_4/enc_inner_1/BiasAdd/ReadVariableOp*^model_4/enc_inner_1/MatMul/ReadVariableOp+^model_4/enc_inner_2/BiasAdd/ReadVariableOp*^model_4/enc_inner_2/MatMul/ReadVariableOp,^model_4/enc_middle_0/BiasAdd/ReadVariableOp+^model_4/enc_middle_0/MatMul/ReadVariableOp,^model_4/enc_middle_1/BiasAdd/ReadVariableOp+^model_4/enc_middle_1/MatMul/ReadVariableOp,^model_4/enc_middle_2/BiasAdd/ReadVariableOp+^model_4/enc_middle_2/MatMul/ReadVariableOp+^model_4/enc_outer_0/BiasAdd/ReadVariableOp*^model_4/enc_outer_0/MatMul/ReadVariableOp+^model_4/enc_outer_1/BiasAdd/ReadVariableOp*^model_4/enc_outer_1/MatMul/ReadVariableOp+^model_4/enc_outer_2/BiasAdd/ReadVariableOp*^model_4/enc_outer_2/MatMul/ReadVariableOp+^model_5/dec_inner_0/BiasAdd/ReadVariableOp*^model_5/dec_inner_0/MatMul/ReadVariableOp+^model_5/dec_inner_1/BiasAdd/ReadVariableOp*^model_5/dec_inner_1/MatMul/ReadVariableOp+^model_5/dec_inner_2/BiasAdd/ReadVariableOp*^model_5/dec_inner_2/MatMul/ReadVariableOp,^model_5/dec_middle_0/BiasAdd/ReadVariableOp+^model_5/dec_middle_0/MatMul/ReadVariableOp,^model_5/dec_middle_1/BiasAdd/ReadVariableOp+^model_5/dec_middle_1/MatMul/ReadVariableOp,^model_5/dec_middle_2/BiasAdd/ReadVariableOp+^model_5/dec_middle_2/MatMul/ReadVariableOp+^model_5/dec_outer_0/BiasAdd/ReadVariableOp*^model_5/dec_outer_0/MatMul/ReadVariableOp+^model_5/dec_outer_1/BiasAdd/ReadVariableOp*^model_5/dec_outer_1/MatMul/ReadVariableOp+^model_5/dec_outer_2/BiasAdd/ReadVariableOp*^model_5/dec_outer_2/MatMul/ReadVariableOp*^model_5/dec_output/BiasAdd/ReadVariableOp)^model_5/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2T
(model_4/channel_0/BiasAdd/ReadVariableOp(model_4/channel_0/BiasAdd/ReadVariableOp2R
'model_4/channel_0/MatMul/ReadVariableOp'model_4/channel_0/MatMul/ReadVariableOp2T
(model_4/channel_1/BiasAdd/ReadVariableOp(model_4/channel_1/BiasAdd/ReadVariableOp2R
'model_4/channel_1/MatMul/ReadVariableOp'model_4/channel_1/MatMul/ReadVariableOp2T
(model_4/channel_2/BiasAdd/ReadVariableOp(model_4/channel_2/BiasAdd/ReadVariableOp2R
'model_4/channel_2/MatMul/ReadVariableOp'model_4/channel_2/MatMul/ReadVariableOp2X
*model_4/enc_inner_0/BiasAdd/ReadVariableOp*model_4/enc_inner_0/BiasAdd/ReadVariableOp2V
)model_4/enc_inner_0/MatMul/ReadVariableOp)model_4/enc_inner_0/MatMul/ReadVariableOp2X
*model_4/enc_inner_1/BiasAdd/ReadVariableOp*model_4/enc_inner_1/BiasAdd/ReadVariableOp2V
)model_4/enc_inner_1/MatMul/ReadVariableOp)model_4/enc_inner_1/MatMul/ReadVariableOp2X
*model_4/enc_inner_2/BiasAdd/ReadVariableOp*model_4/enc_inner_2/BiasAdd/ReadVariableOp2V
)model_4/enc_inner_2/MatMul/ReadVariableOp)model_4/enc_inner_2/MatMul/ReadVariableOp2Z
+model_4/enc_middle_0/BiasAdd/ReadVariableOp+model_4/enc_middle_0/BiasAdd/ReadVariableOp2X
*model_4/enc_middle_0/MatMul/ReadVariableOp*model_4/enc_middle_0/MatMul/ReadVariableOp2Z
+model_4/enc_middle_1/BiasAdd/ReadVariableOp+model_4/enc_middle_1/BiasAdd/ReadVariableOp2X
*model_4/enc_middle_1/MatMul/ReadVariableOp*model_4/enc_middle_1/MatMul/ReadVariableOp2Z
+model_4/enc_middle_2/BiasAdd/ReadVariableOp+model_4/enc_middle_2/BiasAdd/ReadVariableOp2X
*model_4/enc_middle_2/MatMul/ReadVariableOp*model_4/enc_middle_2/MatMul/ReadVariableOp2X
*model_4/enc_outer_0/BiasAdd/ReadVariableOp*model_4/enc_outer_0/BiasAdd/ReadVariableOp2V
)model_4/enc_outer_0/MatMul/ReadVariableOp)model_4/enc_outer_0/MatMul/ReadVariableOp2X
*model_4/enc_outer_1/BiasAdd/ReadVariableOp*model_4/enc_outer_1/BiasAdd/ReadVariableOp2V
)model_4/enc_outer_1/MatMul/ReadVariableOp)model_4/enc_outer_1/MatMul/ReadVariableOp2X
*model_4/enc_outer_2/BiasAdd/ReadVariableOp*model_4/enc_outer_2/BiasAdd/ReadVariableOp2V
)model_4/enc_outer_2/MatMul/ReadVariableOp)model_4/enc_outer_2/MatMul/ReadVariableOp2X
*model_5/dec_inner_0/BiasAdd/ReadVariableOp*model_5/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_5/dec_inner_0/MatMul/ReadVariableOp)model_5/dec_inner_0/MatMul/ReadVariableOp2X
*model_5/dec_inner_1/BiasAdd/ReadVariableOp*model_5/dec_inner_1/BiasAdd/ReadVariableOp2V
)model_5/dec_inner_1/MatMul/ReadVariableOp)model_5/dec_inner_1/MatMul/ReadVariableOp2X
*model_5/dec_inner_2/BiasAdd/ReadVariableOp*model_5/dec_inner_2/BiasAdd/ReadVariableOp2V
)model_5/dec_inner_2/MatMul/ReadVariableOp)model_5/dec_inner_2/MatMul/ReadVariableOp2Z
+model_5/dec_middle_0/BiasAdd/ReadVariableOp+model_5/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_5/dec_middle_0/MatMul/ReadVariableOp*model_5/dec_middle_0/MatMul/ReadVariableOp2Z
+model_5/dec_middle_1/BiasAdd/ReadVariableOp+model_5/dec_middle_1/BiasAdd/ReadVariableOp2X
*model_5/dec_middle_1/MatMul/ReadVariableOp*model_5/dec_middle_1/MatMul/ReadVariableOp2Z
+model_5/dec_middle_2/BiasAdd/ReadVariableOp+model_5/dec_middle_2/BiasAdd/ReadVariableOp2X
*model_5/dec_middle_2/MatMul/ReadVariableOp*model_5/dec_middle_2/MatMul/ReadVariableOp2X
*model_5/dec_outer_0/BiasAdd/ReadVariableOp*model_5/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_5/dec_outer_0/MatMul/ReadVariableOp)model_5/dec_outer_0/MatMul/ReadVariableOp2X
*model_5/dec_outer_1/BiasAdd/ReadVariableOp*model_5/dec_outer_1/BiasAdd/ReadVariableOp2V
)model_5/dec_outer_1/MatMul/ReadVariableOp)model_5/dec_outer_1/MatMul/ReadVariableOp2X
*model_5/dec_outer_2/BiasAdd/ReadVariableOp*model_5/dec_outer_2/BiasAdd/ReadVariableOp2V
)model_5/dec_outer_2/MatMul/ReadVariableOp)model_5/dec_outer_2/MatMul/ReadVariableOp2V
)model_5/dec_output/BiasAdd/ReadVariableOp)model_5/dec_output/BiasAdd/ReadVariableOp2T
(model_5/dec_output/MatMul/ReadVariableOp(model_5/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
,__inference_dec_inner_0_layer_call_fn_239818

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_2371482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_239889

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

*__inference_channel_0_layer_call_fn_239758

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_0_layer_call_and_return_conditional_losses_2367452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
(__inference_model_4_layer_call_fn_236954
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2368992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?	
?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_237283

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_239669

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
E__inference_channel_2_layer_call_and_return_conditional_losses_236691

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
E__inference_channel_2_layer_call_and_return_conditional_losses_239789

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
-__inference_enc_middle_1_layer_call_fn_239658

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_2365562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?K
?	
C__inference_model_4_layer_call_and_return_conditional_losses_236830
encoder_input
enc_outer_2_236767
enc_outer_2_236769
enc_outer_1_236772
enc_outer_1_236774
enc_outer_0_236777
enc_outer_0_236779
enc_middle_2_236782
enc_middle_2_236784
enc_middle_1_236787
enc_middle_1_236789
enc_middle_0_236792
enc_middle_0_236794
enc_inner_2_236797
enc_inner_2_236799
enc_inner_1_236802
enc_inner_1_236804
enc_inner_0_236807
enc_inner_0_236809
channel_2_236812
channel_2_236814
channel_1_236817
channel_1_236819
channel_0_236822
channel_0_236824
identity

identity_1

identity_2??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_2_236767enc_outer_2_236769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_2364482%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_1_236772enc_outer_1_236774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_2364752%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_236777enc_outer_0_236779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_2365022%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_236782enc_middle_2_236784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_2365292&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_236787enc_middle_1_236789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_2365562&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_236792enc_middle_0_236794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_2365832&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_236797enc_inner_2_236799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_2366102%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_236802enc_inner_1_236804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_2366372%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_236807enc_inner_0_236809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_2366642%
#enc_inner_0/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_236812channel_2_236814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_2_layer_call_and_return_conditional_losses_2366912#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_236817channel_1_236819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_1_layer_call_and_return_conditional_losses_2367182#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_236822channel_0_236824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_0_layer_call_and_return_conditional_losses_2367452#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?
?
(__inference_model_5_layer_call_fn_237625
decoder_input_0
decoder_input_1
decoder_input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0decoder_input_1decoder_input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2375822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_1:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_2
?	
?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_237256

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?<
?
C__inference_model_5_layer_call_and_return_conditional_losses_237414
decoder_input_0
decoder_input_1
decoder_input_2
dec_inner_2_237361
dec_inner_2_237363
dec_inner_1_237366
dec_inner_1_237368
dec_inner_0_237371
dec_inner_0_237373
dec_middle_2_237376
dec_middle_2_237378
dec_middle_1_237381
dec_middle_1_237383
dec_middle_0_237386
dec_middle_0_237388
dec_outer_0_237391
dec_outer_0_237393
dec_outer_1_237396
dec_outer_1_237398
dec_outer_2_237401
dec_outer_2_237403
dec_output_237408
dec_output_237410
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_2dec_inner_2_237361dec_inner_2_237363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_2370942%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_1dec_inner_1_237366dec_inner_1_237368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_2371212%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_237371dec_inner_0_237373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_2371482%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_237376dec_middle_2_237378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_2371752&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_237381dec_middle_1_237383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_2372022&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_237386dec_middle_0_237388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_2372292&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_237391dec_outer_0_237393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_2372562%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_237396dec_outer_1_237398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_2372832%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_237401dec_outer_2_237403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_2373102%
#dec_outer_2/StatefulPartitionedCallt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_1/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dec_output_237408dec_output_237410*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dec_output_layer_call_and_return_conditional_losses_2373392$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_1:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_2
?
?
(__inference_model_4_layer_call_fn_239251

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2368992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_239949

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_238508
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_2364332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
E__inference_channel_1_layer_call_and_return_conditional_losses_236718

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_237121

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_237229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_236475

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_239589

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_237310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?;
?
C__inference_model_5_layer_call_and_return_conditional_losses_237582

inputs
inputs_1
inputs_2
dec_inner_2_237529
dec_inner_2_237531
dec_inner_1_237534
dec_inner_1_237536
dec_inner_0_237539
dec_inner_0_237541
dec_middle_2_237544
dec_middle_2_237546
dec_middle_1_237549
dec_middle_1_237551
dec_middle_0_237554
dec_middle_0_237556
dec_outer_0_237559
dec_outer_0_237561
dec_outer_1_237564
dec_outer_1_237566
dec_outer_2_237569
dec_outer_2_237571
dec_output_237576
dec_output_237578
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2dec_inner_2_237529dec_inner_2_237531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_2370942%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1dec_inner_1_237534dec_inner_1_237536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_2371212%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_237539dec_inner_0_237541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_2371482%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_237544dec_middle_2_237546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_2371752&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_237549dec_middle_1_237551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_2372022&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_237554dec_middle_0_237556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_2372292&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_237559dec_outer_0_237561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_2372562%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_237564dec_outer_1_237566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_2372832%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_237569dec_outer_2_237571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_2373102%
#dec_outer_2/StatefulPartitionedCallt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_1/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dec_output_237576dec_output_237578*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dec_output_layer_call_and_return_conditional_losses_2373392$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dec_middle_2_layer_call_fn_239918

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_2371752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238026
input_1
model_4_237933
model_4_237935
model_4_237937
model_4_237939
model_4_237941
model_4_237943
model_4_237945
model_4_237947
model_4_237949
model_4_237951
model_4_237953
model_4_237955
model_4_237957
model_4_237959
model_4_237961
model_4_237963
model_4_237965
model_4_237967
model_4_237969
model_4_237971
model_4_237973
model_4_237975
model_4_237977
model_4_237979
model_5_237984
model_5_237986
model_5_237988
model_5_237990
model_5_237992
model_5_237994
model_5_237996
model_5_237998
model_5_238000
model_5_238002
model_5_238004
model_5_238006
model_5_238008
model_5_238010
model_5_238012
model_5_238014
model_5_238016
model_5_238018
model_5_238020
model_5_238022
identity??model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?
model_4/StatefulPartitionedCallStatefulPartitionedCallinput_1model_4_237933model_4_237935model_4_237937model_4_237939model_4_237941model_4_237943model_4_237945model_4_237947model_4_237949model_4_237951model_4_237953model_4_237955model_4_237957model_4_237959model_4_237961model_4_237963model_4_237965model_4_237967model_4_237969model_4_237971model_4_237973model_4_237975model_4_237977model_4_237979*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2370222!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0(model_4/StatefulPartitionedCall:output:1(model_4/StatefulPartitionedCall:output:2model_5_237984model_5_237986model_5_237988model_5_237990model_5_237992model_5_237994model_5_237996model_5_237998model_5_238000model_5_238002model_5_238004model_5_238006model_5_238008model_5_238010model_5_238012model_5_238014model_5_238016model_5_238018model_5_238020model_5_238022*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2375822!
model_5/StatefulPartitionedCall?
IdentityIdentity(model_5/StatefulPartitionedCall:output:0 ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
-__inference_enc_middle_2_layer_call_fn_239678

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_2365292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
(__inference_model_4_layer_call_fn_237077
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2370222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?h
?
C__inference_model_5_layer_call_and_return_conditional_losses_239464
inputs_0
inputs_1
inputs_2.
*dec_inner_2_matmul_readvariableop_resource/
+dec_inner_2_biasadd_readvariableop_resource.
*dec_inner_1_matmul_readvariableop_resource/
+dec_inner_1_biasadd_readvariableop_resource.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_2_matmul_readvariableop_resource0
,dec_middle_2_biasadd_readvariableop_resource/
+dec_middle_1_matmul_readvariableop_resource0
,dec_middle_1_biasadd_readvariableop_resource/
+dec_middle_0_matmul_readvariableop_resource0
,dec_middle_0_biasadd_readvariableop_resource.
*dec_outer_0_matmul_readvariableop_resource/
+dec_outer_0_biasadd_readvariableop_resource.
*dec_outer_1_matmul_readvariableop_resource/
+dec_outer_1_biasadd_readvariableop_resource.
*dec_outer_2_matmul_readvariableop_resource/
+dec_outer_2_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?"dec_inner_1/BiasAdd/ReadVariableOp?!dec_inner_1/MatMul/ReadVariableOp?"dec_inner_2/BiasAdd/ReadVariableOp?!dec_inner_2/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?#dec_middle_1/BiasAdd/ReadVariableOp?"dec_middle_1/MatMul/ReadVariableOp?#dec_middle_2/BiasAdd/ReadVariableOp?"dec_middle_2/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?"dec_outer_1/BiasAdd/ReadVariableOp?!dec_outer_1/MatMul/ReadVariableOp?"dec_outer_2/BiasAdd/ReadVariableOp?!dec_outer_2/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
!dec_inner_2/MatMul/ReadVariableOpReadVariableOp*dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_2/MatMul/ReadVariableOp?
dec_inner_2/MatMulMatMulinputs_2)dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_2/MatMul?
"dec_inner_2/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_2/BiasAdd/ReadVariableOp?
dec_inner_2/BiasAddBiasAdddec_inner_2/MatMul:product:0*dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_2/BiasAdd|
dec_inner_2/ReluReludec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_2/Relu?
!dec_inner_1/MatMul/ReadVariableOpReadVariableOp*dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_1/MatMul/ReadVariableOp?
dec_inner_1/MatMulMatMulinputs_1)dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_1/MatMul?
"dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_1/BiasAdd/ReadVariableOp?
dec_inner_1/BiasAddBiasAdddec_inner_1/MatMul:product:0*dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_1/BiasAdd|
dec_inner_1/ReluReludec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_1/Relu?
!dec_inner_0/MatMul/ReadVariableOpReadVariableOp*dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_0/MatMul/ReadVariableOp?
dec_inner_0/MatMulMatMulinputs_0)dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_0/MatMul?
"dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_0/BiasAdd/ReadVariableOp?
dec_inner_0/BiasAddBiasAdddec_inner_0/MatMul:product:0*dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_0/BiasAdd|
dec_inner_0/ReluReludec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_0/Relu?
"dec_middle_2/MatMul/ReadVariableOpReadVariableOp+dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_2/MatMul/ReadVariableOp?
dec_middle_2/MatMulMatMuldec_inner_2/Relu:activations:0*dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_2/MatMul?
#dec_middle_2/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_2/BiasAdd/ReadVariableOp?
dec_middle_2/BiasAddBiasAdddec_middle_2/MatMul:product:0+dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_2/BiasAdd
dec_middle_2/ReluReludec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_2/Relu?
"dec_middle_1/MatMul/ReadVariableOpReadVariableOp+dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_1/MatMul/ReadVariableOp?
dec_middle_1/MatMulMatMuldec_inner_1/Relu:activations:0*dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_1/MatMul?
#dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_1/BiasAdd/ReadVariableOp?
dec_middle_1/BiasAddBiasAdddec_middle_1/MatMul:product:0+dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_1/BiasAdd
dec_middle_1/ReluReludec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_1/Relu?
"dec_middle_0/MatMul/ReadVariableOpReadVariableOp+dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_0/MatMul/ReadVariableOp?
dec_middle_0/MatMulMatMuldec_inner_0/Relu:activations:0*dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_0/MatMul?
#dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_0/BiasAdd/ReadVariableOp?
dec_middle_0/BiasAddBiasAdddec_middle_0/MatMul:product:0+dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_0/BiasAdd
dec_middle_0/ReluReludec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_0/Relu?
!dec_outer_0/MatMul/ReadVariableOpReadVariableOp*dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_0/MatMul/ReadVariableOp?
dec_outer_0/MatMulMatMuldec_middle_0/Relu:activations:0)dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_0/MatMul?
"dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_0/BiasAdd/ReadVariableOp?
dec_outer_0/BiasAddBiasAdddec_outer_0/MatMul:product:0*dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_0/BiasAdd|
dec_outer_0/ReluReludec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_0/Relu?
!dec_outer_1/MatMul/ReadVariableOpReadVariableOp*dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_1/MatMul/ReadVariableOp?
dec_outer_1/MatMulMatMuldec_middle_1/Relu:activations:0)dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_1/MatMul?
"dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_1/BiasAdd/ReadVariableOp?
dec_outer_1/BiasAddBiasAdddec_outer_1/MatMul:product:0*dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_1/BiasAdd|
dec_outer_1/ReluReludec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_1/Relu?
!dec_outer_2/MatMul/ReadVariableOpReadVariableOp*dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_2/MatMul/ReadVariableOp?
dec_outer_2/MatMulMatMuldec_middle_2/Relu:activations:0)dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_2/MatMul?
"dec_outer_2/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_2/BiasAdd/ReadVariableOp?
dec_outer_2/BiasAddBiasAdddec_outer_2/MatMul:product:0*dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_2/BiasAdd|
dec_outer_2/ReluReludec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_2/Relut
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2dec_outer_0/Relu:activations:0dec_outer_1/Relu:activations:0dec_outer_2/Relu:activations:0 tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_1/concat?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.concat_1/concat:output:0(dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_output/MatMul?
!dec_output/BiasAdd/ReadVariableOpReadVariableOp*dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!dec_output/BiasAdd/ReadVariableOp?
dec_output/BiasAddBiasAdddec_output/MatMul:product:0)dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_output/BiasAdd?
dec_output/SigmoidSigmoiddec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp#^dec_inner_1/BiasAdd/ReadVariableOp"^dec_inner_1/MatMul/ReadVariableOp#^dec_inner_2/BiasAdd/ReadVariableOp"^dec_inner_2/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp$^dec_middle_1/BiasAdd/ReadVariableOp#^dec_middle_1/MatMul/ReadVariableOp$^dec_middle_2/BiasAdd/ReadVariableOp#^dec_middle_2/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp#^dec_outer_1/BiasAdd/ReadVariableOp"^dec_outer_1/MatMul/ReadVariableOp#^dec_outer_2/BiasAdd/ReadVariableOp"^dec_outer_2/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2H
"dec_inner_1/BiasAdd/ReadVariableOp"dec_inner_1/BiasAdd/ReadVariableOp2F
!dec_inner_1/MatMul/ReadVariableOp!dec_inner_1/MatMul/ReadVariableOp2H
"dec_inner_2/BiasAdd/ReadVariableOp"dec_inner_2/BiasAdd/ReadVariableOp2F
!dec_inner_2/MatMul/ReadVariableOp!dec_inner_2/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2J
#dec_middle_1/BiasAdd/ReadVariableOp#dec_middle_1/BiasAdd/ReadVariableOp2H
"dec_middle_1/MatMul/ReadVariableOp"dec_middle_1/MatMul/ReadVariableOp2J
#dec_middle_2/BiasAdd/ReadVariableOp#dec_middle_2/BiasAdd/ReadVariableOp2H
"dec_middle_2/MatMul/ReadVariableOp"dec_middle_2/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2H
"dec_outer_1/BiasAdd/ReadVariableOp"dec_outer_1/BiasAdd/ReadVariableOp2F
!dec_outer_1/MatMul/ReadVariableOp!dec_outer_1/MatMul/ReadVariableOp2H
"dec_outer_2/BiasAdd/ReadVariableOp"dec_outer_2/BiasAdd/ReadVariableOp2F
!dec_outer_2/MatMul/ReadVariableOp!dec_outer_2/MatMul/ReadVariableOp2F
!dec_output/BiasAdd/ReadVariableOp!dec_output/BiasAdd/ReadVariableOp2D
 dec_output/MatMul/ReadVariableOp dec_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?	
?
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_239729

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?J
?	
C__inference_model_4_layer_call_and_return_conditional_losses_237022

inputs
enc_outer_2_236959
enc_outer_2_236961
enc_outer_1_236964
enc_outer_1_236966
enc_outer_0_236969
enc_outer_0_236971
enc_middle_2_236974
enc_middle_2_236976
enc_middle_1_236979
enc_middle_1_236981
enc_middle_0_236984
enc_middle_0_236986
enc_inner_2_236989
enc_inner_2_236991
enc_inner_1_236994
enc_inner_1_236996
enc_inner_0_236999
enc_inner_0_237001
channel_2_237004
channel_2_237006
channel_1_237009
channel_1_237011
channel_0_237014
channel_0_237016
identity

identity_1

identity_2??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_2_236959enc_outer_2_236961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_2364482%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_1_236964enc_outer_1_236966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_2364752%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_236969enc_outer_0_236971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_2365022%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_236974enc_middle_2_236976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_2365292&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_236979enc_middle_1_236981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_2365562&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_236984enc_middle_0_236986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_2365832&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_236989enc_inner_2_236991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_2366102%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_236994enc_inner_1_236996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_2366372%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_236999enc_inner_0_237001*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_2366642%
#enc_inner_0/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_237004channel_2_237006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_2_layer_call_and_return_conditional_losses_2366912#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_237009channel_1_237011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_1_layer_call_and_return_conditional_losses_2367182#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_237014channel_0_237016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_0_layer_call_and_return_conditional_losses_2367452#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_239649

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_239689

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_channel_0_layer_call_and_return_conditional_losses_236745

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_239829

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_channel_0_layer_call_and_return_conditional_losses_239749

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
,__inference_enc_outer_1_layer_call_fn_239598

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_2364752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_channel_1_layer_call_fn_239778

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_1_layer_call_and_return_conditional_losses_2367182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_2_layer_call_fn_238405
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_2383142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
E__inference_channel_1_layer_call_and_return_conditional_losses_239769

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
ֆ
?
C__inference_model_4_layer_call_and_return_conditional_losses_239104

inputs.
*enc_outer_2_matmul_readvariableop_resource/
+enc_outer_2_biasadd_readvariableop_resource.
*enc_outer_1_matmul_readvariableop_resource/
+enc_outer_1_biasadd_readvariableop_resource.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_2_matmul_readvariableop_resource0
,enc_middle_2_biasadd_readvariableop_resource/
+enc_middle_1_matmul_readvariableop_resource0
,enc_middle_1_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_2_matmul_readvariableop_resource/
+enc_inner_2_biasadd_readvariableop_resource.
*enc_inner_1_matmul_readvariableop_resource/
+enc_inner_1_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_2_matmul_readvariableop_resource-
)channel_2_biasadd_readvariableop_resource,
(channel_1_matmul_readvariableop_resource-
)channel_1_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity

identity_1

identity_2?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp? channel_1/BiasAdd/ReadVariableOp?channel_1/MatMul/ReadVariableOp? channel_2/BiasAdd/ReadVariableOp?channel_2/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?"enc_inner_1/BiasAdd/ReadVariableOp?!enc_inner_1/MatMul/ReadVariableOp?"enc_inner_2/BiasAdd/ReadVariableOp?!enc_inner_2/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?#enc_middle_1/BiasAdd/ReadVariableOp?"enc_middle_1/MatMul/ReadVariableOp?#enc_middle_2/BiasAdd/ReadVariableOp?"enc_middle_2/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?"enc_outer_1/BiasAdd/ReadVariableOp?!enc_outer_1/MatMul/ReadVariableOp?"enc_outer_2/BiasAdd/ReadVariableOp?!enc_outer_2/MatMul/ReadVariableOp?
!enc_outer_2/MatMul/ReadVariableOpReadVariableOp*enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_2/MatMul/ReadVariableOp?
enc_outer_2/MatMulMatMulinputs)enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_2/MatMul?
"enc_outer_2/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_2/BiasAdd/ReadVariableOp?
enc_outer_2/BiasAddBiasAddenc_outer_2/MatMul:product:0*enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_2/BiasAdd|
enc_outer_2/ReluReluenc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_2/Relu?
!enc_outer_1/MatMul/ReadVariableOpReadVariableOp*enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_1/MatMul/ReadVariableOp?
enc_outer_1/MatMulMatMulinputs)enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_1/MatMul?
"enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_1/BiasAdd/ReadVariableOp?
enc_outer_1/BiasAddBiasAddenc_outer_1/MatMul:product:0*enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_1/BiasAdd|
enc_outer_1/ReluReluenc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_1/Relu?
!enc_outer_0/MatMul/ReadVariableOpReadVariableOp*enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_0/MatMul/ReadVariableOp?
enc_outer_0/MatMulMatMulinputs)enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_0/MatMul?
"enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_0/BiasAdd/ReadVariableOp?
enc_outer_0/BiasAddBiasAddenc_outer_0/MatMul:product:0*enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_0/BiasAdd|
enc_outer_0/ReluReluenc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_0/Relu?
"enc_middle_2/MatMul/ReadVariableOpReadVariableOp+enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_2/MatMul/ReadVariableOp?
enc_middle_2/MatMulMatMulenc_outer_2/Relu:activations:0*enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_2/MatMul?
#enc_middle_2/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_2/BiasAdd/ReadVariableOp?
enc_middle_2/BiasAddBiasAddenc_middle_2/MatMul:product:0+enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_2/BiasAdd
enc_middle_2/ReluReluenc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_2/Relu?
"enc_middle_1/MatMul/ReadVariableOpReadVariableOp+enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_1/MatMul/ReadVariableOp?
enc_middle_1/MatMulMatMulenc_outer_1/Relu:activations:0*enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_1/MatMul?
#enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_1/BiasAdd/ReadVariableOp?
enc_middle_1/BiasAddBiasAddenc_middle_1/MatMul:product:0+enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_1/BiasAdd
enc_middle_1/ReluReluenc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_1/Relu?
"enc_middle_0/MatMul/ReadVariableOpReadVariableOp+enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_0/MatMul/ReadVariableOp?
enc_middle_0/MatMulMatMulenc_outer_0/Relu:activations:0*enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_0/MatMul?
#enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_0/BiasAdd/ReadVariableOp?
enc_middle_0/BiasAddBiasAddenc_middle_0/MatMul:product:0+enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_0/BiasAdd
enc_middle_0/ReluReluenc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_0/Relu?
!enc_inner_2/MatMul/ReadVariableOpReadVariableOp*enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_2/MatMul/ReadVariableOp?
enc_inner_2/MatMulMatMulenc_middle_2/Relu:activations:0)enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_2/MatMul?
"enc_inner_2/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_2/BiasAdd/ReadVariableOp?
enc_inner_2/BiasAddBiasAddenc_inner_2/MatMul:product:0*enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_2/BiasAdd|
enc_inner_2/ReluReluenc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_2/Relu?
!enc_inner_1/MatMul/ReadVariableOpReadVariableOp*enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_1/MatMul/ReadVariableOp?
enc_inner_1/MatMulMatMulenc_middle_1/Relu:activations:0)enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_1/MatMul?
"enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_1/BiasAdd/ReadVariableOp?
enc_inner_1/BiasAddBiasAddenc_inner_1/MatMul:product:0*enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_1/BiasAdd|
enc_inner_1/ReluReluenc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_1/Relu?
!enc_inner_0/MatMul/ReadVariableOpReadVariableOp*enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_0/MatMul/ReadVariableOp?
enc_inner_0/MatMulMatMulenc_middle_0/Relu:activations:0)enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_0/MatMul?
"enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_0/BiasAdd/ReadVariableOp?
enc_inner_0/BiasAddBiasAddenc_inner_0/MatMul:product:0*enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_0/BiasAdd|
enc_inner_0/ReluReluenc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_0/Relu?
channel_2/MatMul/ReadVariableOpReadVariableOp(channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_2/MatMul/ReadVariableOp?
channel_2/MatMulMatMulenc_inner_2/Relu:activations:0'channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_2/MatMul?
 channel_2/BiasAdd/ReadVariableOpReadVariableOp)channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_2/BiasAdd/ReadVariableOp?
channel_2/BiasAddBiasAddchannel_2/MatMul:product:0(channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_2/BiasAdd?
channel_2/SoftsignSoftsignchannel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_2/Softsign?
channel_1/MatMul/ReadVariableOpReadVariableOp(channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_1/MatMul/ReadVariableOp?
channel_1/MatMulMatMulenc_inner_1/Relu:activations:0'channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_1/MatMul?
 channel_1/BiasAdd/ReadVariableOpReadVariableOp)channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_1/BiasAdd/ReadVariableOp?
channel_1/BiasAddBiasAddchannel_1/MatMul:product:0(channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_1/BiasAdd?
channel_1/SoftsignSoftsignchannel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_1/Softsign?
channel_0/MatMul/ReadVariableOpReadVariableOp(channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_0/MatMul/ReadVariableOp?
channel_0/MatMulMatMulenc_inner_0/Relu:activations:0'channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_0/MatMul?
 channel_0/BiasAdd/ReadVariableOpReadVariableOp)channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_0/BiasAdd/ReadVariableOp?
channel_0/BiasAddBiasAddchannel_0/MatMul:product:0(channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_0/BiasAdd?
channel_0/SoftsignSoftsignchannel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_0/Softsign?
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity channel_1/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity channel_2/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2D
 channel_1/BiasAdd/ReadVariableOp channel_1/BiasAdd/ReadVariableOp2B
channel_1/MatMul/ReadVariableOpchannel_1/MatMul/ReadVariableOp2D
 channel_2/BiasAdd/ReadVariableOp channel_2/BiasAdd/ReadVariableOp2B
channel_2/MatMul/ReadVariableOpchannel_2/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2H
"enc_inner_1/BiasAdd/ReadVariableOp"enc_inner_1/BiasAdd/ReadVariableOp2F
!enc_inner_1/MatMul/ReadVariableOp!enc_inner_1/MatMul/ReadVariableOp2H
"enc_inner_2/BiasAdd/ReadVariableOp"enc_inner_2/BiasAdd/ReadVariableOp2F
!enc_inner_2/MatMul/ReadVariableOp!enc_inner_2/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2J
#enc_middle_1/BiasAdd/ReadVariableOp#enc_middle_1/BiasAdd/ReadVariableOp2H
"enc_middle_1/MatMul/ReadVariableOp"enc_middle_1/MatMul/ReadVariableOp2J
#enc_middle_2/BiasAdd/ReadVariableOp#enc_middle_2/BiasAdd/ReadVariableOp2H
"enc_middle_2/MatMul/ReadVariableOp"enc_middle_2/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp2H
"enc_outer_1/BiasAdd/ReadVariableOp"enc_outer_1/BiasAdd/ReadVariableOp2F
!enc_outer_1/MatMul/ReadVariableOp!enc_outer_1/MatMul/ReadVariableOp2H
"enc_outer_2/BiasAdd/ReadVariableOp"enc_outer_2/BiasAdd/ReadVariableOp2F
!enc_outer_2/MatMul/ReadVariableOp!enc_outer_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dec_output_layer_call_and_return_conditional_losses_239989

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_236664

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_model_5_layer_call_fn_237520
decoder_input_0
decoder_input_1
decoder_input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0decoder_input_1decoder_input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2374772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_1:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_2
?

*__inference_channel_2_layer_call_fn_239798

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_channel_2_layer_call_and_return_conditional_losses_2366912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
(__inference_model_4_layer_call_fn_239308

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2370222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapesv
t:??????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_239849

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_5_layer_call_fn_239558
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2375822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
-__inference_dec_middle_0_layer_call_fn_239878

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_2372292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
,__inference_dec_inner_1_layer_call_fn_239838

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_2371212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_239869

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
,__inference_dec_outer_2_layer_call_fn_239978

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_2373102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_236610

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238314
x
model_4_238221
model_4_238223
model_4_238225
model_4_238227
model_4_238229
model_4_238231
model_4_238233
model_4_238235
model_4_238237
model_4_238239
model_4_238241
model_4_238243
model_4_238245
model_4_238247
model_4_238249
model_4_238251
model_4_238253
model_4_238255
model_4_238257
model_4_238259
model_4_238261
model_4_238263
model_4_238265
model_4_238267
model_5_238272
model_5_238274
model_5_238276
model_5_238278
model_5_238280
model_5_238282
model_5_238284
model_5_238286
model_5_238288
model_5_238290
model_5_238292
model_5_238294
model_5_238296
model_5_238298
model_5_238300
model_5_238302
model_5_238304
model_5_238306
model_5_238308
model_5_238310
identity??model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?
model_4/StatefulPartitionedCallStatefulPartitionedCallxmodel_4_238221model_4_238223model_4_238225model_4_238227model_4_238229model_4_238231model_4_238233model_4_238235model_4_238237model_4_238239model_4_238241model_4_238243model_4_238245model_4_238247model_4_238249model_4_238251model_4_238253model_4_238255model_4_238257model_4_238259model_4_238261model_4_238263model_4_238265model_4_238267*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2370222!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0(model_4/StatefulPartitionedCall:output:1(model_4/StatefulPartitionedCall:output:2model_5_238272model_5_238274model_5_238276model_5_238278model_5_238280model_5_238282model_5_238284model_5_238286model_5_238288model_5_238290model_5_238292model_5_238294model_5_238296model_5_238298model_5_238300model_5_238302model_5_238304model_5_238306model_5_238308model_5_238310*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2375822!
model_5/StatefulPartitionedCall?
IdentityIdentity(model_5/StatefulPartitionedCall:output:0 ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_236583

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_237175

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
,__inference_dec_outer_0_layer_call_fn_239938

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_2372562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_236529

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_239709

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?;
?
C__inference_model_5_layer_call_and_return_conditional_losses_237477

inputs
inputs_1
inputs_2
dec_inner_2_237424
dec_inner_2_237426
dec_inner_1_237429
dec_inner_1_237431
dec_inner_0_237434
dec_inner_0_237436
dec_middle_2_237439
dec_middle_2_237441
dec_middle_1_237444
dec_middle_1_237446
dec_middle_0_237449
dec_middle_0_237451
dec_outer_0_237454
dec_outer_0_237456
dec_outer_1_237459
dec_outer_1_237461
dec_outer_2_237464
dec_outer_2_237466
dec_output_237471
dec_output_237473
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2dec_inner_2_237424dec_inner_2_237426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_2370942%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1dec_inner_1_237429dec_inner_1_237431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_2371212%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_237434dec_inner_0_237436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_2371482%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_237439dec_middle_2_237441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_2371752&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_237444dec_middle_1_237446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_2372022&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_237449dec_middle_0_237451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_2372292&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_237454dec_outer_0_237456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_2372562%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_237459dec_outer_1_237461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_2372832%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_237464dec_outer_2_237466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_2373102%
#dec_outer_2/StatefulPartitionedCallt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_1/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dec_output_237471dec_output_237473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dec_output_layer_call_and_return_conditional_losses_2373392$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_239609

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????=
output_11
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?m
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
layer_with_weights-7
layer-8
layer_with_weights-8
layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?h
_tf_keras_network?h{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_1", "inbound_nodes": [[["enc_outer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_2", "inbound_nodes": [[["enc_outer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_1", "inbound_nodes": [[["enc_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_2", "inbound_nodes": [[["enc_middle_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_1", "inbound_nodes": [[["enc_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_2", "inbound_nodes": [[["enc_inner_2", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0], ["channel_1", 0, 0], ["channel_2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 784]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_1", "inbound_nodes": [[["enc_outer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_2", "inbound_nodes": [[["enc_outer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_1", "inbound_nodes": [[["enc_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_2", "inbound_nodes": [[["enc_middle_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_1", "inbound_nodes": [[["enc_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_2", "inbound_nodes": [[["enc_inner_2", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0], ["channel_1", 0, 0], ["channel_2", 0, 0]]}}}
?m
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
 layer_with_weights-3
 layer-6
!layer_with_weights-4
!layer-7
"layer_with_weights-5
"layer-8
#layer_with_weights-6
#layer-9
$layer_with_weights-7
$layer-10
%layer_with_weights-8
%layer-11
&layer-12
'layer_with_weights-9
'layer-13
(	variables
)trainable_variables
*regularization_losses
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?h
_tf_keras_network?h{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}, "name": "decoder_input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_2"}, "name": "decoder_input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_1", "inbound_nodes": [[["decoder_input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_2", "inbound_nodes": [[["decoder_input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_1", "inbound_nodes": [[["dec_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_2", "inbound_nodes": [[["dec_inner_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_1", "inbound_nodes": [[["dec_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_2", "inbound_nodes": [[["dec_middle_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_1", "inbound_nodes": [[["dec_outer_0", 0, 0, {"axis": 1}], ["dec_outer_1", 0, 0, {"axis": 1}], ["dec_outer_2", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.concat_1", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0], ["decoder_input_1", 0, 0], ["decoder_input_2", 0, 0]], "output_layers": [["dec_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}, "name": "decoder_input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_2"}, "name": "decoder_input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_1", "inbound_nodes": [[["decoder_input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_2", "inbound_nodes": [[["decoder_input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_1", "inbound_nodes": [[["dec_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_2", "inbound_nodes": [[["dec_inner_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_1", "inbound_nodes": [[["dec_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_2", "inbound_nodes": [[["dec_middle_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_1", "inbound_nodes": [[["dec_outer_0", 0, 0, {"axis": 1}], ["dec_outer_1", 0, 0, {"axis": 1}], ["dec_outer_2", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.concat_1", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0], ["decoder_input_1", 0, 0], ["decoder_input_2", 0, 0]], "output_layers": [["dec_output", 0, 0]]}}}
?
,iter

-beta_1

.beta_2
	/decay
0learning_rate1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?"
	optimizer
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
I24
J25
K26
L27
M28
N29
O30
P31
Q32
R33
S34
T35
U36
V37
W38
X39
Y40
Z41
[42
\43"
trackable_list_wrapper
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
I24
J25
K26
L27
M28
N29
O30
P31
Q32
R33
S34
T35
U36
V37
W38
X39
Y40
Z41
[42
\43"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
]metrics
^non_trainable_variables
_layer_regularization_losses
trainable_variables
`layer_metrics
regularization_losses

alayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
?

1kernel
2bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

3kernel
4bias
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

5kernel
6bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

7kernel
8bias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

9kernel
:bias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

;kernel
<bias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

=kernel
>bias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

?kernel
@bias
~trainable_variables
	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

Ckernel
Dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Gkernel
Hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23"
trackable_list_wrapper
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
trainable_variables
?layer_metrics
regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_2"}}
?

Ikernel
Jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

Kkernel
Lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

Mkernel
Nbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Qkernel
Rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Skernel
Tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Ukernel
Vbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Wkernel
Xbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Ykernel
Zbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}}
?

[kernel
\bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 180}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180]}}
?
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19"
trackable_list_wrapper
?
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
)trainable_variables
?layer_metrics
*regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
%:#	?<2enc_outer_0/kernel
:<2enc_outer_0/bias
%:#	?<2enc_outer_1/kernel
:<2enc_outer_1/bias
%:#	?<2enc_outer_2/kernel
:<2enc_outer_2/bias
%:#<22enc_middle_0/kernel
:22enc_middle_0/bias
%:#<22enc_middle_1/kernel
:22enc_middle_1/bias
%:#<22enc_middle_2/kernel
:22enc_middle_2/bias
$:"2(2enc_inner_0/kernel
:(2enc_inner_0/bias
$:"2(2enc_inner_1/kernel
:(2enc_inner_1/bias
$:"2(2enc_inner_2/kernel
:(2enc_inner_2/bias
": (2channel_0/kernel
:2channel_0/bias
": (2channel_1/kernel
:2channel_1/bias
": (2channel_2/kernel
:2channel_2/bias
$:"(2dec_inner_0/kernel
:(2dec_inner_0/bias
$:"(2dec_inner_1/kernel
:(2dec_inner_1/bias
$:"(2dec_inner_2/kernel
:(2dec_inner_2/bias
%:#(<2dec_middle_0/kernel
:<2dec_middle_0/bias
%:#(<2dec_middle_1/kernel
:<2dec_middle_1/bias
%:#(<2dec_middle_2/kernel
:<2dec_middle_2/bias
$:"<<2dec_outer_0/kernel
:<2dec_outer_0/bias
$:"<<2dec_outer_1/kernel
:<2dec_outer_1/bias
$:"<<2dec_outer_2/kernel
:<2dec_outer_2/bias
%:#
??2dec_output/kernel
:?2dec_output/bias
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
btrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
c	variables
?layer_metrics
dregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ftrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
g	variables
?layer_metrics
hregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
k	variables
?layer_metrics
lregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ntrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
o	variables
?layer_metrics
pregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
s	variables
?layer_metrics
tregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
w	variables
?layer_metrics
xregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ztrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
{	variables
?layer_metrics
|regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(	?<2Adam/enc_outer_0/kernel/m
#:!<2Adam/enc_outer_0/bias/m
*:(	?<2Adam/enc_outer_1/kernel/m
#:!<2Adam/enc_outer_1/bias/m
*:(	?<2Adam/enc_outer_2/kernel/m
#:!<2Adam/enc_outer_2/bias/m
*:(<22Adam/enc_middle_0/kernel/m
$:"22Adam/enc_middle_0/bias/m
*:(<22Adam/enc_middle_1/kernel/m
$:"22Adam/enc_middle_1/bias/m
*:(<22Adam/enc_middle_2/kernel/m
$:"22Adam/enc_middle_2/bias/m
):'2(2Adam/enc_inner_0/kernel/m
#:!(2Adam/enc_inner_0/bias/m
):'2(2Adam/enc_inner_1/kernel/m
#:!(2Adam/enc_inner_1/bias/m
):'2(2Adam/enc_inner_2/kernel/m
#:!(2Adam/enc_inner_2/bias/m
':%(2Adam/channel_0/kernel/m
!:2Adam/channel_0/bias/m
':%(2Adam/channel_1/kernel/m
!:2Adam/channel_1/bias/m
':%(2Adam/channel_2/kernel/m
!:2Adam/channel_2/bias/m
):'(2Adam/dec_inner_0/kernel/m
#:!(2Adam/dec_inner_0/bias/m
):'(2Adam/dec_inner_1/kernel/m
#:!(2Adam/dec_inner_1/bias/m
):'(2Adam/dec_inner_2/kernel/m
#:!(2Adam/dec_inner_2/bias/m
*:((<2Adam/dec_middle_0/kernel/m
$:"<2Adam/dec_middle_0/bias/m
*:((<2Adam/dec_middle_1/kernel/m
$:"<2Adam/dec_middle_1/bias/m
*:((<2Adam/dec_middle_2/kernel/m
$:"<2Adam/dec_middle_2/bias/m
):'<<2Adam/dec_outer_0/kernel/m
#:!<2Adam/dec_outer_0/bias/m
):'<<2Adam/dec_outer_1/kernel/m
#:!<2Adam/dec_outer_1/bias/m
):'<<2Adam/dec_outer_2/kernel/m
#:!<2Adam/dec_outer_2/bias/m
*:(
??2Adam/dec_output/kernel/m
#:!?2Adam/dec_output/bias/m
*:(	?<2Adam/enc_outer_0/kernel/v
#:!<2Adam/enc_outer_0/bias/v
*:(	?<2Adam/enc_outer_1/kernel/v
#:!<2Adam/enc_outer_1/bias/v
*:(	?<2Adam/enc_outer_2/kernel/v
#:!<2Adam/enc_outer_2/bias/v
*:(<22Adam/enc_middle_0/kernel/v
$:"22Adam/enc_middle_0/bias/v
*:(<22Adam/enc_middle_1/kernel/v
$:"22Adam/enc_middle_1/bias/v
*:(<22Adam/enc_middle_2/kernel/v
$:"22Adam/enc_middle_2/bias/v
):'2(2Adam/enc_inner_0/kernel/v
#:!(2Adam/enc_inner_0/bias/v
):'2(2Adam/enc_inner_1/kernel/v
#:!(2Adam/enc_inner_1/bias/v
):'2(2Adam/enc_inner_2/kernel/v
#:!(2Adam/enc_inner_2/bias/v
':%(2Adam/channel_0/kernel/v
!:2Adam/channel_0/bias/v
':%(2Adam/channel_1/kernel/v
!:2Adam/channel_1/bias/v
':%(2Adam/channel_2/kernel/v
!:2Adam/channel_2/bias/v
):'(2Adam/dec_inner_0/kernel/v
#:!(2Adam/dec_inner_0/bias/v
):'(2Adam/dec_inner_1/kernel/v
#:!(2Adam/dec_inner_1/bias/v
):'(2Adam/dec_inner_2/kernel/v
#:!(2Adam/dec_inner_2/bias/v
*:((<2Adam/dec_middle_0/kernel/v
$:"<2Adam/dec_middle_0/bias/v
*:((<2Adam/dec_middle_1/kernel/v
$:"<2Adam/dec_middle_1/bias/v
*:((<2Adam/dec_middle_2/kernel/v
$:"<2Adam/dec_middle_2/bias/v
):'<<2Adam/dec_outer_0/kernel/v
#:!<2Adam/dec_outer_0/bias/v
):'<<2Adam/dec_outer_1/kernel/v
#:!<2Adam/dec_outer_1/bias/v
):'<<2Adam/dec_outer_2/kernel/v
#:!<2Adam/dec_outer_2/bias/v
*:(
??2Adam/dec_output/kernel/v
#:!?2Adam/dec_output/bias/v
?2?
!__inference__wrapped_model_236433?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238828
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_237930
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238026
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238668?
???
FullArgSpec
args?
jself
jx
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
.__inference_autoencoder_2_layer_call_fn_238405
.__inference_autoencoder_2_layer_call_fn_238216
.__inference_autoencoder_2_layer_call_fn_238921
.__inference_autoencoder_2_layer_call_fn_239014?
???
FullArgSpec
args?
jself
jx
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
C__inference_model_4_layer_call_and_return_conditional_losses_239194
C__inference_model_4_layer_call_and_return_conditional_losses_236830
C__inference_model_4_layer_call_and_return_conditional_losses_236764
C__inference_model_4_layer_call_and_return_conditional_losses_239104?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_4_layer_call_fn_236954
(__inference_model_4_layer_call_fn_239308
(__inference_model_4_layer_call_fn_239251
(__inference_model_4_layer_call_fn_237077?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_5_layer_call_and_return_conditional_losses_239464
C__inference_model_5_layer_call_and_return_conditional_losses_237356
C__inference_model_5_layer_call_and_return_conditional_losses_237414
C__inference_model_5_layer_call_and_return_conditional_losses_239386?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_5_layer_call_fn_237520
(__inference_model_5_layer_call_fn_239511
(__inference_model_5_layer_call_fn_239558
(__inference_model_5_layer_call_fn_237625?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_238508input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_239569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_enc_outer_0_layer_call_fn_239578?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_239589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_enc_outer_1_layer_call_fn_239598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_239609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_enc_outer_2_layer_call_fn_239618?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_239629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_enc_middle_0_layer_call_fn_239638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_239649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_enc_middle_1_layer_call_fn_239658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_239669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_enc_middle_2_layer_call_fn_239678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_239689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_enc_inner_0_layer_call_fn_239698?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_239709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_enc_inner_1_layer_call_fn_239718?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_239729?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_enc_inner_2_layer_call_fn_239738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_channel_0_layer_call_and_return_conditional_losses_239749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_channel_0_layer_call_fn_239758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_channel_1_layer_call_and_return_conditional_losses_239769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_channel_1_layer_call_fn_239778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_channel_2_layer_call_and_return_conditional_losses_239789?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_channel_2_layer_call_fn_239798?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_239809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dec_inner_0_layer_call_fn_239818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_239829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dec_inner_1_layer_call_fn_239838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_239849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dec_inner_2_layer_call_fn_239858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_239869?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dec_middle_0_layer_call_fn_239878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_239889?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dec_middle_1_layer_call_fn_239898?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_239909?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dec_middle_2_layer_call_fn_239918?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_239929?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dec_outer_0_layer_call_fn_239938?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_239949?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dec_outer_1_layer_call_fn_239958?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_239969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dec_outer_2_layer_call_fn_239978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dec_output_layer_call_and_return_conditional_losses_239989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dec_output_layer_call_fn_239998?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_236433?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\1?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_237930?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\A?>
'?$
"?
input_1??????????
?

trainingp"&?#
?
0??????????
? ?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238026?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\A?>
'?$
"?
input_1??????????
?

trainingp "&?#
?
0??????????
? ?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238668?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\;?8
!?
?
x??????????
?

trainingp"&?#
?
0??????????
? ?
I__inference_autoencoder_2_layer_call_and_return_conditional_losses_238828?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\;?8
!?
?
x??????????
?

trainingp "&?#
?
0??????????
? ?
.__inference_autoencoder_2_layer_call_fn_238216?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\A?>
'?$
"?
input_1??????????
?

trainingp"????????????
.__inference_autoencoder_2_layer_call_fn_238405?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\A?>
'?$
"?
input_1??????????
?

trainingp "????????????
.__inference_autoencoder_2_layer_call_fn_238921?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\;?8
!?
?
x??????????
?

trainingp"????????????
.__inference_autoencoder_2_layer_call_fn_239014?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\;?8
!?
?
x??????????
?

trainingp "????????????
E__inference_channel_0_layer_call_and_return_conditional_losses_239749\CD/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_0_layer_call_fn_239758OCD/?,
%?"
 ?
inputs?????????(
? "???????????
E__inference_channel_1_layer_call_and_return_conditional_losses_239769\EF/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_1_layer_call_fn_239778OEF/?,
%?"
 ?
inputs?????????(
? "???????????
E__inference_channel_2_layer_call_and_return_conditional_losses_239789\GH/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_2_layer_call_fn_239798OGH/?,
%?"
 ?
inputs?????????(
? "???????????
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_239809\IJ/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_0_layer_call_fn_239818OIJ/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_239829\KL/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_1_layer_call_fn_239838OKL/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_239849\MN/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_2_layer_call_fn_239858OMN/?,
%?"
 ?
inputs?????????
? "??????????(?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_239869\OP/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_0_layer_call_fn_239878OOP/?,
%?"
 ?
inputs?????????(
? "??????????<?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_239889\QR/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_1_layer_call_fn_239898OQR/?,
%?"
 ?
inputs?????????(
? "??????????<?
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_239909\ST/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_2_layer_call_fn_239918OST/?,
%?"
 ?
inputs?????????(
? "??????????<?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_239929\UV/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_0_layer_call_fn_239938OUV/?,
%?"
 ?
inputs?????????<
? "??????????<?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_239949\WX/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_1_layer_call_fn_239958OWX/?,
%?"
 ?
inputs?????????<
? "??????????<?
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_239969\YZ/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_2_layer_call_fn_239978OYZ/?,
%?"
 ?
inputs?????????<
? "??????????<?
F__inference_dec_output_layer_call_and_return_conditional_losses_239989^[\0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dec_output_layer_call_fn_239998Q[\0?-
&?#
!?
inputs??????????
? "????????????
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_239689\=>/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_0_layer_call_fn_239698O=>/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_239709\?@/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_1_layer_call_fn_239718O?@/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_239729\AB/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_2_layer_call_fn_239738OAB/?,
%?"
 ?
inputs?????????2
? "??????????(?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_239629\78/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_0_layer_call_fn_239638O78/?,
%?"
 ?
inputs?????????<
? "??????????2?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_239649\9:/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_1_layer_call_fn_239658O9:/?,
%?"
 ?
inputs?????????<
? "??????????2?
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_239669\;</?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_2_layer_call_fn_239678O;</?,
%?"
 ?
inputs?????????<
? "??????????2?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_239569]120?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_0_layer_call_fn_239578P120?-
&?#
!?
inputs??????????
? "??????????<?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_239589]340?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_1_layer_call_fn_239598P340?-
&?#
!?
inputs??????????
? "??????????<?
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_239609]560?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_2_layer_call_fn_239618P560?-
&?#
!?
inputs??????????
? "??????????<?
C__inference_model_4_layer_call_and_return_conditional_losses_236764?563412;<9:78AB?@=>GHEFCD??<
5?2
(?%
encoder_input??????????
p

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_236830?563412;<9:78AB?@=>GHEFCD??<
5?2
(?%
encoder_input??????????
p 

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_239104?563412;<9:78AB?@=>GHEFCD8?5
.?+
!?
inputs??????????
p

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_239194?563412;<9:78AB?@=>GHEFCD8?5
.?+
!?
inputs??????????
p 

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
(__inference_model_4_layer_call_fn_236954?563412;<9:78AB?@=>GHEFCD??<
5?2
(?%
encoder_input??????????
p

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
(__inference_model_4_layer_call_fn_237077?563412;<9:78AB?@=>GHEFCD??<
5?2
(?%
encoder_input??????????
p 

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
(__inference_model_4_layer_call_fn_239251?563412;<9:78AB?@=>GHEFCD8?5
.?+
!?
inputs??????????
p

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
(__inference_model_4_layer_call_fn_239308?563412;<9:78AB?@=>GHEFCD8?5
.?+
!?
inputs??????????
p 

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
C__inference_model_5_layer_call_and_return_conditional_losses_237356?MNKLIJSTQROPUVWXYZ[\???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_237414?MNKLIJSTQROPUVWXYZ[\???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_239386?MNKLIJSTQROPUVWXYZ[\???
|?y
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_239464?MNKLIJSTQROPUVWXYZ[\???
|?y
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
p 

 
? "&?#
?
0??????????
? ?
(__inference_model_5_layer_call_fn_237520?MNKLIJSTQROPUVWXYZ[\???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
p

 
? "????????????
(__inference_model_5_layer_call_fn_237625?MNKLIJSTQROPUVWXYZ[\???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
p 

 
? "????????????
(__inference_model_5_layer_call_fn_239511?MNKLIJSTQROPUVWXYZ[\???
|?y
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
p

 
? "????????????
(__inference_model_5_layer_call_fn_239558?MNKLIJSTQROPUVWXYZ[\???
|?y
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
p 

 
? "????????????
$__inference_signature_wrapper_238508?,563412;<9:78AB?@=>GHEFCDMNKLIJSTQROPUVWXYZ[\<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????