??2
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
 ?"serve*2.4.12unknown8ü*
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
enc_outer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<*#
shared_nameenc_outer_3/kernel
z
&enc_outer_3/kernel/Read/ReadVariableOpReadVariableOpenc_outer_3/kernel*
_output_shapes
:	?<*
dtype0
x
enc_outer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_nameenc_outer_3/bias
q
$enc_outer_3/bias/Read/ReadVariableOpReadVariableOpenc_outer_3/bias*
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
enc_middle_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*$
shared_nameenc_middle_3/kernel
{
'enc_middle_3/kernel/Read/ReadVariableOpReadVariableOpenc_middle_3/kernel*
_output_shapes

:<2*
dtype0
z
enc_middle_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameenc_middle_3/bias
s
%enc_middle_3/bias/Read/ReadVariableOpReadVariableOpenc_middle_3/bias*
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
?
enc_inner_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*#
shared_nameenc_inner_3/kernel
y
&enc_inner_3/kernel/Read/ReadVariableOpReadVariableOpenc_inner_3/kernel*
_output_shapes

:2(*
dtype0
x
enc_inner_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameenc_inner_3/bias
q
$enc_inner_3/bias/Read/ReadVariableOpReadVariableOpenc_inner_3/bias*
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
|
channel_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namechannel_3/kernel
u
$channel_3/kernel/Read/ReadVariableOpReadVariableOpchannel_3/kernel*
_output_shapes

:(*
dtype0
t
channel_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namechannel_3/bias
m
"channel_3/bias/Read/ReadVariableOpReadVariableOpchannel_3/bias*
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
dec_inner_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedec_inner_3/kernel
y
&dec_inner_3/kernel/Read/ReadVariableOpReadVariableOpdec_inner_3/kernel*
_output_shapes

:(*
dtype0
x
dec_inner_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedec_inner_3/bias
q
$dec_inner_3/bias/Read/ReadVariableOpReadVariableOpdec_inner_3/bias*
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
dec_middle_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*$
shared_namedec_middle_3/kernel
{
'dec_middle_3/kernel/Read/ReadVariableOpReadVariableOpdec_middle_3/kernel*
_output_shapes

:(<*
dtype0
z
dec_middle_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*"
shared_namedec_middle_3/bias
s
%dec_middle_3/bias/Read/ReadVariableOpReadVariableOpdec_middle_3/bias*
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
dec_outer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*#
shared_namedec_outer_3/kernel
y
&dec_outer_3/kernel/Read/ReadVariableOpReadVariableOpdec_outer_3/kernel*
_output_shapes

:<<*
dtype0
x
dec_outer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_namedec_outer_3/bias
q
$dec_outer_3/bias/Read/ReadVariableOpReadVariableOpdec_outer_3/bias*
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
Adam/enc_outer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_3/kernel/m
?
-Adam/enc_outer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_3/kernel/m*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_3/bias/m

+Adam/enc_outer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_outer_3/bias/m*
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
Adam/enc_middle_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_3/kernel/m
?
.Adam/enc_middle_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_3/kernel/m*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_3/bias/m
?
,Adam/enc_middle_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_middle_3/bias/m*
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
Adam/enc_inner_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_3/kernel/m
?
-Adam/enc_inner_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_3/kernel/m*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_3/bias/m

+Adam/enc_inner_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_inner_3/bias/m*
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
Adam/channel_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_3/kernel/m
?
+Adam/channel_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/channel_3/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/channel_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_3/bias/m
{
)Adam/channel_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/channel_3/bias/m*
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
Adam/dec_inner_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_3/kernel/m
?
-Adam/dec_inner_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_3/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_3/bias/m

+Adam/dec_inner_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_inner_3/bias/m*
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
Adam/dec_middle_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_3/kernel/m
?
.Adam/dec_middle_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_3/kernel/m*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_3/bias/m
?
,Adam/dec_middle_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_middle_3/bias/m*
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
Adam/dec_outer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_3/kernel/m
?
-Adam/dec_outer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_3/kernel/m*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_3/bias/m

+Adam/dec_outer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_outer_3/bias/m*
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
Adam/enc_outer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?<**
shared_nameAdam/enc_outer_3/kernel/v
?
-Adam/enc_outer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_3/kernel/v*
_output_shapes
:	?<*
dtype0
?
Adam/enc_outer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/enc_outer_3/bias/v

+Adam/enc_outer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_outer_3/bias/v*
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
Adam/enc_middle_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*+
shared_nameAdam/enc_middle_3/kernel/v
?
.Adam/enc_middle_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_3/kernel/v*
_output_shapes

:<2*
dtype0
?
Adam/enc_middle_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/enc_middle_3/bias/v
?
,Adam/enc_middle_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_middle_3/bias/v*
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
Adam/enc_inner_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(**
shared_nameAdam/enc_inner_3/kernel/v
?
-Adam/enc_inner_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_3/kernel/v*
_output_shapes

:2(*
dtype0
?
Adam/enc_inner_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/enc_inner_3/bias/v

+Adam/enc_inner_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_inner_3/bias/v*
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
Adam/channel_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/channel_3/kernel/v
?
+Adam/channel_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/channel_3/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/channel_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/channel_3/bias/v
{
)Adam/channel_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/channel_3/bias/v*
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
Adam/dec_inner_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_nameAdam/dec_inner_3/kernel/v
?
-Adam/dec_inner_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_3/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/dec_inner_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/dec_inner_3/bias/v

+Adam/dec_inner_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_inner_3/bias/v*
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
Adam/dec_middle_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*+
shared_nameAdam/dec_middle_3/kernel/v
?
.Adam/dec_middle_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_3/kernel/v*
_output_shapes

:(<*
dtype0
?
Adam/dec_middle_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*)
shared_nameAdam/dec_middle_3/bias/v
?
,Adam/dec_middle_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_middle_3/bias/v*
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
Adam/dec_outer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<**
shared_nameAdam/dec_outer_3/kernel/v
?
-Adam/dec_outer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_3/kernel/v*
_output_shapes

:<<*
dtype0
?
Adam/dec_outer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dec_outer_3/bias/v

+Adam/dec_outer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_outer_3/bias/v*
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
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*܈
valueшB͈ Bň
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
?
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
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
?
layer-0
layer-1
 layer-2
!layer-3
"layer_with_weights-0
"layer-4
#layer_with_weights-1
#layer-5
$layer_with_weights-2
$layer-6
%layer_with_weights-3
%layer-7
&layer_with_weights-4
&layer-8
'layer_with_weights-5
'layer-9
(layer_with_weights-6
(layer-10
)layer_with_weights-7
)layer-11
*layer_with_weights-8
*layer-12
+layer_with_weights-9
+layer-13
,layer_with_weights-10
,layer-14
-layer_with_weights-11
-layer-15
.layer-16
/layer_with_weights-12
/layer-17
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?	
4iter

5beta_1

6beta_2
	7decay
8learning_rate9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
[34
\35
]36
^37
_38
`39
a40
b41
c42
d43
e44
f45
g46
h47
i48
j49
k50
l51
m52
n53
o54
p55
q56
r57
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
[34
\35
]36
^37
_38
`39
a40
b41
c42
d43
e44
f45
g46
h47
i48
j49
k50
l51
m52
n53
o54
p55
q56
r57
 
?
	variables
smetrics
tnon_trainable_variables
ulayer_regularization_losses
trainable_variables
vlayer_metrics
regularization_losses

wlayers
 
 
h

9kernel
:bias
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
h

;kernel
<bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
l

=kernel
>bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

?kernel
@bias
?trainable_variables
?	variables
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
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
 
?
	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
trainable_variables
?layer_metrics
regularization_losses
?layers
 
 
 
 
l

Ykernel
Zbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

[kernel
\bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

]kernel
^bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

_kernel
`bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

akernel
bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

ckernel
dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

ekernel
fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

gkernel
hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

ikernel
jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

kkernel
lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

mkernel
nbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

okernel
pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api
l

qkernel
rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25
 
?
0	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
1trainable_variables
?layer_metrics
2regularization_losses
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
NL
VARIABLE_VALUEenc_outer_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEenc_outer_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_middle_0/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_middle_0/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEenc_middle_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_middle_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEenc_middle_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_middle_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEenc_middle_3/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_middle_3/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_0/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_0/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_1/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_1/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_2/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_2/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_3/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_3/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_0/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_0/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_1/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_1/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_3/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_3/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_0/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_0/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_1/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_1/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_2/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_2/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_3/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_3/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_0/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_0/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_1/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_1/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_2/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_2/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_3/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_3/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_0/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_0/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_1/kernel'variables/50/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_1/bias'variables/51/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_2/kernel'variables/52/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_2/bias'variables/53/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_3/kernel'variables/54/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_3/bias'variables/55/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_output/kernel'variables/56/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdec_output/bias'variables/57/.ATTRIBUTES/VARIABLE_VALUE

?0
 
 
 

0
1

90
:1

90
:1
 
?
xtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
y	variables
?layer_metrics
zregularization_losses
?layers

;0
<1

;0
<1
 
?
|trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
}	variables
?layer_metrics
~regularization_losses
?layers

=0
>1

=0
>1
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
?0
@1

?0
@1
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
A0
B1

A0
B1
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
C0
D1

C0
D1
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
E0
F1

E0
F1
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
G0
H1

G0
H1
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
 
 
 
 
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
12
13
14
15
16
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

]0
^1

]0
^1
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
_0
`1

_0
`1
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
a0
b1

a0
b1
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
c0
d1

c0
d1
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
e0
f1

e0
f1
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
g0
h1

g0
h1
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
i0
j1

i0
j1
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
k0
l1

k0
l1
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
m0
n1

m0
n1
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
o0
p1

o0
p1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
 

q0
r1

q0
r1
 
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
 
 
 
 
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
8

?total

?count
?	variables
?	keras_api
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
?0
?1

?	variables
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
qo
VARIABLE_VALUEAdam/enc_outer_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_1/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_1/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_2/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_2/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_3/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_3/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_0/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_0/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_1/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_1/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_2/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_2/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_3/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_3/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_0/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_0/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_1/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_1/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_2/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_2/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_3/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_3/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_0/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_0/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_1/kernel/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_1/bias/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_2/kernel/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_2/bias/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_3/kernel/mCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_3/bias/mCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/mCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/mCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_1/kernel/mCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_1/bias/mCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_2/kernel/mCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_2/bias/mCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_3/kernel/mCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_3/bias/mCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/mCvariables/48/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/mCvariables/49/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_1/kernel/mCvariables/50/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_1/bias/mCvariables/51/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_2/kernel/mCvariables/52/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_2/bias/mCvariables/53/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_3/kernel/mCvariables/54/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_3/bias/mCvariables/55/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/mCvariables/56/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/mCvariables/57/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
qo
VARIABLE_VALUEAdam/enc_outer_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_1/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_1/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_2/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_2/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/enc_middle_3/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_middle_3/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_0/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_0/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_1/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_1/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_2/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_2/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_3/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_3/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_0/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_0/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_1/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_1/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_2/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_2/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_3/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_3/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_0/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_0/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_1/kernel/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_1/bias/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_2/kernel/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_2/bias/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_3/kernel/vCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_3/bias/vCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/vCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/vCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_1/kernel/vCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_1/bias/vCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_2/kernel/vCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_2/bias/vCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_3/kernel/vCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_3/bias/vCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/vCvariables/48/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/vCvariables/49/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_1/kernel/vCvariables/50/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_1/bias/vCvariables/51/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_2/kernel/vCvariables/52/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_2/bias/vCvariables/53/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_3/kernel/vCvariables/54/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_3/bias/vCvariables/55/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/vCvariables/56/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/vCvariables/57/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1enc_outer_3/kernelenc_outer_3/biasenc_outer_2/kernelenc_outer_2/biasenc_outer_1/kernelenc_outer_1/biasenc_outer_0/kernelenc_outer_0/biasenc_middle_3/kernelenc_middle_3/biasenc_middle_2/kernelenc_middle_2/biasenc_middle_1/kernelenc_middle_1/biasenc_middle_0/kernelenc_middle_0/biasenc_inner_3/kernelenc_inner_3/biasenc_inner_2/kernelenc_inner_2/biasenc_inner_1/kernelenc_inner_1/biasenc_inner_0/kernelenc_inner_0/biaschannel_3/kernelchannel_3/biaschannel_2/kernelchannel_2/biaschannel_1/kernelchannel_1/biaschannel_0/kernelchannel_0/biasdec_inner_3/kerneldec_inner_3/biasdec_inner_2/kerneldec_inner_2/biasdec_inner_1/kerneldec_inner_1/biasdec_inner_0/kerneldec_inner_0/biasdec_middle_3/kerneldec_middle_3/biasdec_middle_2/kerneldec_middle_2/biasdec_middle_1/kerneldec_middle_1/biasdec_middle_0/kerneldec_middle_0/biasdec_outer_0/kerneldec_outer_0/biasdec_outer_1/kerneldec_outer_1/biasdec_outer_2/kerneldec_outer_2/biasdec_outer_3/kerneldec_outer_3/biasdec_output/kerneldec_output/bias*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_307286
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?@
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp&enc_outer_0/kernel/Read/ReadVariableOp$enc_outer_0/bias/Read/ReadVariableOp&enc_outer_1/kernel/Read/ReadVariableOp$enc_outer_1/bias/Read/ReadVariableOp&enc_outer_2/kernel/Read/ReadVariableOp$enc_outer_2/bias/Read/ReadVariableOp&enc_outer_3/kernel/Read/ReadVariableOp$enc_outer_3/bias/Read/ReadVariableOp'enc_middle_0/kernel/Read/ReadVariableOp%enc_middle_0/bias/Read/ReadVariableOp'enc_middle_1/kernel/Read/ReadVariableOp%enc_middle_1/bias/Read/ReadVariableOp'enc_middle_2/kernel/Read/ReadVariableOp%enc_middle_2/bias/Read/ReadVariableOp'enc_middle_3/kernel/Read/ReadVariableOp%enc_middle_3/bias/Read/ReadVariableOp&enc_inner_0/kernel/Read/ReadVariableOp$enc_inner_0/bias/Read/ReadVariableOp&enc_inner_1/kernel/Read/ReadVariableOp$enc_inner_1/bias/Read/ReadVariableOp&enc_inner_2/kernel/Read/ReadVariableOp$enc_inner_2/bias/Read/ReadVariableOp&enc_inner_3/kernel/Read/ReadVariableOp$enc_inner_3/bias/Read/ReadVariableOp$channel_0/kernel/Read/ReadVariableOp"channel_0/bias/Read/ReadVariableOp$channel_1/kernel/Read/ReadVariableOp"channel_1/bias/Read/ReadVariableOp$channel_2/kernel/Read/ReadVariableOp"channel_2/bias/Read/ReadVariableOp$channel_3/kernel/Read/ReadVariableOp"channel_3/bias/Read/ReadVariableOp&dec_inner_0/kernel/Read/ReadVariableOp$dec_inner_0/bias/Read/ReadVariableOp&dec_inner_1/kernel/Read/ReadVariableOp$dec_inner_1/bias/Read/ReadVariableOp&dec_inner_2/kernel/Read/ReadVariableOp$dec_inner_2/bias/Read/ReadVariableOp&dec_inner_3/kernel/Read/ReadVariableOp$dec_inner_3/bias/Read/ReadVariableOp'dec_middle_0/kernel/Read/ReadVariableOp%dec_middle_0/bias/Read/ReadVariableOp'dec_middle_1/kernel/Read/ReadVariableOp%dec_middle_1/bias/Read/ReadVariableOp'dec_middle_2/kernel/Read/ReadVariableOp%dec_middle_2/bias/Read/ReadVariableOp'dec_middle_3/kernel/Read/ReadVariableOp%dec_middle_3/bias/Read/ReadVariableOp&dec_outer_0/kernel/Read/ReadVariableOp$dec_outer_0/bias/Read/ReadVariableOp&dec_outer_1/kernel/Read/ReadVariableOp$dec_outer_1/bias/Read/ReadVariableOp&dec_outer_2/kernel/Read/ReadVariableOp$dec_outer_2/bias/Read/ReadVariableOp&dec_outer_3/kernel/Read/ReadVariableOp$dec_outer_3/bias/Read/ReadVariableOp%dec_output/kernel/Read/ReadVariableOp#dec_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/enc_outer_0/kernel/m/Read/ReadVariableOp+Adam/enc_outer_0/bias/m/Read/ReadVariableOp-Adam/enc_outer_1/kernel/m/Read/ReadVariableOp+Adam/enc_outer_1/bias/m/Read/ReadVariableOp-Adam/enc_outer_2/kernel/m/Read/ReadVariableOp+Adam/enc_outer_2/bias/m/Read/ReadVariableOp-Adam/enc_outer_3/kernel/m/Read/ReadVariableOp+Adam/enc_outer_3/bias/m/Read/ReadVariableOp.Adam/enc_middle_0/kernel/m/Read/ReadVariableOp,Adam/enc_middle_0/bias/m/Read/ReadVariableOp.Adam/enc_middle_1/kernel/m/Read/ReadVariableOp,Adam/enc_middle_1/bias/m/Read/ReadVariableOp.Adam/enc_middle_2/kernel/m/Read/ReadVariableOp,Adam/enc_middle_2/bias/m/Read/ReadVariableOp.Adam/enc_middle_3/kernel/m/Read/ReadVariableOp,Adam/enc_middle_3/bias/m/Read/ReadVariableOp-Adam/enc_inner_0/kernel/m/Read/ReadVariableOp+Adam/enc_inner_0/bias/m/Read/ReadVariableOp-Adam/enc_inner_1/kernel/m/Read/ReadVariableOp+Adam/enc_inner_1/bias/m/Read/ReadVariableOp-Adam/enc_inner_2/kernel/m/Read/ReadVariableOp+Adam/enc_inner_2/bias/m/Read/ReadVariableOp-Adam/enc_inner_3/kernel/m/Read/ReadVariableOp+Adam/enc_inner_3/bias/m/Read/ReadVariableOp+Adam/channel_0/kernel/m/Read/ReadVariableOp)Adam/channel_0/bias/m/Read/ReadVariableOp+Adam/channel_1/kernel/m/Read/ReadVariableOp)Adam/channel_1/bias/m/Read/ReadVariableOp+Adam/channel_2/kernel/m/Read/ReadVariableOp)Adam/channel_2/bias/m/Read/ReadVariableOp+Adam/channel_3/kernel/m/Read/ReadVariableOp)Adam/channel_3/bias/m/Read/ReadVariableOp-Adam/dec_inner_0/kernel/m/Read/ReadVariableOp+Adam/dec_inner_0/bias/m/Read/ReadVariableOp-Adam/dec_inner_1/kernel/m/Read/ReadVariableOp+Adam/dec_inner_1/bias/m/Read/ReadVariableOp-Adam/dec_inner_2/kernel/m/Read/ReadVariableOp+Adam/dec_inner_2/bias/m/Read/ReadVariableOp-Adam/dec_inner_3/kernel/m/Read/ReadVariableOp+Adam/dec_inner_3/bias/m/Read/ReadVariableOp.Adam/dec_middle_0/kernel/m/Read/ReadVariableOp,Adam/dec_middle_0/bias/m/Read/ReadVariableOp.Adam/dec_middle_1/kernel/m/Read/ReadVariableOp,Adam/dec_middle_1/bias/m/Read/ReadVariableOp.Adam/dec_middle_2/kernel/m/Read/ReadVariableOp,Adam/dec_middle_2/bias/m/Read/ReadVariableOp.Adam/dec_middle_3/kernel/m/Read/ReadVariableOp,Adam/dec_middle_3/bias/m/Read/ReadVariableOp-Adam/dec_outer_0/kernel/m/Read/ReadVariableOp+Adam/dec_outer_0/bias/m/Read/ReadVariableOp-Adam/dec_outer_1/kernel/m/Read/ReadVariableOp+Adam/dec_outer_1/bias/m/Read/ReadVariableOp-Adam/dec_outer_2/kernel/m/Read/ReadVariableOp+Adam/dec_outer_2/bias/m/Read/ReadVariableOp-Adam/dec_outer_3/kernel/m/Read/ReadVariableOp+Adam/dec_outer_3/bias/m/Read/ReadVariableOp,Adam/dec_output/kernel/m/Read/ReadVariableOp*Adam/dec_output/bias/m/Read/ReadVariableOp-Adam/enc_outer_0/kernel/v/Read/ReadVariableOp+Adam/enc_outer_0/bias/v/Read/ReadVariableOp-Adam/enc_outer_1/kernel/v/Read/ReadVariableOp+Adam/enc_outer_1/bias/v/Read/ReadVariableOp-Adam/enc_outer_2/kernel/v/Read/ReadVariableOp+Adam/enc_outer_2/bias/v/Read/ReadVariableOp-Adam/enc_outer_3/kernel/v/Read/ReadVariableOp+Adam/enc_outer_3/bias/v/Read/ReadVariableOp.Adam/enc_middle_0/kernel/v/Read/ReadVariableOp,Adam/enc_middle_0/bias/v/Read/ReadVariableOp.Adam/enc_middle_1/kernel/v/Read/ReadVariableOp,Adam/enc_middle_1/bias/v/Read/ReadVariableOp.Adam/enc_middle_2/kernel/v/Read/ReadVariableOp,Adam/enc_middle_2/bias/v/Read/ReadVariableOp.Adam/enc_middle_3/kernel/v/Read/ReadVariableOp,Adam/enc_middle_3/bias/v/Read/ReadVariableOp-Adam/enc_inner_0/kernel/v/Read/ReadVariableOp+Adam/enc_inner_0/bias/v/Read/ReadVariableOp-Adam/enc_inner_1/kernel/v/Read/ReadVariableOp+Adam/enc_inner_1/bias/v/Read/ReadVariableOp-Adam/enc_inner_2/kernel/v/Read/ReadVariableOp+Adam/enc_inner_2/bias/v/Read/ReadVariableOp-Adam/enc_inner_3/kernel/v/Read/ReadVariableOp+Adam/enc_inner_3/bias/v/Read/ReadVariableOp+Adam/channel_0/kernel/v/Read/ReadVariableOp)Adam/channel_0/bias/v/Read/ReadVariableOp+Adam/channel_1/kernel/v/Read/ReadVariableOp)Adam/channel_1/bias/v/Read/ReadVariableOp+Adam/channel_2/kernel/v/Read/ReadVariableOp)Adam/channel_2/bias/v/Read/ReadVariableOp+Adam/channel_3/kernel/v/Read/ReadVariableOp)Adam/channel_3/bias/v/Read/ReadVariableOp-Adam/dec_inner_0/kernel/v/Read/ReadVariableOp+Adam/dec_inner_0/bias/v/Read/ReadVariableOp-Adam/dec_inner_1/kernel/v/Read/ReadVariableOp+Adam/dec_inner_1/bias/v/Read/ReadVariableOp-Adam/dec_inner_2/kernel/v/Read/ReadVariableOp+Adam/dec_inner_2/bias/v/Read/ReadVariableOp-Adam/dec_inner_3/kernel/v/Read/ReadVariableOp+Adam/dec_inner_3/bias/v/Read/ReadVariableOp.Adam/dec_middle_0/kernel/v/Read/ReadVariableOp,Adam/dec_middle_0/bias/v/Read/ReadVariableOp.Adam/dec_middle_1/kernel/v/Read/ReadVariableOp,Adam/dec_middle_1/bias/v/Read/ReadVariableOp.Adam/dec_middle_2/kernel/v/Read/ReadVariableOp,Adam/dec_middle_2/bias/v/Read/ReadVariableOp.Adam/dec_middle_3/kernel/v/Read/ReadVariableOp,Adam/dec_middle_3/bias/v/Read/ReadVariableOp-Adam/dec_outer_0/kernel/v/Read/ReadVariableOp+Adam/dec_outer_0/bias/v/Read/ReadVariableOp-Adam/dec_outer_1/kernel/v/Read/ReadVariableOp+Adam/dec_outer_1/bias/v/Read/ReadVariableOp-Adam/dec_outer_2/kernel/v/Read/ReadVariableOp+Adam/dec_outer_2/bias/v/Read/ReadVariableOp-Adam/dec_outer_3/kernel/v/Read/ReadVariableOp+Adam/dec_outer_3/bias/v/Read/ReadVariableOp,Adam/dec_output/kernel/v/Read/ReadVariableOp*Adam/dec_output/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
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
__inference__traced_save_309800
?$
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateenc_outer_0/kernelenc_outer_0/biasenc_outer_1/kernelenc_outer_1/biasenc_outer_2/kernelenc_outer_2/biasenc_outer_3/kernelenc_outer_3/biasenc_middle_0/kernelenc_middle_0/biasenc_middle_1/kernelenc_middle_1/biasenc_middle_2/kernelenc_middle_2/biasenc_middle_3/kernelenc_middle_3/biasenc_inner_0/kernelenc_inner_0/biasenc_inner_1/kernelenc_inner_1/biasenc_inner_2/kernelenc_inner_2/biasenc_inner_3/kernelenc_inner_3/biaschannel_0/kernelchannel_0/biaschannel_1/kernelchannel_1/biaschannel_2/kernelchannel_2/biaschannel_3/kernelchannel_3/biasdec_inner_0/kerneldec_inner_0/biasdec_inner_1/kerneldec_inner_1/biasdec_inner_2/kerneldec_inner_2/biasdec_inner_3/kerneldec_inner_3/biasdec_middle_0/kerneldec_middle_0/biasdec_middle_1/kerneldec_middle_1/biasdec_middle_2/kerneldec_middle_2/biasdec_middle_3/kerneldec_middle_3/biasdec_outer_0/kerneldec_outer_0/biasdec_outer_1/kerneldec_outer_1/biasdec_outer_2/kerneldec_outer_2/biasdec_outer_3/kerneldec_outer_3/biasdec_output/kerneldec_output/biastotalcountAdam/enc_outer_0/kernel/mAdam/enc_outer_0/bias/mAdam/enc_outer_1/kernel/mAdam/enc_outer_1/bias/mAdam/enc_outer_2/kernel/mAdam/enc_outer_2/bias/mAdam/enc_outer_3/kernel/mAdam/enc_outer_3/bias/mAdam/enc_middle_0/kernel/mAdam/enc_middle_0/bias/mAdam/enc_middle_1/kernel/mAdam/enc_middle_1/bias/mAdam/enc_middle_2/kernel/mAdam/enc_middle_2/bias/mAdam/enc_middle_3/kernel/mAdam/enc_middle_3/bias/mAdam/enc_inner_0/kernel/mAdam/enc_inner_0/bias/mAdam/enc_inner_1/kernel/mAdam/enc_inner_1/bias/mAdam/enc_inner_2/kernel/mAdam/enc_inner_2/bias/mAdam/enc_inner_3/kernel/mAdam/enc_inner_3/bias/mAdam/channel_0/kernel/mAdam/channel_0/bias/mAdam/channel_1/kernel/mAdam/channel_1/bias/mAdam/channel_2/kernel/mAdam/channel_2/bias/mAdam/channel_3/kernel/mAdam/channel_3/bias/mAdam/dec_inner_0/kernel/mAdam/dec_inner_0/bias/mAdam/dec_inner_1/kernel/mAdam/dec_inner_1/bias/mAdam/dec_inner_2/kernel/mAdam/dec_inner_2/bias/mAdam/dec_inner_3/kernel/mAdam/dec_inner_3/bias/mAdam/dec_middle_0/kernel/mAdam/dec_middle_0/bias/mAdam/dec_middle_1/kernel/mAdam/dec_middle_1/bias/mAdam/dec_middle_2/kernel/mAdam/dec_middle_2/bias/mAdam/dec_middle_3/kernel/mAdam/dec_middle_3/bias/mAdam/dec_outer_0/kernel/mAdam/dec_outer_0/bias/mAdam/dec_outer_1/kernel/mAdam/dec_outer_1/bias/mAdam/dec_outer_2/kernel/mAdam/dec_outer_2/bias/mAdam/dec_outer_3/kernel/mAdam/dec_outer_3/bias/mAdam/dec_output/kernel/mAdam/dec_output/bias/mAdam/enc_outer_0/kernel/vAdam/enc_outer_0/bias/vAdam/enc_outer_1/kernel/vAdam/enc_outer_1/bias/vAdam/enc_outer_2/kernel/vAdam/enc_outer_2/bias/vAdam/enc_outer_3/kernel/vAdam/enc_outer_3/bias/vAdam/enc_middle_0/kernel/vAdam/enc_middle_0/bias/vAdam/enc_middle_1/kernel/vAdam/enc_middle_1/bias/vAdam/enc_middle_2/kernel/vAdam/enc_middle_2/bias/vAdam/enc_middle_3/kernel/vAdam/enc_middle_3/bias/vAdam/enc_inner_0/kernel/vAdam/enc_inner_0/bias/vAdam/enc_inner_1/kernel/vAdam/enc_inner_1/bias/vAdam/enc_inner_2/kernel/vAdam/enc_inner_2/bias/vAdam/enc_inner_3/kernel/vAdam/enc_inner_3/bias/vAdam/channel_0/kernel/vAdam/channel_0/bias/vAdam/channel_1/kernel/vAdam/channel_1/bias/vAdam/channel_2/kernel/vAdam/channel_2/bias/vAdam/channel_3/kernel/vAdam/channel_3/bias/vAdam/dec_inner_0/kernel/vAdam/dec_inner_0/bias/vAdam/dec_inner_1/kernel/vAdam/dec_inner_1/bias/vAdam/dec_inner_2/kernel/vAdam/dec_inner_2/bias/vAdam/dec_inner_3/kernel/vAdam/dec_inner_3/bias/vAdam/dec_middle_0/kernel/vAdam/dec_middle_0/bias/vAdam/dec_middle_1/kernel/vAdam/dec_middle_1/bias/vAdam/dec_middle_2/kernel/vAdam/dec_middle_2/bias/vAdam/dec_middle_3/kernel/vAdam/dec_middle_3/bias/vAdam/dec_outer_0/kernel/vAdam/dec_outer_0/bias/vAdam/dec_outer_1/kernel/vAdam/dec_outer_1/bias/vAdam/dec_outer_2/kernel/vAdam/dec_outer_2/bias/vAdam/dec_outer_3/kernel/vAdam/dec_outer_3/bias/vAdam/dec_output/kernel/vAdam/dec_output/bias/v*?
Tin?
?2?*
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
"__inference__traced_restore_310353??$
?	
?
E__inference_channel_0_layer_call_and_return_conditional_losses_308905

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
,__inference_dec_inner_3_layer_call_fn_309054

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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_3054552
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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_304600

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
F__inference_dec_output_layer_call_and_return_conditional_losses_305781

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
?
?
,__inference_enc_inner_0_layer_call_fn_308834

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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_3048972
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
??
?
C__inference_model_6_layer_call_and_return_conditional_losses_308065

inputs.
*enc_outer_3_matmul_readvariableop_resource/
+enc_outer_3_biasadd_readvariableop_resource.
*enc_outer_2_matmul_readvariableop_resource/
+enc_outer_2_biasadd_readvariableop_resource.
*enc_outer_1_matmul_readvariableop_resource/
+enc_outer_1_biasadd_readvariableop_resource.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_3_matmul_readvariableop_resource0
,enc_middle_3_biasadd_readvariableop_resource/
+enc_middle_2_matmul_readvariableop_resource0
,enc_middle_2_biasadd_readvariableop_resource/
+enc_middle_1_matmul_readvariableop_resource0
,enc_middle_1_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_3_matmul_readvariableop_resource/
+enc_inner_3_biasadd_readvariableop_resource.
*enc_inner_2_matmul_readvariableop_resource/
+enc_inner_2_biasadd_readvariableop_resource.
*enc_inner_1_matmul_readvariableop_resource/
+enc_inner_1_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_3_matmul_readvariableop_resource-
)channel_3_biasadd_readvariableop_resource,
(channel_2_matmul_readvariableop_resource-
)channel_2_biasadd_readvariableop_resource,
(channel_1_matmul_readvariableop_resource-
)channel_1_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp? channel_1/BiasAdd/ReadVariableOp?channel_1/MatMul/ReadVariableOp? channel_2/BiasAdd/ReadVariableOp?channel_2/MatMul/ReadVariableOp? channel_3/BiasAdd/ReadVariableOp?channel_3/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?"enc_inner_1/BiasAdd/ReadVariableOp?!enc_inner_1/MatMul/ReadVariableOp?"enc_inner_2/BiasAdd/ReadVariableOp?!enc_inner_2/MatMul/ReadVariableOp?"enc_inner_3/BiasAdd/ReadVariableOp?!enc_inner_3/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?#enc_middle_1/BiasAdd/ReadVariableOp?"enc_middle_1/MatMul/ReadVariableOp?#enc_middle_2/BiasAdd/ReadVariableOp?"enc_middle_2/MatMul/ReadVariableOp?#enc_middle_3/BiasAdd/ReadVariableOp?"enc_middle_3/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?"enc_outer_1/BiasAdd/ReadVariableOp?!enc_outer_1/MatMul/ReadVariableOp?"enc_outer_2/BiasAdd/ReadVariableOp?!enc_outer_2/MatMul/ReadVariableOp?"enc_outer_3/BiasAdd/ReadVariableOp?!enc_outer_3/MatMul/ReadVariableOp?
!enc_outer_3/MatMul/ReadVariableOpReadVariableOp*enc_outer_3_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_3/MatMul/ReadVariableOp?
enc_outer_3/MatMulMatMulinputs)enc_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_3/MatMul?
"enc_outer_3/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_3/BiasAdd/ReadVariableOp?
enc_outer_3/BiasAddBiasAddenc_outer_3/MatMul:product:0*enc_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_3/BiasAdd|
enc_outer_3/ReluReluenc_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_3/Relu?
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
"enc_middle_3/MatMul/ReadVariableOpReadVariableOp+enc_middle_3_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_3/MatMul/ReadVariableOp?
enc_middle_3/MatMulMatMulenc_outer_3/Relu:activations:0*enc_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_3/MatMul?
#enc_middle_3/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_3/BiasAdd/ReadVariableOp?
enc_middle_3/BiasAddBiasAddenc_middle_3/MatMul:product:0+enc_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_3/BiasAdd
enc_middle_3/ReluReluenc_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_3/Relu?
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
!enc_inner_3/MatMul/ReadVariableOpReadVariableOp*enc_inner_3_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_3/MatMul/ReadVariableOp?
enc_inner_3/MatMulMatMulenc_middle_3/Relu:activations:0)enc_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_3/MatMul?
"enc_inner_3/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_3/BiasAdd/ReadVariableOp?
enc_inner_3/BiasAddBiasAddenc_inner_3/MatMul:product:0*enc_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_3/BiasAdd|
enc_inner_3/ReluReluenc_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_3/Relu?
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
channel_3/MatMul/ReadVariableOpReadVariableOp(channel_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_3/MatMul/ReadVariableOp?
channel_3/MatMulMatMulenc_inner_3/Relu:activations:0'channel_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_3/MatMul?
 channel_3/BiasAdd/ReadVariableOpReadVariableOp)channel_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_3/BiasAdd/ReadVariableOp?
channel_3/BiasAddBiasAddchannel_3/MatMul:product:0(channel_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_3/BiasAdd?
channel_3/SoftsignSoftsignchannel_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_3/Softsign?
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
channel_0/Softsign?	
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?


Identity_1Identity channel_1/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?


Identity_2Identity channel_2/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?


Identity_3Identity channel_3/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2D
 channel_1/BiasAdd/ReadVariableOp channel_1/BiasAdd/ReadVariableOp2B
channel_1/MatMul/ReadVariableOpchannel_1/MatMul/ReadVariableOp2D
 channel_2/BiasAdd/ReadVariableOp channel_2/BiasAdd/ReadVariableOp2B
channel_2/MatMul/ReadVariableOpchannel_2/MatMul/ReadVariableOp2D
 channel_3/BiasAdd/ReadVariableOp channel_3/BiasAdd/ReadVariableOp2B
channel_3/MatMul/ReadVariableOpchannel_3/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2H
"enc_inner_1/BiasAdd/ReadVariableOp"enc_inner_1/BiasAdd/ReadVariableOp2F
!enc_inner_1/MatMul/ReadVariableOp!enc_inner_1/MatMul/ReadVariableOp2H
"enc_inner_2/BiasAdd/ReadVariableOp"enc_inner_2/BiasAdd/ReadVariableOp2F
!enc_inner_2/MatMul/ReadVariableOp!enc_inner_2/MatMul/ReadVariableOp2H
"enc_inner_3/BiasAdd/ReadVariableOp"enc_inner_3/BiasAdd/ReadVariableOp2F
!enc_inner_3/MatMul/ReadVariableOp!enc_inner_3/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2J
#enc_middle_1/BiasAdd/ReadVariableOp#enc_middle_1/BiasAdd/ReadVariableOp2H
"enc_middle_1/MatMul/ReadVariableOp"enc_middle_1/MatMul/ReadVariableOp2J
#enc_middle_2/BiasAdd/ReadVariableOp#enc_middle_2/BiasAdd/ReadVariableOp2H
"enc_middle_2/MatMul/ReadVariableOp"enc_middle_2/MatMul/ReadVariableOp2J
#enc_middle_3/BiasAdd/ReadVariableOp#enc_middle_3/BiasAdd/ReadVariableOp2H
"enc_middle_3/MatMul/ReadVariableOp"enc_middle_3/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp2H
"enc_outer_1/BiasAdd/ReadVariableOp"enc_outer_1/BiasAdd/ReadVariableOp2F
!enc_outer_1/MatMul/ReadVariableOp!enc_outer_1/MatMul/ReadVariableOp2H
"enc_outer_2/BiasAdd/ReadVariableOp"enc_outer_2/BiasAdd/ReadVariableOp2F
!enc_outer_2/MatMul/ReadVariableOp!enc_outer_2/MatMul/ReadVariableOp2H
"enc_outer_3/BiasAdd/ReadVariableOp"enc_outer_3/BiasAdd/ReadVariableOp2F
!enc_outer_3/MatMul/ReadVariableOp!enc_outer_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_305563

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
?
?
(__inference_model_6_layer_call_fn_305437
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30
identity

identity_1

identity_2

identity_3??StatefulPartitionedCall?
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3053642
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

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?
?
(__inference_model_7_layer_call_fn_308654
inputs_0
inputs_1
inputs_2
inputs_3
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

unknown_24
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3060862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::22
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
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
??
?
C__inference_model_6_layer_call_and_return_conditional_losses_308184

inputs.
*enc_outer_3_matmul_readvariableop_resource/
+enc_outer_3_biasadd_readvariableop_resource.
*enc_outer_2_matmul_readvariableop_resource/
+enc_outer_2_biasadd_readvariableop_resource.
*enc_outer_1_matmul_readvariableop_resource/
+enc_outer_1_biasadd_readvariableop_resource.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_3_matmul_readvariableop_resource0
,enc_middle_3_biasadd_readvariableop_resource/
+enc_middle_2_matmul_readvariableop_resource0
,enc_middle_2_biasadd_readvariableop_resource/
+enc_middle_1_matmul_readvariableop_resource0
,enc_middle_1_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_3_matmul_readvariableop_resource/
+enc_inner_3_biasadd_readvariableop_resource.
*enc_inner_2_matmul_readvariableop_resource/
+enc_inner_2_biasadd_readvariableop_resource.
*enc_inner_1_matmul_readvariableop_resource/
+enc_inner_1_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_3_matmul_readvariableop_resource-
)channel_3_biasadd_readvariableop_resource,
(channel_2_matmul_readvariableop_resource-
)channel_2_biasadd_readvariableop_resource,
(channel_1_matmul_readvariableop_resource-
)channel_1_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp? channel_1/BiasAdd/ReadVariableOp?channel_1/MatMul/ReadVariableOp? channel_2/BiasAdd/ReadVariableOp?channel_2/MatMul/ReadVariableOp? channel_3/BiasAdd/ReadVariableOp?channel_3/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?"enc_inner_1/BiasAdd/ReadVariableOp?!enc_inner_1/MatMul/ReadVariableOp?"enc_inner_2/BiasAdd/ReadVariableOp?!enc_inner_2/MatMul/ReadVariableOp?"enc_inner_3/BiasAdd/ReadVariableOp?!enc_inner_3/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?#enc_middle_1/BiasAdd/ReadVariableOp?"enc_middle_1/MatMul/ReadVariableOp?#enc_middle_2/BiasAdd/ReadVariableOp?"enc_middle_2/MatMul/ReadVariableOp?#enc_middle_3/BiasAdd/ReadVariableOp?"enc_middle_3/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?"enc_outer_1/BiasAdd/ReadVariableOp?!enc_outer_1/MatMul/ReadVariableOp?"enc_outer_2/BiasAdd/ReadVariableOp?!enc_outer_2/MatMul/ReadVariableOp?"enc_outer_3/BiasAdd/ReadVariableOp?!enc_outer_3/MatMul/ReadVariableOp?
!enc_outer_3/MatMul/ReadVariableOpReadVariableOp*enc_outer_3_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02#
!enc_outer_3/MatMul/ReadVariableOp?
enc_outer_3/MatMulMatMulinputs)enc_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_3/MatMul?
"enc_outer_3/BiasAdd/ReadVariableOpReadVariableOp+enc_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"enc_outer_3/BiasAdd/ReadVariableOp?
enc_outer_3/BiasAddBiasAddenc_outer_3/MatMul:product:0*enc_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
enc_outer_3/BiasAdd|
enc_outer_3/ReluReluenc_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
enc_outer_3/Relu?
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
"enc_middle_3/MatMul/ReadVariableOpReadVariableOp+enc_middle_3_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02$
"enc_middle_3/MatMul/ReadVariableOp?
enc_middle_3/MatMulMatMulenc_outer_3/Relu:activations:0*enc_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_3/MatMul?
#enc_middle_3/BiasAdd/ReadVariableOpReadVariableOp,enc_middle_3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#enc_middle_3/BiasAdd/ReadVariableOp?
enc_middle_3/BiasAddBiasAddenc_middle_3/MatMul:product:0+enc_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
enc_middle_3/BiasAdd
enc_middle_3/ReluReluenc_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
enc_middle_3/Relu?
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
!enc_inner_3/MatMul/ReadVariableOpReadVariableOp*enc_inner_3_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02#
!enc_inner_3/MatMul/ReadVariableOp?
enc_inner_3/MatMulMatMulenc_middle_3/Relu:activations:0)enc_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_3/MatMul?
"enc_inner_3/BiasAdd/ReadVariableOpReadVariableOp+enc_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"enc_inner_3/BiasAdd/ReadVariableOp?
enc_inner_3/BiasAddBiasAddenc_inner_3/MatMul:product:0*enc_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
enc_inner_3/BiasAdd|
enc_inner_3/ReluReluenc_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
enc_inner_3/Relu?
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
channel_3/MatMul/ReadVariableOpReadVariableOp(channel_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02!
channel_3/MatMul/ReadVariableOp?
channel_3/MatMulMatMulenc_inner_3/Relu:activations:0'channel_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_3/MatMul?
 channel_3/BiasAdd/ReadVariableOpReadVariableOp)channel_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 channel_3/BiasAdd/ReadVariableOp?
channel_3/BiasAddBiasAddchannel_3/MatMul:product:0(channel_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
channel_3/BiasAdd?
channel_3/SoftsignSoftsignchannel_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
channel_3/Softsign?
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
channel_0/Softsign?	
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?


Identity_1Identity channel_1/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?


Identity_2Identity channel_2/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?


Identity_3Identity channel_3/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp!^channel_2/BiasAdd/ReadVariableOp ^channel_2/MatMul/ReadVariableOp!^channel_3/BiasAdd/ReadVariableOp ^channel_3/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp#^enc_inner_2/BiasAdd/ReadVariableOp"^enc_inner_2/MatMul/ReadVariableOp#^enc_inner_3/BiasAdd/ReadVariableOp"^enc_inner_3/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp$^enc_middle_2/BiasAdd/ReadVariableOp#^enc_middle_2/MatMul/ReadVariableOp$^enc_middle_3/BiasAdd/ReadVariableOp#^enc_middle_3/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp#^enc_outer_2/BiasAdd/ReadVariableOp"^enc_outer_2/MatMul/ReadVariableOp#^enc_outer_3/BiasAdd/ReadVariableOp"^enc_outer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2D
 channel_1/BiasAdd/ReadVariableOp channel_1/BiasAdd/ReadVariableOp2B
channel_1/MatMul/ReadVariableOpchannel_1/MatMul/ReadVariableOp2D
 channel_2/BiasAdd/ReadVariableOp channel_2/BiasAdd/ReadVariableOp2B
channel_2/MatMul/ReadVariableOpchannel_2/MatMul/ReadVariableOp2D
 channel_3/BiasAdd/ReadVariableOp channel_3/BiasAdd/ReadVariableOp2B
channel_3/MatMul/ReadVariableOpchannel_3/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2H
"enc_inner_1/BiasAdd/ReadVariableOp"enc_inner_1/BiasAdd/ReadVariableOp2F
!enc_inner_1/MatMul/ReadVariableOp!enc_inner_1/MatMul/ReadVariableOp2H
"enc_inner_2/BiasAdd/ReadVariableOp"enc_inner_2/BiasAdd/ReadVariableOp2F
!enc_inner_2/MatMul/ReadVariableOp!enc_inner_2/MatMul/ReadVariableOp2H
"enc_inner_3/BiasAdd/ReadVariableOp"enc_inner_3/BiasAdd/ReadVariableOp2F
!enc_inner_3/MatMul/ReadVariableOp!enc_inner_3/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2J
#enc_middle_1/BiasAdd/ReadVariableOp#enc_middle_1/BiasAdd/ReadVariableOp2H
"enc_middle_1/MatMul/ReadVariableOp"enc_middle_1/MatMul/ReadVariableOp2J
#enc_middle_2/BiasAdd/ReadVariableOp#enc_middle_2/BiasAdd/ReadVariableOp2H
"enc_middle_2/MatMul/ReadVariableOp"enc_middle_2/MatMul/ReadVariableOp2J
#enc_middle_3/BiasAdd/ReadVariableOp#enc_middle_3/BiasAdd/ReadVariableOp2H
"enc_middle_3/MatMul/ReadVariableOp"enc_middle_3/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp2H
"enc_outer_1/BiasAdd/ReadVariableOp"enc_outer_1/BiasAdd/ReadVariableOp2F
!enc_outer_1/MatMul/ReadVariableOp!enc_outer_1/MatMul/ReadVariableOp2H
"enc_outer_2/BiasAdd/ReadVariableOp"enc_outer_2/BiasAdd/ReadVariableOp2F
!enc_outer_2/MatMul/ReadVariableOp!enc_outer_2/MatMul/ReadVariableOp2H
"enc_outer_3/BiasAdd/ReadVariableOp"enc_outer_3/BiasAdd/ReadVariableOp2F
!enc_outer_3/MatMul/ReadVariableOp!enc_outer_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_304654

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
,__inference_enc_inner_1_layer_call_fn_308854

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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_3048702
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
E__inference_channel_0_layer_call_and_return_conditional_losses_305005

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
E__inference_channel_3_layer_call_and_return_conditional_losses_304924

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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_305536

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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_308725

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
,__inference_enc_outer_3_layer_call_fn_308734

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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_3046002
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
?L
?	
C__inference_model_7_layer_call_and_return_conditional_losses_305952

inputs
inputs_1
inputs_2
inputs_3
dec_inner_3_305884
dec_inner_3_305886
dec_inner_2_305889
dec_inner_2_305891
dec_inner_1_305894
dec_inner_1_305896
dec_inner_0_305899
dec_inner_0_305901
dec_middle_3_305904
dec_middle_3_305906
dec_middle_2_305909
dec_middle_2_305911
dec_middle_1_305914
dec_middle_1_305916
dec_middle_0_305919
dec_middle_0_305921
dec_outer_0_305924
dec_outer_0_305926
dec_outer_1_305929
dec_outer_1_305931
dec_outer_2_305934
dec_outer_2_305936
dec_outer_3_305939
dec_outer_3_305941
dec_output_305946
dec_output_305948
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?#dec_inner_3/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?$dec_middle_3/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?#dec_outer_3/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_3/StatefulPartitionedCallStatefulPartitionedCallinputs_3dec_inner_3_305884dec_inner_3_305886*
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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_3054552%
#dec_inner_3/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2dec_inner_2_305889dec_inner_2_305891*
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
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_3054822%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1dec_inner_1_305894dec_inner_1_305896*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_3055092%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_305899dec_inner_0_305901*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_3055362%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_3/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_3/StatefulPartitionedCall:output:0dec_middle_3_305904dec_middle_3_305906*
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
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_3055632&
$dec_middle_3/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_305909dec_middle_2_305911*
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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_3055902&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_305914dec_middle_1_305916*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_3056172&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_305919dec_middle_0_305921*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_3056442&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_305924dec_outer_0_305926*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_3056712%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_305929dec_outer_1_305931*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_3056982%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_305934dec_outer_2_305936*
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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_3057252%
#dec_outer_2/StatefulPartitionedCall?
#dec_outer_3/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_3/StatefulPartitionedCall:output:0dec_outer_3_305939dec_outer_3_305941*
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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_3057522%
#dec_outer_3/StatefulPartitionedCallt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0,dec_outer_3/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dec_output_305946dec_output_305948*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_3057812$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall$^dec_inner_3/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall%^dec_middle_3/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall$^dec_outer_3/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2J
#dec_inner_3/StatefulPartitionedCall#dec_inner_3/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2L
$dec_middle_3/StatefulPartitionedCall$dec_middle_3/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2J
#dec_outer_3/StatefulPartitionedCall#dec_outer_3/StatefulPartitionedCall2H
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_channel_0_layer_call_fn_308914

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
E__inference_channel_0_layer_call_and_return_conditional_losses_3050052
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
?
?
,__inference_dec_outer_3_layer_call_fn_309214

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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_3057522
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
E__inference_channel_3_layer_call_and_return_conditional_losses_308965

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
?
?
(__inference_model_7_layer_call_fn_306007
decoder_input_0
decoder_input_1
decoder_input_2
decoder_input_3
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

unknown_24
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0decoder_input_1decoder_input_2decoder_input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3059522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::22
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
_user_specified_namedecoder_input_2:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_3
?
?
,__inference_enc_outer_2_layer_call_fn_308714

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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_3046272
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
?	
?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_305644

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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_308745

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
?

*__inference_channel_2_layer_call_fn_308954

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
E__inference_channel_2_layer_call_and_return_conditional_losses_3049512
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
?
?
+__inference_dec_output_layer_call_fn_309234

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
F__inference_dec_output_layer_call_and_return_conditional_losses_3057812
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
?	
?
E__inference_channel_2_layer_call_and_return_conditional_losses_304951

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
?

I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307036
x
model_6_306914
model_6_306916
model_6_306918
model_6_306920
model_6_306922
model_6_306924
model_6_306926
model_6_306928
model_6_306930
model_6_306932
model_6_306934
model_6_306936
model_6_306938
model_6_306940
model_6_306942
model_6_306944
model_6_306946
model_6_306948
model_6_306950
model_6_306952
model_6_306954
model_6_306956
model_6_306958
model_6_306960
model_6_306962
model_6_306964
model_6_306966
model_6_306968
model_6_306970
model_6_306972
model_6_306974
model_6_306976
model_7_306982
model_7_306984
model_7_306986
model_7_306988
model_7_306990
model_7_306992
model_7_306994
model_7_306996
model_7_306998
model_7_307000
model_7_307002
model_7_307004
model_7_307006
model_7_307008
model_7_307010
model_7_307012
model_7_307014
model_7_307016
model_7_307018
model_7_307020
model_7_307022
model_7_307024
model_7_307026
model_7_307028
model_7_307030
model_7_307032
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallxmodel_6_306914model_6_306916model_6_306918model_6_306920model_6_306922model_6_306924model_6_306926model_6_306928model_6_306930model_6_306932model_6_306934model_6_306936model_6_306938model_6_306940model_6_306942model_6_306944model_6_306946model_6_306948model_6_306950model_6_306952model_6_306954model_6_306956model_6_306958model_6_306960model_6_306962model_6_306964model_6_306966model_6_306968model_6_306970model_6_306972model_6_306974model_6_306976*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3053642!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0(model_6/StatefulPartitionedCall:output:1(model_6/StatefulPartitionedCall:output:2(model_6/StatefulPartitionedCall:output:3model_7_306982model_7_306984model_7_306986model_7_306988model_7_306990model_7_306992model_7_306994model_7_306996model_7_306998model_7_307000model_7_307002model_7_307004model_7_307006model_7_307008model_7_307010model_7_307012model_7_307014model_7_307016model_7_307018model_7_307020model_7_307022model_7_307024model_7_307026model_7_307028model_7_307030model_7_307032*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3060862!
model_7/StatefulPartitionedCall?
IdentityIdentity(model_7/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
,__inference_dec_outer_1_layer_call_fn_309174

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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_3056982
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_309065

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
?
?
.__inference_autoencoder_3_layer_call_fn_307155
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

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56
identity??StatefulPartitionedCall?	
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
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_3070362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
(__inference_model_7_layer_call_fn_306141
decoder_input_0
decoder_input_1
decoder_input_2
decoder_input_3
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

unknown_24
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0decoder_input_1decoder_input_2decoder_input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3060862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::22
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
_user_specified_namedecoder_input_2:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_3
?
?
-__inference_dec_middle_3_layer_call_fn_309134

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
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_3055632
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
-__inference_enc_middle_1_layer_call_fn_308774

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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_3047622
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
?	
?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_305671

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
?
?
-__inference_enc_middle_3_layer_call_fn_308814

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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_3047082
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
??
?:
!__inference__wrapped_model_304585
input_1D
@autoencoder_3_model_6_enc_outer_3_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_outer_3_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_outer_2_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_outer_2_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_outer_1_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_outer_1_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_outer_0_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_outer_0_biasadd_readvariableop_resourceE
Aautoencoder_3_model_6_enc_middle_3_matmul_readvariableop_resourceF
Bautoencoder_3_model_6_enc_middle_3_biasadd_readvariableop_resourceE
Aautoencoder_3_model_6_enc_middle_2_matmul_readvariableop_resourceF
Bautoencoder_3_model_6_enc_middle_2_biasadd_readvariableop_resourceE
Aautoencoder_3_model_6_enc_middle_1_matmul_readvariableop_resourceF
Bautoencoder_3_model_6_enc_middle_1_biasadd_readvariableop_resourceE
Aautoencoder_3_model_6_enc_middle_0_matmul_readvariableop_resourceF
Bautoencoder_3_model_6_enc_middle_0_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_inner_3_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_inner_3_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_inner_2_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_inner_2_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_inner_1_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_inner_1_biasadd_readvariableop_resourceD
@autoencoder_3_model_6_enc_inner_0_matmul_readvariableop_resourceE
Aautoencoder_3_model_6_enc_inner_0_biasadd_readvariableop_resourceB
>autoencoder_3_model_6_channel_3_matmul_readvariableop_resourceC
?autoencoder_3_model_6_channel_3_biasadd_readvariableop_resourceB
>autoencoder_3_model_6_channel_2_matmul_readvariableop_resourceC
?autoencoder_3_model_6_channel_2_biasadd_readvariableop_resourceB
>autoencoder_3_model_6_channel_1_matmul_readvariableop_resourceC
?autoencoder_3_model_6_channel_1_biasadd_readvariableop_resourceB
>autoencoder_3_model_6_channel_0_matmul_readvariableop_resourceC
?autoencoder_3_model_6_channel_0_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_inner_3_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_inner_3_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_inner_2_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_inner_2_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_inner_1_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_inner_1_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_inner_0_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_inner_0_biasadd_readvariableop_resourceE
Aautoencoder_3_model_7_dec_middle_3_matmul_readvariableop_resourceF
Bautoencoder_3_model_7_dec_middle_3_biasadd_readvariableop_resourceE
Aautoencoder_3_model_7_dec_middle_2_matmul_readvariableop_resourceF
Bautoencoder_3_model_7_dec_middle_2_biasadd_readvariableop_resourceE
Aautoencoder_3_model_7_dec_middle_1_matmul_readvariableop_resourceF
Bautoencoder_3_model_7_dec_middle_1_biasadd_readvariableop_resourceE
Aautoencoder_3_model_7_dec_middle_0_matmul_readvariableop_resourceF
Bautoencoder_3_model_7_dec_middle_0_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_outer_0_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_outer_0_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_outer_1_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_outer_1_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_outer_2_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_outer_2_biasadd_readvariableop_resourceD
@autoencoder_3_model_7_dec_outer_3_matmul_readvariableop_resourceE
Aautoencoder_3_model_7_dec_outer_3_biasadd_readvariableop_resourceC
?autoencoder_3_model_7_dec_output_matmul_readvariableop_resourceD
@autoencoder_3_model_7_dec_output_biasadd_readvariableop_resource
identity??6autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOp?5autoencoder_3/model_6/channel_0/MatMul/ReadVariableOp?6autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOp?5autoencoder_3/model_6/channel_1/MatMul/ReadVariableOp?6autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOp?5autoencoder_3/model_6/channel_2/MatMul/ReadVariableOp?6autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOp?5autoencoder_3/model_6/channel_3/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOp?9autoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOp?8autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOp?9autoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOp?8autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOp?9autoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOp?8autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOp?9autoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOp?8autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOp?8autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOp?7autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOp?9autoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOp?8autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOp?9autoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOp?8autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOp?9autoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOp?8autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOp?9autoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOp?8autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOp?8autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOp?7autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOp?7autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOp?6autoencoder_3/model_7/dec_output/MatMul/ReadVariableOp?
7autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_outer_3_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_outer_3/MatMulMatMulinput_1?autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_6/enc_outer_3/MatMul?
8autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_outer_3/BiasAddBiasAdd2autoencoder_3/model_6/enc_outer_3/MatMul:product:0@autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_6/enc_outer_3/BiasAdd?
&autoencoder_3/model_6/enc_outer_3/ReluRelu2autoencoder_3/model_6/enc_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_6/enc_outer_3/Relu?
7autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_outer_2/MatMulMatMulinput_1?autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_6/enc_outer_2/MatMul?
8autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_outer_2/BiasAddBiasAdd2autoencoder_3/model_6/enc_outer_2/MatMul:product:0@autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_6/enc_outer_2/BiasAdd?
&autoencoder_3/model_6/enc_outer_2/ReluRelu2autoencoder_3/model_6/enc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_6/enc_outer_2/Relu?
7autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_outer_1/MatMulMatMulinput_1?autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_6/enc_outer_1/MatMul?
8autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_outer_1/BiasAddBiasAdd2autoencoder_3/model_6/enc_outer_1/MatMul:product:0@autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_6/enc_outer_1/BiasAdd?
&autoencoder_3/model_6/enc_outer_1/ReluRelu2autoencoder_3/model_6/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_6/enc_outer_1/Relu?
7autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_outer_0/MatMulMatMulinput_1?autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_6/enc_outer_0/MatMul?
8autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_outer_0/BiasAddBiasAdd2autoencoder_3/model_6/enc_outer_0/MatMul:product:0@autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_6/enc_outer_0/BiasAdd?
&autoencoder_3/model_6/enc_outer_0/ReluRelu2autoencoder_3/model_6/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_6/enc_outer_0/Relu?
8autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_middle_3_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOp?
)autoencoder_3/model_6/enc_middle_3/MatMulMatMul4autoencoder_3/model_6/enc_outer_3/Relu:activations:0@autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_3/model_6/enc_middle_3/MatMul?
9autoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_6_enc_middle_3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOp?
*autoencoder_3/model_6/enc_middle_3/BiasAddBiasAdd3autoencoder_3/model_6/enc_middle_3/MatMul:product:0Aautoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_3/model_6/enc_middle_3/BiasAdd?
'autoencoder_3/model_6/enc_middle_3/ReluRelu3autoencoder_3/model_6/enc_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_3/model_6/enc_middle_3/Relu?
8autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOp?
)autoencoder_3/model_6/enc_middle_2/MatMulMatMul4autoencoder_3/model_6/enc_outer_2/Relu:activations:0@autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_3/model_6/enc_middle_2/MatMul?
9autoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_6_enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOp?
*autoencoder_3/model_6/enc_middle_2/BiasAddBiasAdd3autoencoder_3/model_6/enc_middle_2/MatMul:product:0Aautoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_3/model_6/enc_middle_2/BiasAdd?
'autoencoder_3/model_6/enc_middle_2/ReluRelu3autoencoder_3/model_6/enc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_3/model_6/enc_middle_2/Relu?
8autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOp?
)autoencoder_3/model_6/enc_middle_1/MatMulMatMul4autoencoder_3/model_6/enc_outer_1/Relu:activations:0@autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_3/model_6/enc_middle_1/MatMul?
9autoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_6_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOp?
*autoencoder_3/model_6/enc_middle_1/BiasAddBiasAdd3autoencoder_3/model_6/enc_middle_1/MatMul:product:0Aautoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_3/model_6/enc_middle_1/BiasAdd?
'autoencoder_3/model_6/enc_middle_1/ReluRelu3autoencoder_3/model_6/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_3/model_6/enc_middle_1/Relu?
8autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOp?
)autoencoder_3/model_6/enc_middle_0/MatMulMatMul4autoencoder_3/model_6/enc_outer_0/Relu:activations:0@autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_3/model_6/enc_middle_0/MatMul?
9autoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_6_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOp?
*autoencoder_3/model_6/enc_middle_0/BiasAddBiasAdd3autoencoder_3/model_6/enc_middle_0/MatMul:product:0Aautoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_3/model_6/enc_middle_0/BiasAdd?
'autoencoder_3/model_6/enc_middle_0/ReluRelu3autoencoder_3/model_6/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_3/model_6/enc_middle_0/Relu?
7autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_inner_3_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_inner_3/MatMulMatMul5autoencoder_3/model_6/enc_middle_3/Relu:activations:0?autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_6/enc_inner_3/MatMul?
8autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_inner_3/BiasAddBiasAdd2autoencoder_3/model_6/enc_inner_3/MatMul:product:0@autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_6/enc_inner_3/BiasAdd?
&autoencoder_3/model_6/enc_inner_3/ReluRelu2autoencoder_3/model_6/enc_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_6/enc_inner_3/Relu?
7autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_inner_2/MatMulMatMul5autoencoder_3/model_6/enc_middle_2/Relu:activations:0?autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_6/enc_inner_2/MatMul?
8autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_inner_2/BiasAddBiasAdd2autoencoder_3/model_6/enc_inner_2/MatMul:product:0@autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_6/enc_inner_2/BiasAdd?
&autoencoder_3/model_6/enc_inner_2/ReluRelu2autoencoder_3/model_6/enc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_6/enc_inner_2/Relu?
7autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_inner_1/MatMulMatMul5autoencoder_3/model_6/enc_middle_1/Relu:activations:0?autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_6/enc_inner_1/MatMul?
8autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_inner_1/BiasAddBiasAdd2autoencoder_3/model_6/enc_inner_1/MatMul:product:0@autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_6/enc_inner_1/BiasAdd?
&autoencoder_3/model_6/enc_inner_1/ReluRelu2autoencoder_3/model_6/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_6/enc_inner_1/Relu?
7autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_6_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOp?
(autoencoder_3/model_6/enc_inner_0/MatMulMatMul5autoencoder_3/model_6/enc_middle_0/Relu:activations:0?autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_6/enc_inner_0/MatMul?
8autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_6_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOp?
)autoencoder_3/model_6/enc_inner_0/BiasAddBiasAdd2autoencoder_3/model_6/enc_inner_0/MatMul:product:0@autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_6/enc_inner_0/BiasAdd?
&autoencoder_3/model_6/enc_inner_0/ReluRelu2autoencoder_3/model_6/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_6/enc_inner_0/Relu?
5autoencoder_3/model_6/channel_3/MatMul/ReadVariableOpReadVariableOp>autoencoder_3_model_6_channel_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_3/model_6/channel_3/MatMul/ReadVariableOp?
&autoencoder_3/model_6/channel_3/MatMulMatMul4autoencoder_3/model_6/enc_inner_3/Relu:activations:0=autoencoder_3/model_6/channel_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_3/model_6/channel_3/MatMul?
6autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_3_model_6_channel_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOp?
'autoencoder_3/model_6/channel_3/BiasAddBiasAdd0autoencoder_3/model_6/channel_3/MatMul:product:0>autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_3/model_6/channel_3/BiasAdd?
(autoencoder_3/model_6/channel_3/SoftsignSoftsign0autoencoder_3/model_6/channel_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_3/model_6/channel_3/Softsign?
5autoencoder_3/model_6/channel_2/MatMul/ReadVariableOpReadVariableOp>autoencoder_3_model_6_channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_3/model_6/channel_2/MatMul/ReadVariableOp?
&autoencoder_3/model_6/channel_2/MatMulMatMul4autoencoder_3/model_6/enc_inner_2/Relu:activations:0=autoencoder_3/model_6/channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_3/model_6/channel_2/MatMul?
6autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_3_model_6_channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOp?
'autoencoder_3/model_6/channel_2/BiasAddBiasAdd0autoencoder_3/model_6/channel_2/MatMul:product:0>autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_3/model_6/channel_2/BiasAdd?
(autoencoder_3/model_6/channel_2/SoftsignSoftsign0autoencoder_3/model_6/channel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_3/model_6/channel_2/Softsign?
5autoencoder_3/model_6/channel_1/MatMul/ReadVariableOpReadVariableOp>autoencoder_3_model_6_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_3/model_6/channel_1/MatMul/ReadVariableOp?
&autoencoder_3/model_6/channel_1/MatMulMatMul4autoencoder_3/model_6/enc_inner_1/Relu:activations:0=autoencoder_3/model_6/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_3/model_6/channel_1/MatMul?
6autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_3_model_6_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOp?
'autoencoder_3/model_6/channel_1/BiasAddBiasAdd0autoencoder_3/model_6/channel_1/MatMul:product:0>autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_3/model_6/channel_1/BiasAdd?
(autoencoder_3/model_6/channel_1/SoftsignSoftsign0autoencoder_3/model_6/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_3/model_6/channel_1/Softsign?
5autoencoder_3/model_6/channel_0/MatMul/ReadVariableOpReadVariableOp>autoencoder_3_model_6_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_3/model_6/channel_0/MatMul/ReadVariableOp?
&autoencoder_3/model_6/channel_0/MatMulMatMul4autoencoder_3/model_6/enc_inner_0/Relu:activations:0=autoencoder_3/model_6/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_3/model_6/channel_0/MatMul?
6autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_3_model_6_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOp?
'autoencoder_3/model_6/channel_0/BiasAddBiasAdd0autoencoder_3/model_6/channel_0/MatMul:product:0>autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_3/model_6/channel_0/BiasAdd?
(autoencoder_3/model_6/channel_0/SoftsignSoftsign0autoencoder_3/model_6/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_3/model_6/channel_0/Softsign?
7autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_inner_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_inner_3/MatMulMatMul6autoencoder_3/model_6/channel_3/Softsign:activations:0?autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_7/dec_inner_3/MatMul?
8autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_inner_3/BiasAddBiasAdd2autoencoder_3/model_7/dec_inner_3/MatMul:product:0@autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_7/dec_inner_3/BiasAdd?
&autoencoder_3/model_7/dec_inner_3/ReluRelu2autoencoder_3/model_7/dec_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_7/dec_inner_3/Relu?
7autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_inner_2/MatMulMatMul6autoencoder_3/model_6/channel_2/Softsign:activations:0?autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_7/dec_inner_2/MatMul?
8autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_inner_2/BiasAddBiasAdd2autoencoder_3/model_7/dec_inner_2/MatMul:product:0@autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_7/dec_inner_2/BiasAdd?
&autoencoder_3/model_7/dec_inner_2/ReluRelu2autoencoder_3/model_7/dec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_7/dec_inner_2/Relu?
7autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_inner_1/MatMulMatMul6autoencoder_3/model_6/channel_1/Softsign:activations:0?autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_7/dec_inner_1/MatMul?
8autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_inner_1/BiasAddBiasAdd2autoencoder_3/model_7/dec_inner_1/MatMul:product:0@autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_7/dec_inner_1/BiasAdd?
&autoencoder_3/model_7/dec_inner_1/ReluRelu2autoencoder_3/model_7/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_7/dec_inner_1/Relu?
7autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_inner_0/MatMulMatMul6autoencoder_3/model_6/channel_0/Softsign:activations:0?autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_3/model_7/dec_inner_0/MatMul?
8autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_inner_0/BiasAddBiasAdd2autoencoder_3/model_7/dec_inner_0/MatMul:product:0@autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_3/model_7/dec_inner_0/BiasAdd?
&autoencoder_3/model_7/dec_inner_0/ReluRelu2autoencoder_3/model_7/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_3/model_7/dec_inner_0/Relu?
8autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_middle_3_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOp?
)autoencoder_3/model_7/dec_middle_3/MatMulMatMul4autoencoder_3/model_7/dec_inner_3/Relu:activations:0@autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_middle_3/MatMul?
9autoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_7_dec_middle_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOp?
*autoencoder_3/model_7/dec_middle_3/BiasAddBiasAdd3autoencoder_3/model_7/dec_middle_3/MatMul:product:0Aautoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_3/model_7/dec_middle_3/BiasAdd?
'autoencoder_3/model_7/dec_middle_3/ReluRelu3autoencoder_3/model_7/dec_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_3/model_7/dec_middle_3/Relu?
8autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOp?
)autoencoder_3/model_7/dec_middle_2/MatMulMatMul4autoencoder_3/model_7/dec_inner_2/Relu:activations:0@autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_middle_2/MatMul?
9autoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_7_dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOp?
*autoencoder_3/model_7/dec_middle_2/BiasAddBiasAdd3autoencoder_3/model_7/dec_middle_2/MatMul:product:0Aautoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_3/model_7/dec_middle_2/BiasAdd?
'autoencoder_3/model_7/dec_middle_2/ReluRelu3autoencoder_3/model_7/dec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_3/model_7/dec_middle_2/Relu?
8autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOp?
)autoencoder_3/model_7/dec_middle_1/MatMulMatMul4autoencoder_3/model_7/dec_inner_1/Relu:activations:0@autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_middle_1/MatMul?
9autoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_7_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOp?
*autoencoder_3/model_7/dec_middle_1/BiasAddBiasAdd3autoencoder_3/model_7/dec_middle_1/MatMul:product:0Aautoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_3/model_7/dec_middle_1/BiasAdd?
'autoencoder_3/model_7/dec_middle_1/ReluRelu3autoencoder_3/model_7/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_3/model_7/dec_middle_1/Relu?
8autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOp?
)autoencoder_3/model_7/dec_middle_0/MatMulMatMul4autoencoder_3/model_7/dec_inner_0/Relu:activations:0@autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_middle_0/MatMul?
9autoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_model_7_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOp?
*autoencoder_3/model_7/dec_middle_0/BiasAddBiasAdd3autoencoder_3/model_7/dec_middle_0/MatMul:product:0Aautoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_3/model_7/dec_middle_0/BiasAdd?
'autoencoder_3/model_7/dec_middle_0/ReluRelu3autoencoder_3/model_7/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_3/model_7/dec_middle_0/Relu?
7autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_outer_0/MatMulMatMul5autoencoder_3/model_7/dec_middle_0/Relu:activations:0?autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_7/dec_outer_0/MatMul?
8autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_outer_0/BiasAddBiasAdd2autoencoder_3/model_7/dec_outer_0/MatMul:product:0@autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_outer_0/BiasAdd?
&autoencoder_3/model_7/dec_outer_0/ReluRelu2autoencoder_3/model_7/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_7/dec_outer_0/Relu?
7autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_outer_1/MatMulMatMul5autoencoder_3/model_7/dec_middle_1/Relu:activations:0?autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_7/dec_outer_1/MatMul?
8autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_outer_1/BiasAddBiasAdd2autoencoder_3/model_7/dec_outer_1/MatMul:product:0@autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_outer_1/BiasAdd?
&autoencoder_3/model_7/dec_outer_1/ReluRelu2autoencoder_3/model_7/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_7/dec_outer_1/Relu?
7autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_outer_2/MatMulMatMul5autoencoder_3/model_7/dec_middle_2/Relu:activations:0?autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_7/dec_outer_2/MatMul?
8autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_outer_2/BiasAddBiasAdd2autoencoder_3/model_7/dec_outer_2/MatMul:product:0@autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_outer_2/BiasAdd?
&autoencoder_3/model_7/dec_outer_2/ReluRelu2autoencoder_3/model_7/dec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_7/dec_outer_2/Relu?
7autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_outer_3_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOp?
(autoencoder_3/model_7/dec_outer_3/MatMulMatMul5autoencoder_3/model_7/dec_middle_3/Relu:activations:0?autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_3/model_7/dec_outer_3/MatMul?
8autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_3_model_7_dec_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOp?
)autoencoder_3/model_7/dec_outer_3/BiasAddBiasAdd2autoencoder_3/model_7/dec_outer_3/MatMul:product:0@autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_3/model_7/dec_outer_3/BiasAdd?
&autoencoder_3/model_7/dec_outer_3/ReluRelu2autoencoder_3/model_7/dec_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_3/model_7/dec_outer_3/Relu?
-autoencoder_3/model_7/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder_3/model_7/tf.concat_2/concat/axis?
(autoencoder_3/model_7/tf.concat_2/concatConcatV24autoencoder_3/model_7/dec_outer_0/Relu:activations:04autoencoder_3/model_7/dec_outer_1/Relu:activations:04autoencoder_3/model_7/dec_outer_2/Relu:activations:04autoencoder_3/model_7/dec_outer_3/Relu:activations:06autoencoder_3/model_7/tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2*
(autoencoder_3/model_7/tf.concat_2/concat?
6autoencoder_3/model_7/dec_output/MatMul/ReadVariableOpReadVariableOp?autoencoder_3_model_7_dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6autoencoder_3/model_7/dec_output/MatMul/ReadVariableOp?
'autoencoder_3/model_7/dec_output/MatMulMatMul1autoencoder_3/model_7/tf.concat_2/concat:output:0>autoencoder_3/model_7/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'autoencoder_3/model_7/dec_output/MatMul?
7autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_3_model_7_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOp?
(autoencoder_3/model_7/dec_output/BiasAddBiasAdd1autoencoder_3/model_7/dec_output/MatMul:product:0?autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(autoencoder_3/model_7/dec_output/BiasAdd?
(autoencoder_3/model_7/dec_output/SigmoidSigmoid1autoencoder_3/model_7/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2*
(autoencoder_3/model_7/dec_output/Sigmoid?
IdentityIdentity,autoencoder_3/model_7/dec_output/Sigmoid:y:07^autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOp6^autoencoder_3/model_6/channel_0/MatMul/ReadVariableOp7^autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOp6^autoencoder_3/model_6/channel_1/MatMul/ReadVariableOp7^autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOp6^autoencoder_3/model_6/channel_2/MatMul/ReadVariableOp7^autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOp6^autoencoder_3/model_6/channel_3/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOp:^autoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOp9^autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOp:^autoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOp9^autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOp:^autoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOp9^autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOp:^autoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOp9^autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOp9^autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOp8^autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOp:^autoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOp9^autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOp:^autoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOp9^autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOp:^autoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOp9^autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOp:^autoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOp9^autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOp9^autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOp8^autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOp8^autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOp7^autoencoder_3/model_7/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2p
6autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOp6autoencoder_3/model_6/channel_0/BiasAdd/ReadVariableOp2n
5autoencoder_3/model_6/channel_0/MatMul/ReadVariableOp5autoencoder_3/model_6/channel_0/MatMul/ReadVariableOp2p
6autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOp6autoencoder_3/model_6/channel_1/BiasAdd/ReadVariableOp2n
5autoencoder_3/model_6/channel_1/MatMul/ReadVariableOp5autoencoder_3/model_6/channel_1/MatMul/ReadVariableOp2p
6autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOp6autoencoder_3/model_6/channel_2/BiasAdd/ReadVariableOp2n
5autoencoder_3/model_6/channel_2/MatMul/ReadVariableOp5autoencoder_3/model_6/channel_2/MatMul/ReadVariableOp2p
6autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOp6autoencoder_3/model_6/channel_3/BiasAdd/ReadVariableOp2n
5autoencoder_3/model_6/channel_3/MatMul/ReadVariableOp5autoencoder_3/model_6/channel_3/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_inner_0/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_inner_0/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_inner_1/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_inner_1/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_inner_2/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_inner_2/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_inner_3/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_inner_3/MatMul/ReadVariableOp2v
9autoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOp9autoencoder_3/model_6/enc_middle_0/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOp8autoencoder_3/model_6/enc_middle_0/MatMul/ReadVariableOp2v
9autoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOp9autoencoder_3/model_6/enc_middle_1/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOp8autoencoder_3/model_6/enc_middle_1/MatMul/ReadVariableOp2v
9autoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOp9autoencoder_3/model_6/enc_middle_2/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOp8autoencoder_3/model_6/enc_middle_2/MatMul/ReadVariableOp2v
9autoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOp9autoencoder_3/model_6/enc_middle_3/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOp8autoencoder_3/model_6/enc_middle_3/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_outer_0/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_outer_0/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_outer_1/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_outer_1/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_outer_2/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_outer_2/MatMul/ReadVariableOp2t
8autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOp8autoencoder_3/model_6/enc_outer_3/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOp7autoencoder_3/model_6/enc_outer_3/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_inner_0/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_inner_0/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_inner_1/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_inner_1/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_inner_2/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_inner_2/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_inner_3/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_inner_3/MatMul/ReadVariableOp2v
9autoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOp9autoencoder_3/model_7/dec_middle_0/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOp8autoencoder_3/model_7/dec_middle_0/MatMul/ReadVariableOp2v
9autoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOp9autoencoder_3/model_7/dec_middle_1/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOp8autoencoder_3/model_7/dec_middle_1/MatMul/ReadVariableOp2v
9autoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOp9autoencoder_3/model_7/dec_middle_2/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOp8autoencoder_3/model_7/dec_middle_2/MatMul/ReadVariableOp2v
9autoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOp9autoencoder_3/model_7/dec_middle_3/BiasAdd/ReadVariableOp2t
8autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOp8autoencoder_3/model_7/dec_middle_3/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_outer_0/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_outer_0/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_outer_1/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_outer_1/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_outer_2/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_outer_2/MatMul/ReadVariableOp2t
8autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOp8autoencoder_3/model_7/dec_outer_3/BiasAdd/ReadVariableOp2r
7autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOp7autoencoder_3/model_7/dec_outer_3/MatMul/ReadVariableOp2r
7autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOp7autoencoder_3/model_7/dec_output/BiasAdd/ReadVariableOp2p
6autoencoder_3/model_7/dec_output/MatMul/ReadVariableOp6autoencoder_3/model_7/dec_output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
E__inference_channel_2_layer_call_and_return_conditional_losses_308945

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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_304870

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
?
?
-__inference_enc_middle_0_layer_call_fn_308754

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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_3047892
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
?	
?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_309005

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
??
?M
__inference__traced_save_309800
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
+savev2_enc_outer_2_bias_read_readvariableop1
-savev2_enc_outer_3_kernel_read_readvariableop/
+savev2_enc_outer_3_bias_read_readvariableop2
.savev2_enc_middle_0_kernel_read_readvariableop0
,savev2_enc_middle_0_bias_read_readvariableop2
.savev2_enc_middle_1_kernel_read_readvariableop0
,savev2_enc_middle_1_bias_read_readvariableop2
.savev2_enc_middle_2_kernel_read_readvariableop0
,savev2_enc_middle_2_bias_read_readvariableop2
.savev2_enc_middle_3_kernel_read_readvariableop0
,savev2_enc_middle_3_bias_read_readvariableop1
-savev2_enc_inner_0_kernel_read_readvariableop/
+savev2_enc_inner_0_bias_read_readvariableop1
-savev2_enc_inner_1_kernel_read_readvariableop/
+savev2_enc_inner_1_bias_read_readvariableop1
-savev2_enc_inner_2_kernel_read_readvariableop/
+savev2_enc_inner_2_bias_read_readvariableop1
-savev2_enc_inner_3_kernel_read_readvariableop/
+savev2_enc_inner_3_bias_read_readvariableop/
+savev2_channel_0_kernel_read_readvariableop-
)savev2_channel_0_bias_read_readvariableop/
+savev2_channel_1_kernel_read_readvariableop-
)savev2_channel_1_bias_read_readvariableop/
+savev2_channel_2_kernel_read_readvariableop-
)savev2_channel_2_bias_read_readvariableop/
+savev2_channel_3_kernel_read_readvariableop-
)savev2_channel_3_bias_read_readvariableop1
-savev2_dec_inner_0_kernel_read_readvariableop/
+savev2_dec_inner_0_bias_read_readvariableop1
-savev2_dec_inner_1_kernel_read_readvariableop/
+savev2_dec_inner_1_bias_read_readvariableop1
-savev2_dec_inner_2_kernel_read_readvariableop/
+savev2_dec_inner_2_bias_read_readvariableop1
-savev2_dec_inner_3_kernel_read_readvariableop/
+savev2_dec_inner_3_bias_read_readvariableop2
.savev2_dec_middle_0_kernel_read_readvariableop0
,savev2_dec_middle_0_bias_read_readvariableop2
.savev2_dec_middle_1_kernel_read_readvariableop0
,savev2_dec_middle_1_bias_read_readvariableop2
.savev2_dec_middle_2_kernel_read_readvariableop0
,savev2_dec_middle_2_bias_read_readvariableop2
.savev2_dec_middle_3_kernel_read_readvariableop0
,savev2_dec_middle_3_bias_read_readvariableop1
-savev2_dec_outer_0_kernel_read_readvariableop/
+savev2_dec_outer_0_bias_read_readvariableop1
-savev2_dec_outer_1_kernel_read_readvariableop/
+savev2_dec_outer_1_bias_read_readvariableop1
-savev2_dec_outer_2_kernel_read_readvariableop/
+savev2_dec_outer_2_bias_read_readvariableop1
-savev2_dec_outer_3_kernel_read_readvariableop/
+savev2_dec_outer_3_bias_read_readvariableop0
,savev2_dec_output_kernel_read_readvariableop.
*savev2_dec_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_0_bias_m_read_readvariableop8
4savev2_adam_enc_outer_1_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_1_bias_m_read_readvariableop8
4savev2_adam_enc_outer_2_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_2_bias_m_read_readvariableop8
4savev2_adam_enc_outer_3_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_3_bias_m_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_0_bias_m_read_readvariableop9
5savev2_adam_enc_middle_1_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_1_bias_m_read_readvariableop9
5savev2_adam_enc_middle_2_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_2_bias_m_read_readvariableop9
5savev2_adam_enc_middle_3_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_3_bias_m_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_0_bias_m_read_readvariableop8
4savev2_adam_enc_inner_1_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_1_bias_m_read_readvariableop8
4savev2_adam_enc_inner_2_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_2_bias_m_read_readvariableop8
4savev2_adam_enc_inner_3_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_3_bias_m_read_readvariableop6
2savev2_adam_channel_0_kernel_m_read_readvariableop4
0savev2_adam_channel_0_bias_m_read_readvariableop6
2savev2_adam_channel_1_kernel_m_read_readvariableop4
0savev2_adam_channel_1_bias_m_read_readvariableop6
2savev2_adam_channel_2_kernel_m_read_readvariableop4
0savev2_adam_channel_2_bias_m_read_readvariableop6
2savev2_adam_channel_3_kernel_m_read_readvariableop4
0savev2_adam_channel_3_bias_m_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_0_bias_m_read_readvariableop8
4savev2_adam_dec_inner_1_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_1_bias_m_read_readvariableop8
4savev2_adam_dec_inner_2_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_2_bias_m_read_readvariableop8
4savev2_adam_dec_inner_3_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_3_bias_m_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_0_bias_m_read_readvariableop9
5savev2_adam_dec_middle_1_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_1_bias_m_read_readvariableop9
5savev2_adam_dec_middle_2_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_2_bias_m_read_readvariableop9
5savev2_adam_dec_middle_3_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_3_bias_m_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_0_bias_m_read_readvariableop8
4savev2_adam_dec_outer_1_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_1_bias_m_read_readvariableop8
4savev2_adam_dec_outer_2_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_2_bias_m_read_readvariableop8
4savev2_adam_dec_outer_3_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_3_bias_m_read_readvariableop7
3savev2_adam_dec_output_kernel_m_read_readvariableop5
1savev2_adam_dec_output_bias_m_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_0_bias_v_read_readvariableop8
4savev2_adam_enc_outer_1_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_1_bias_v_read_readvariableop8
4savev2_adam_enc_outer_2_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_2_bias_v_read_readvariableop8
4savev2_adam_enc_outer_3_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_3_bias_v_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_0_bias_v_read_readvariableop9
5savev2_adam_enc_middle_1_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_1_bias_v_read_readvariableop9
5savev2_adam_enc_middle_2_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_2_bias_v_read_readvariableop9
5savev2_adam_enc_middle_3_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_3_bias_v_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_0_bias_v_read_readvariableop8
4savev2_adam_enc_inner_1_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_1_bias_v_read_readvariableop8
4savev2_adam_enc_inner_2_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_2_bias_v_read_readvariableop8
4savev2_adam_enc_inner_3_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_3_bias_v_read_readvariableop6
2savev2_adam_channel_0_kernel_v_read_readvariableop4
0savev2_adam_channel_0_bias_v_read_readvariableop6
2savev2_adam_channel_1_kernel_v_read_readvariableop4
0savev2_adam_channel_1_bias_v_read_readvariableop6
2savev2_adam_channel_2_kernel_v_read_readvariableop4
0savev2_adam_channel_2_bias_v_read_readvariableop6
2savev2_adam_channel_3_kernel_v_read_readvariableop4
0savev2_adam_channel_3_bias_v_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_0_bias_v_read_readvariableop8
4savev2_adam_dec_inner_1_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_1_bias_v_read_readvariableop8
4savev2_adam_dec_inner_2_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_2_bias_v_read_readvariableop8
4savev2_adam_dec_inner_3_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_3_bias_v_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_0_bias_v_read_readvariableop9
5savev2_adam_dec_middle_1_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_1_bias_v_read_readvariableop9
5savev2_adam_dec_middle_2_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_2_bias_v_read_readvariableop9
5savev2_adam_dec_middle_3_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_3_bias_v_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_0_bias_v_read_readvariableop8
4savev2_adam_dec_outer_1_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_1_bias_v_read_readvariableop8
4savev2_adam_dec_outer_2_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_2_bias_v_read_readvariableop8
4savev2_adam_dec_outer_3_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_3_bias_v_read_readvariableop7
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
ShardedFilename?T
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?S
value?SB?S?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/54/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/55/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/56/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/57/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/54/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/55/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/56/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/57/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?I
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop-savev2_enc_outer_0_kernel_read_readvariableop+savev2_enc_outer_0_bias_read_readvariableop-savev2_enc_outer_1_kernel_read_readvariableop+savev2_enc_outer_1_bias_read_readvariableop-savev2_enc_outer_2_kernel_read_readvariableop+savev2_enc_outer_2_bias_read_readvariableop-savev2_enc_outer_3_kernel_read_readvariableop+savev2_enc_outer_3_bias_read_readvariableop.savev2_enc_middle_0_kernel_read_readvariableop,savev2_enc_middle_0_bias_read_readvariableop.savev2_enc_middle_1_kernel_read_readvariableop,savev2_enc_middle_1_bias_read_readvariableop.savev2_enc_middle_2_kernel_read_readvariableop,savev2_enc_middle_2_bias_read_readvariableop.savev2_enc_middle_3_kernel_read_readvariableop,savev2_enc_middle_3_bias_read_readvariableop-savev2_enc_inner_0_kernel_read_readvariableop+savev2_enc_inner_0_bias_read_readvariableop-savev2_enc_inner_1_kernel_read_readvariableop+savev2_enc_inner_1_bias_read_readvariableop-savev2_enc_inner_2_kernel_read_readvariableop+savev2_enc_inner_2_bias_read_readvariableop-savev2_enc_inner_3_kernel_read_readvariableop+savev2_enc_inner_3_bias_read_readvariableop+savev2_channel_0_kernel_read_readvariableop)savev2_channel_0_bias_read_readvariableop+savev2_channel_1_kernel_read_readvariableop)savev2_channel_1_bias_read_readvariableop+savev2_channel_2_kernel_read_readvariableop)savev2_channel_2_bias_read_readvariableop+savev2_channel_3_kernel_read_readvariableop)savev2_channel_3_bias_read_readvariableop-savev2_dec_inner_0_kernel_read_readvariableop+savev2_dec_inner_0_bias_read_readvariableop-savev2_dec_inner_1_kernel_read_readvariableop+savev2_dec_inner_1_bias_read_readvariableop-savev2_dec_inner_2_kernel_read_readvariableop+savev2_dec_inner_2_bias_read_readvariableop-savev2_dec_inner_3_kernel_read_readvariableop+savev2_dec_inner_3_bias_read_readvariableop.savev2_dec_middle_0_kernel_read_readvariableop,savev2_dec_middle_0_bias_read_readvariableop.savev2_dec_middle_1_kernel_read_readvariableop,savev2_dec_middle_1_bias_read_readvariableop.savev2_dec_middle_2_kernel_read_readvariableop,savev2_dec_middle_2_bias_read_readvariableop.savev2_dec_middle_3_kernel_read_readvariableop,savev2_dec_middle_3_bias_read_readvariableop-savev2_dec_outer_0_kernel_read_readvariableop+savev2_dec_outer_0_bias_read_readvariableop-savev2_dec_outer_1_kernel_read_readvariableop+savev2_dec_outer_1_bias_read_readvariableop-savev2_dec_outer_2_kernel_read_readvariableop+savev2_dec_outer_2_bias_read_readvariableop-savev2_dec_outer_3_kernel_read_readvariableop+savev2_dec_outer_3_bias_read_readvariableop,savev2_dec_output_kernel_read_readvariableop*savev2_dec_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_enc_outer_0_kernel_m_read_readvariableop2savev2_adam_enc_outer_0_bias_m_read_readvariableop4savev2_adam_enc_outer_1_kernel_m_read_readvariableop2savev2_adam_enc_outer_1_bias_m_read_readvariableop4savev2_adam_enc_outer_2_kernel_m_read_readvariableop2savev2_adam_enc_outer_2_bias_m_read_readvariableop4savev2_adam_enc_outer_3_kernel_m_read_readvariableop2savev2_adam_enc_outer_3_bias_m_read_readvariableop5savev2_adam_enc_middle_0_kernel_m_read_readvariableop3savev2_adam_enc_middle_0_bias_m_read_readvariableop5savev2_adam_enc_middle_1_kernel_m_read_readvariableop3savev2_adam_enc_middle_1_bias_m_read_readvariableop5savev2_adam_enc_middle_2_kernel_m_read_readvariableop3savev2_adam_enc_middle_2_bias_m_read_readvariableop5savev2_adam_enc_middle_3_kernel_m_read_readvariableop3savev2_adam_enc_middle_3_bias_m_read_readvariableop4savev2_adam_enc_inner_0_kernel_m_read_readvariableop2savev2_adam_enc_inner_0_bias_m_read_readvariableop4savev2_adam_enc_inner_1_kernel_m_read_readvariableop2savev2_adam_enc_inner_1_bias_m_read_readvariableop4savev2_adam_enc_inner_2_kernel_m_read_readvariableop2savev2_adam_enc_inner_2_bias_m_read_readvariableop4savev2_adam_enc_inner_3_kernel_m_read_readvariableop2savev2_adam_enc_inner_3_bias_m_read_readvariableop2savev2_adam_channel_0_kernel_m_read_readvariableop0savev2_adam_channel_0_bias_m_read_readvariableop2savev2_adam_channel_1_kernel_m_read_readvariableop0savev2_adam_channel_1_bias_m_read_readvariableop2savev2_adam_channel_2_kernel_m_read_readvariableop0savev2_adam_channel_2_bias_m_read_readvariableop2savev2_adam_channel_3_kernel_m_read_readvariableop0savev2_adam_channel_3_bias_m_read_readvariableop4savev2_adam_dec_inner_0_kernel_m_read_readvariableop2savev2_adam_dec_inner_0_bias_m_read_readvariableop4savev2_adam_dec_inner_1_kernel_m_read_readvariableop2savev2_adam_dec_inner_1_bias_m_read_readvariableop4savev2_adam_dec_inner_2_kernel_m_read_readvariableop2savev2_adam_dec_inner_2_bias_m_read_readvariableop4savev2_adam_dec_inner_3_kernel_m_read_readvariableop2savev2_adam_dec_inner_3_bias_m_read_readvariableop5savev2_adam_dec_middle_0_kernel_m_read_readvariableop3savev2_adam_dec_middle_0_bias_m_read_readvariableop5savev2_adam_dec_middle_1_kernel_m_read_readvariableop3savev2_adam_dec_middle_1_bias_m_read_readvariableop5savev2_adam_dec_middle_2_kernel_m_read_readvariableop3savev2_adam_dec_middle_2_bias_m_read_readvariableop5savev2_adam_dec_middle_3_kernel_m_read_readvariableop3savev2_adam_dec_middle_3_bias_m_read_readvariableop4savev2_adam_dec_outer_0_kernel_m_read_readvariableop2savev2_adam_dec_outer_0_bias_m_read_readvariableop4savev2_adam_dec_outer_1_kernel_m_read_readvariableop2savev2_adam_dec_outer_1_bias_m_read_readvariableop4savev2_adam_dec_outer_2_kernel_m_read_readvariableop2savev2_adam_dec_outer_2_bias_m_read_readvariableop4savev2_adam_dec_outer_3_kernel_m_read_readvariableop2savev2_adam_dec_outer_3_bias_m_read_readvariableop3savev2_adam_dec_output_kernel_m_read_readvariableop1savev2_adam_dec_output_bias_m_read_readvariableop4savev2_adam_enc_outer_0_kernel_v_read_readvariableop2savev2_adam_enc_outer_0_bias_v_read_readvariableop4savev2_adam_enc_outer_1_kernel_v_read_readvariableop2savev2_adam_enc_outer_1_bias_v_read_readvariableop4savev2_adam_enc_outer_2_kernel_v_read_readvariableop2savev2_adam_enc_outer_2_bias_v_read_readvariableop4savev2_adam_enc_outer_3_kernel_v_read_readvariableop2savev2_adam_enc_outer_3_bias_v_read_readvariableop5savev2_adam_enc_middle_0_kernel_v_read_readvariableop3savev2_adam_enc_middle_0_bias_v_read_readvariableop5savev2_adam_enc_middle_1_kernel_v_read_readvariableop3savev2_adam_enc_middle_1_bias_v_read_readvariableop5savev2_adam_enc_middle_2_kernel_v_read_readvariableop3savev2_adam_enc_middle_2_bias_v_read_readvariableop5savev2_adam_enc_middle_3_kernel_v_read_readvariableop3savev2_adam_enc_middle_3_bias_v_read_readvariableop4savev2_adam_enc_inner_0_kernel_v_read_readvariableop2savev2_adam_enc_inner_0_bias_v_read_readvariableop4savev2_adam_enc_inner_1_kernel_v_read_readvariableop2savev2_adam_enc_inner_1_bias_v_read_readvariableop4savev2_adam_enc_inner_2_kernel_v_read_readvariableop2savev2_adam_enc_inner_2_bias_v_read_readvariableop4savev2_adam_enc_inner_3_kernel_v_read_readvariableop2savev2_adam_enc_inner_3_bias_v_read_readvariableop2savev2_adam_channel_0_kernel_v_read_readvariableop0savev2_adam_channel_0_bias_v_read_readvariableop2savev2_adam_channel_1_kernel_v_read_readvariableop0savev2_adam_channel_1_bias_v_read_readvariableop2savev2_adam_channel_2_kernel_v_read_readvariableop0savev2_adam_channel_2_bias_v_read_readvariableop2savev2_adam_channel_3_kernel_v_read_readvariableop0savev2_adam_channel_3_bias_v_read_readvariableop4savev2_adam_dec_inner_0_kernel_v_read_readvariableop2savev2_adam_dec_inner_0_bias_v_read_readvariableop4savev2_adam_dec_inner_1_kernel_v_read_readvariableop2savev2_adam_dec_inner_1_bias_v_read_readvariableop4savev2_adam_dec_inner_2_kernel_v_read_readvariableop2savev2_adam_dec_inner_2_bias_v_read_readvariableop4savev2_adam_dec_inner_3_kernel_v_read_readvariableop2savev2_adam_dec_inner_3_bias_v_read_readvariableop5savev2_adam_dec_middle_0_kernel_v_read_readvariableop3savev2_adam_dec_middle_0_bias_v_read_readvariableop5savev2_adam_dec_middle_1_kernel_v_read_readvariableop3savev2_adam_dec_middle_1_bias_v_read_readvariableop5savev2_adam_dec_middle_2_kernel_v_read_readvariableop3savev2_adam_dec_middle_2_bias_v_read_readvariableop5savev2_adam_dec_middle_3_kernel_v_read_readvariableop3savev2_adam_dec_middle_3_bias_v_read_readvariableop4savev2_adam_dec_outer_0_kernel_v_read_readvariableop2savev2_adam_dec_outer_0_bias_v_read_readvariableop4savev2_adam_dec_outer_1_kernel_v_read_readvariableop2savev2_adam_dec_outer_1_bias_v_read_readvariableop4savev2_adam_dec_outer_2_kernel_v_read_readvariableop2savev2_adam_dec_outer_2_bias_v_read_readvariableop4savev2_adam_dec_outer_3_kernel_v_read_readvariableop2savev2_adam_dec_outer_3_bias_v_read_readvariableop3savev2_adam_dec_output_kernel_v_read_readvariableop1savev2_adam_dec_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	?<:<:	?<:<:	?<:<:	?<:<:<2:2:<2:2:<2:2:<2:2:2(:(:2(:(:2(:(:2(:(:(::(::(::(::(:(:(:(:(:(:(:(:(<:<:(<:<:(<:<:(<:<:<<:<:<<:<:<<:<:<<:<:
??:?: : :	?<:<:	?<:<:	?<:<:	?<:<:<2:2:<2:2:<2:2:<2:2:2(:(:2(:(:2(:(:2(:(:(::(::(::(::(:(:(:(:(:(:(:(:(<:<:(<:<:(<:<:(<:<:<<:<:<<:<:<<:<:<<:<:
??:?:	?<:<:	?<:<:	?<:<:	?<:<:<2:2:<2:2:<2:2:<2:2:2(:(:2(:(:2(:(:2(:(:(::(::(::(::(:(:(:(:(:(:(:(:(<:<:(<:<:(<:<:(<:<:<<:<:<<:<:<<:<:<<:<:
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
:<:%!

_output_shapes
:	?<: 

_output_shapes
:<:$ 

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

:<2: 

_output_shapes
:2:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::$  

_output_shapes

:(: !

_output_shapes
::$" 

_output_shapes

:(: #

_output_shapes
::$$ 

_output_shapes

:(: %

_output_shapes
::$& 

_output_shapes

:(: '

_output_shapes
:(:$( 

_output_shapes

:(: )

_output_shapes
:(:$* 

_output_shapes

:(: +

_output_shapes
:(:$, 

_output_shapes

:(: -

_output_shapes
:(:$. 

_output_shapes

:(<: /

_output_shapes
:<:$0 

_output_shapes

:(<: 1

_output_shapes
:<:$2 

_output_shapes

:(<: 3

_output_shapes
:<:$4 

_output_shapes

:(<: 5

_output_shapes
:<:$6 

_output_shapes

:<<: 7

_output_shapes
:<:$8 

_output_shapes

:<<: 9

_output_shapes
:<:$: 

_output_shapes

:<<: ;

_output_shapes
:<:$< 

_output_shapes

:<<: =

_output_shapes
:<:&>"
 
_output_shapes
:
??:!?

_output_shapes	
:?:@

_output_shapes
: :A

_output_shapes
: :%B!

_output_shapes
:	?<: C

_output_shapes
:<:%D!

_output_shapes
:	?<: E

_output_shapes
:<:%F!

_output_shapes
:	?<: G

_output_shapes
:<:%H!

_output_shapes
:	?<: I

_output_shapes
:<:$J 

_output_shapes

:<2: K

_output_shapes
:2:$L 

_output_shapes

:<2: M

_output_shapes
:2:$N 

_output_shapes

:<2: O

_output_shapes
:2:$P 

_output_shapes

:<2: Q

_output_shapes
:2:$R 

_output_shapes

:2(: S

_output_shapes
:(:$T 

_output_shapes

:2(: U

_output_shapes
:(:$V 

_output_shapes

:2(: W

_output_shapes
:(:$X 

_output_shapes

:2(: Y

_output_shapes
:(:$Z 

_output_shapes

:(: [

_output_shapes
::$\ 

_output_shapes

:(: ]

_output_shapes
::$^ 

_output_shapes

:(: _

_output_shapes
::$` 

_output_shapes

:(: a

_output_shapes
::$b 

_output_shapes

:(: c

_output_shapes
:(:$d 

_output_shapes

:(: e

_output_shapes
:(:$f 

_output_shapes

:(: g

_output_shapes
:(:$h 

_output_shapes

:(: i

_output_shapes
:(:$j 

_output_shapes

:(<: k

_output_shapes
:<:$l 

_output_shapes

:(<: m

_output_shapes
:<:$n 

_output_shapes

:(<: o

_output_shapes
:<:$p 

_output_shapes

:(<: q

_output_shapes
:<:$r 

_output_shapes

:<<: s

_output_shapes
:<:$t 

_output_shapes

:<<: u

_output_shapes
:<:$v 

_output_shapes

:<<: w

_output_shapes
:<:$x 

_output_shapes

:<<: y

_output_shapes
:<:&z"
 
_output_shapes
:
??:!{

_output_shapes	
:?:%|!

_output_shapes
:	?<: }

_output_shapes
:<:%~!

_output_shapes
:	?<: 

_output_shapes
:<:&?!

_output_shapes
:	?<:!?

_output_shapes
:<:&?!

_output_shapes
:	?<:!?

_output_shapes
:<:%? 

_output_shapes

:<2:!?

_output_shapes
:2:%? 

_output_shapes

:<2:!?

_output_shapes
:2:%? 

_output_shapes

:<2:!?

_output_shapes
:2:%? 

_output_shapes

:<2:!?

_output_shapes
:2:%? 

_output_shapes

:2(:!?

_output_shapes
:(:%? 

_output_shapes

:2(:!?

_output_shapes
:(:%? 

_output_shapes

:2(:!?

_output_shapes
:(:%? 

_output_shapes

:2(:!?

_output_shapes
:(:%? 

_output_shapes

:(:!?

_output_shapes
::%? 

_output_shapes

:(:!?

_output_shapes
::%? 

_output_shapes

:(:!?

_output_shapes
::%? 

_output_shapes

:(:!?

_output_shapes
::%? 

_output_shapes

:(:!?

_output_shapes
:(:%? 

_output_shapes

:(:!?

_output_shapes
:(:%? 

_output_shapes

:(:!?

_output_shapes
:(:%? 

_output_shapes

:(:!?

_output_shapes
:(:%? 

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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_304843

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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_304708

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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_309185

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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_304681

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
,__inference_dec_inner_1_layer_call_fn_309014

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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_3055092
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
?

I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306662
input_1
model_6_306540
model_6_306542
model_6_306544
model_6_306546
model_6_306548
model_6_306550
model_6_306552
model_6_306554
model_6_306556
model_6_306558
model_6_306560
model_6_306562
model_6_306564
model_6_306566
model_6_306568
model_6_306570
model_6_306572
model_6_306574
model_6_306576
model_6_306578
model_6_306580
model_6_306582
model_6_306584
model_6_306586
model_6_306588
model_6_306590
model_6_306592
model_6_306594
model_6_306596
model_6_306598
model_6_306600
model_6_306602
model_7_306608
model_7_306610
model_7_306612
model_7_306614
model_7_306616
model_7_306618
model_7_306620
model_7_306622
model_7_306624
model_7_306626
model_7_306628
model_7_306630
model_7_306632
model_7_306634
model_7_306636
model_7_306638
model_7_306640
model_7_306642
model_7_306644
model_7_306646
model_7_306648
model_7_306650
model_7_306652
model_7_306654
model_7_306656
model_7_306658
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallinput_1model_6_306540model_6_306542model_6_306544model_6_306546model_6_306548model_6_306550model_6_306552model_6_306554model_6_306556model_6_306558model_6_306560model_6_306562model_6_306564model_6_306566model_6_306568model_6_306570model_6_306572model_6_306574model_6_306576model_6_306578model_6_306580model_6_306582model_6_306584model_6_306586model_6_306588model_6_306590model_6_306592model_6_306594model_6_306596model_6_306598model_6_306600model_6_306602*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3053642!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0(model_6/StatefulPartitionedCall:output:1(model_6/StatefulPartitionedCall:output:2(model_6/StatefulPartitionedCall:output:3model_7_306608model_7_306610model_7_306612model_7_306614model_7_306616model_7_306618model_7_306620model_7_306622model_7_306624model_7_306626model_7_306628model_7_306630model_7_306632model_7_306634model_7_306636model_7_306638model_7_306640model_7_306642model_7_306644model_7_306646model_7_306648model_7_306650model_7_306652model_7_306654model_7_306656model_7_306658*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3060862!
model_7/StatefulPartitionedCall?
IdentityIdentity(model_7/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
,__inference_enc_outer_1_layer_call_fn_308694

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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_3046542
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
?	
?
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_308825

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
E__inference_channel_1_layer_call_and_return_conditional_losses_308925

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
,__inference_dec_outer_0_layer_call_fn_309154

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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_3056712
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
?
?
,__inference_dec_inner_0_layer_call_fn_308994

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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_3055362
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_305617

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
-__inference_dec_middle_2_layer_call_fn_309114

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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_3055902
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
-__inference_enc_middle_2_layer_call_fn_308794

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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_3047352
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
?	
?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_305482

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
?
?
,__inference_enc_inner_3_layer_call_fn_308894

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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_3048162
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
?
?
(__inference_model_6_layer_call_fn_308334

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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30
identity

identity_1

identity_2

identity_3??StatefulPartitionedCall?
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3053642
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

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_305509

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
?
?
.__inference_autoencoder_3_layer_call_fn_307825
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

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56
identity??StatefulPartitionedCall?	
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
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_3067902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_308785

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
?
?
,__inference_enc_inner_2_layer_call_fn_308874

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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_3048432
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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_308885

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
?h
?
C__inference_model_6_layer_call_and_return_conditional_losses_305025
encoder_input
enc_outer_3_304611
enc_outer_3_304613
enc_outer_2_304638
enc_outer_2_304640
enc_outer_1_304665
enc_outer_1_304667
enc_outer_0_304692
enc_outer_0_304694
enc_middle_3_304719
enc_middle_3_304721
enc_middle_2_304746
enc_middle_2_304748
enc_middle_1_304773
enc_middle_1_304775
enc_middle_0_304800
enc_middle_0_304802
enc_inner_3_304827
enc_inner_3_304829
enc_inner_2_304854
enc_inner_2_304856
enc_inner_1_304881
enc_inner_1_304883
enc_inner_0_304908
enc_inner_0_304910
channel_3_304935
channel_3_304937
channel_2_304962
channel_2_304964
channel_1_304989
channel_1_304991
channel_0_305016
channel_0_305018
identity

identity_1

identity_2

identity_3??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?!channel_3/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?#enc_inner_3/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?$enc_middle_3/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?#enc_outer_3/StatefulPartitionedCall?
#enc_outer_3/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_3_304611enc_outer_3_304613*
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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_3046002%
#enc_outer_3/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_2_304638enc_outer_2_304640*
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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_3046272%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_1_304665enc_outer_1_304667*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_3046542%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_304692enc_outer_0_304694*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_3046812%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_3/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_3/StatefulPartitionedCall:output:0enc_middle_3_304719enc_middle_3_304721*
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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_3047082&
$enc_middle_3/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_304746enc_middle_2_304748*
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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_3047352&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_304773enc_middle_1_304775*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_3047622&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_304800enc_middle_0_304802*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_3047892&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_3/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_3/StatefulPartitionedCall:output:0enc_inner_3_304827enc_inner_3_304829*
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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_3048162%
#enc_inner_3/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_304854enc_inner_2_304856*
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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_3048432%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_304881enc_inner_1_304883*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_3048702%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_304908enc_inner_0_304910*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_3048972%
#enc_inner_0/StatefulPartitionedCall?
!channel_3/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_3/StatefulPartitionedCall:output:0channel_3_304935channel_3_304937*
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
E__inference_channel_3_layer_call_and_return_conditional_losses_3049242#
!channel_3/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_304962channel_2_304964*
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
E__inference_channel_2_layer_call_and_return_conditional_losses_3049512#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_304989channel_1_304991*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_3049782#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_305016channel_0_305018*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_3050052#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity*channel_3/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2F
!channel_3/StatefulPartitionedCall!channel_3/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2J
#enc_inner_3/StatefulPartitionedCall#enc_inner_3/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2L
$enc_middle_3/StatefulPartitionedCall$enc_middle_3/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall2J
#enc_outer_3/StatefulPartitionedCall#enc_outer_3/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?	
?
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_304816

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
?
?
-__inference_dec_middle_0_layer_call_fn_309074

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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_3056442
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
??
?.
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307495
x6
2model_6_enc_outer_3_matmul_readvariableop_resource7
3model_6_enc_outer_3_biasadd_readvariableop_resource6
2model_6_enc_outer_2_matmul_readvariableop_resource7
3model_6_enc_outer_2_biasadd_readvariableop_resource6
2model_6_enc_outer_1_matmul_readvariableop_resource7
3model_6_enc_outer_1_biasadd_readvariableop_resource6
2model_6_enc_outer_0_matmul_readvariableop_resource7
3model_6_enc_outer_0_biasadd_readvariableop_resource7
3model_6_enc_middle_3_matmul_readvariableop_resource8
4model_6_enc_middle_3_biasadd_readvariableop_resource7
3model_6_enc_middle_2_matmul_readvariableop_resource8
4model_6_enc_middle_2_biasadd_readvariableop_resource7
3model_6_enc_middle_1_matmul_readvariableop_resource8
4model_6_enc_middle_1_biasadd_readvariableop_resource7
3model_6_enc_middle_0_matmul_readvariableop_resource8
4model_6_enc_middle_0_biasadd_readvariableop_resource6
2model_6_enc_inner_3_matmul_readvariableop_resource7
3model_6_enc_inner_3_biasadd_readvariableop_resource6
2model_6_enc_inner_2_matmul_readvariableop_resource7
3model_6_enc_inner_2_biasadd_readvariableop_resource6
2model_6_enc_inner_1_matmul_readvariableop_resource7
3model_6_enc_inner_1_biasadd_readvariableop_resource6
2model_6_enc_inner_0_matmul_readvariableop_resource7
3model_6_enc_inner_0_biasadd_readvariableop_resource4
0model_6_channel_3_matmul_readvariableop_resource5
1model_6_channel_3_biasadd_readvariableop_resource4
0model_6_channel_2_matmul_readvariableop_resource5
1model_6_channel_2_biasadd_readvariableop_resource4
0model_6_channel_1_matmul_readvariableop_resource5
1model_6_channel_1_biasadd_readvariableop_resource4
0model_6_channel_0_matmul_readvariableop_resource5
1model_6_channel_0_biasadd_readvariableop_resource6
2model_7_dec_inner_3_matmul_readvariableop_resource7
3model_7_dec_inner_3_biasadd_readvariableop_resource6
2model_7_dec_inner_2_matmul_readvariableop_resource7
3model_7_dec_inner_2_biasadd_readvariableop_resource6
2model_7_dec_inner_1_matmul_readvariableop_resource7
3model_7_dec_inner_1_biasadd_readvariableop_resource6
2model_7_dec_inner_0_matmul_readvariableop_resource7
3model_7_dec_inner_0_biasadd_readvariableop_resource7
3model_7_dec_middle_3_matmul_readvariableop_resource8
4model_7_dec_middle_3_biasadd_readvariableop_resource7
3model_7_dec_middle_2_matmul_readvariableop_resource8
4model_7_dec_middle_2_biasadd_readvariableop_resource7
3model_7_dec_middle_1_matmul_readvariableop_resource8
4model_7_dec_middle_1_biasadd_readvariableop_resource7
3model_7_dec_middle_0_matmul_readvariableop_resource8
4model_7_dec_middle_0_biasadd_readvariableop_resource6
2model_7_dec_outer_0_matmul_readvariableop_resource7
3model_7_dec_outer_0_biasadd_readvariableop_resource6
2model_7_dec_outer_1_matmul_readvariableop_resource7
3model_7_dec_outer_1_biasadd_readvariableop_resource6
2model_7_dec_outer_2_matmul_readvariableop_resource7
3model_7_dec_outer_2_biasadd_readvariableop_resource6
2model_7_dec_outer_3_matmul_readvariableop_resource7
3model_7_dec_outer_3_biasadd_readvariableop_resource5
1model_7_dec_output_matmul_readvariableop_resource6
2model_7_dec_output_biasadd_readvariableop_resource
identity??(model_6/channel_0/BiasAdd/ReadVariableOp?'model_6/channel_0/MatMul/ReadVariableOp?(model_6/channel_1/BiasAdd/ReadVariableOp?'model_6/channel_1/MatMul/ReadVariableOp?(model_6/channel_2/BiasAdd/ReadVariableOp?'model_6/channel_2/MatMul/ReadVariableOp?(model_6/channel_3/BiasAdd/ReadVariableOp?'model_6/channel_3/MatMul/ReadVariableOp?*model_6/enc_inner_0/BiasAdd/ReadVariableOp?)model_6/enc_inner_0/MatMul/ReadVariableOp?*model_6/enc_inner_1/BiasAdd/ReadVariableOp?)model_6/enc_inner_1/MatMul/ReadVariableOp?*model_6/enc_inner_2/BiasAdd/ReadVariableOp?)model_6/enc_inner_2/MatMul/ReadVariableOp?*model_6/enc_inner_3/BiasAdd/ReadVariableOp?)model_6/enc_inner_3/MatMul/ReadVariableOp?+model_6/enc_middle_0/BiasAdd/ReadVariableOp?*model_6/enc_middle_0/MatMul/ReadVariableOp?+model_6/enc_middle_1/BiasAdd/ReadVariableOp?*model_6/enc_middle_1/MatMul/ReadVariableOp?+model_6/enc_middle_2/BiasAdd/ReadVariableOp?*model_6/enc_middle_2/MatMul/ReadVariableOp?+model_6/enc_middle_3/BiasAdd/ReadVariableOp?*model_6/enc_middle_3/MatMul/ReadVariableOp?*model_6/enc_outer_0/BiasAdd/ReadVariableOp?)model_6/enc_outer_0/MatMul/ReadVariableOp?*model_6/enc_outer_1/BiasAdd/ReadVariableOp?)model_6/enc_outer_1/MatMul/ReadVariableOp?*model_6/enc_outer_2/BiasAdd/ReadVariableOp?)model_6/enc_outer_2/MatMul/ReadVariableOp?*model_6/enc_outer_3/BiasAdd/ReadVariableOp?)model_6/enc_outer_3/MatMul/ReadVariableOp?*model_7/dec_inner_0/BiasAdd/ReadVariableOp?)model_7/dec_inner_0/MatMul/ReadVariableOp?*model_7/dec_inner_1/BiasAdd/ReadVariableOp?)model_7/dec_inner_1/MatMul/ReadVariableOp?*model_7/dec_inner_2/BiasAdd/ReadVariableOp?)model_7/dec_inner_2/MatMul/ReadVariableOp?*model_7/dec_inner_3/BiasAdd/ReadVariableOp?)model_7/dec_inner_3/MatMul/ReadVariableOp?+model_7/dec_middle_0/BiasAdd/ReadVariableOp?*model_7/dec_middle_0/MatMul/ReadVariableOp?+model_7/dec_middle_1/BiasAdd/ReadVariableOp?*model_7/dec_middle_1/MatMul/ReadVariableOp?+model_7/dec_middle_2/BiasAdd/ReadVariableOp?*model_7/dec_middle_2/MatMul/ReadVariableOp?+model_7/dec_middle_3/BiasAdd/ReadVariableOp?*model_7/dec_middle_3/MatMul/ReadVariableOp?*model_7/dec_outer_0/BiasAdd/ReadVariableOp?)model_7/dec_outer_0/MatMul/ReadVariableOp?*model_7/dec_outer_1/BiasAdd/ReadVariableOp?)model_7/dec_outer_1/MatMul/ReadVariableOp?*model_7/dec_outer_2/BiasAdd/ReadVariableOp?)model_7/dec_outer_2/MatMul/ReadVariableOp?*model_7/dec_outer_3/BiasAdd/ReadVariableOp?)model_7/dec_outer_3/MatMul/ReadVariableOp?)model_7/dec_output/BiasAdd/ReadVariableOp?(model_7/dec_output/MatMul/ReadVariableOp?
)model_6/enc_outer_3/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_3_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_3/MatMul/ReadVariableOp?
model_6/enc_outer_3/MatMulMatMulx1model_6/enc_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_3/MatMul?
*model_6/enc_outer_3/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_3/BiasAdd/ReadVariableOp?
model_6/enc_outer_3/BiasAddBiasAdd$model_6/enc_outer_3/MatMul:product:02model_6/enc_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_3/BiasAdd?
model_6/enc_outer_3/ReluRelu$model_6/enc_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_3/Relu?
)model_6/enc_outer_2/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_2/MatMul/ReadVariableOp?
model_6/enc_outer_2/MatMulMatMulx1model_6/enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_2/MatMul?
*model_6/enc_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_2/BiasAdd/ReadVariableOp?
model_6/enc_outer_2/BiasAddBiasAdd$model_6/enc_outer_2/MatMul:product:02model_6/enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_2/BiasAdd?
model_6/enc_outer_2/ReluRelu$model_6/enc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_2/Relu?
)model_6/enc_outer_1/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_1/MatMul/ReadVariableOp?
model_6/enc_outer_1/MatMulMatMulx1model_6/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_1/MatMul?
*model_6/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_1/BiasAdd/ReadVariableOp?
model_6/enc_outer_1/BiasAddBiasAdd$model_6/enc_outer_1/MatMul:product:02model_6/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_1/BiasAdd?
model_6/enc_outer_1/ReluRelu$model_6/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_1/Relu?
)model_6/enc_outer_0/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_0/MatMul/ReadVariableOp?
model_6/enc_outer_0/MatMulMatMulx1model_6/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_0/MatMul?
*model_6/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_0/BiasAdd/ReadVariableOp?
model_6/enc_outer_0/BiasAddBiasAdd$model_6/enc_outer_0/MatMul:product:02model_6/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_0/BiasAdd?
model_6/enc_outer_0/ReluRelu$model_6/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_0/Relu?
*model_6/enc_middle_3/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_3_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_3/MatMul/ReadVariableOp?
model_6/enc_middle_3/MatMulMatMul&model_6/enc_outer_3/Relu:activations:02model_6/enc_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_3/MatMul?
+model_6/enc_middle_3/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_3/BiasAdd/ReadVariableOp?
model_6/enc_middle_3/BiasAddBiasAdd%model_6/enc_middle_3/MatMul:product:03model_6/enc_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_3/BiasAdd?
model_6/enc_middle_3/ReluRelu%model_6/enc_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_3/Relu?
*model_6/enc_middle_2/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_2/MatMul/ReadVariableOp?
model_6/enc_middle_2/MatMulMatMul&model_6/enc_outer_2/Relu:activations:02model_6/enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_2/MatMul?
+model_6/enc_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_2/BiasAdd/ReadVariableOp?
model_6/enc_middle_2/BiasAddBiasAdd%model_6/enc_middle_2/MatMul:product:03model_6/enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_2/BiasAdd?
model_6/enc_middle_2/ReluRelu%model_6/enc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_2/Relu?
*model_6/enc_middle_1/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_1/MatMul/ReadVariableOp?
model_6/enc_middle_1/MatMulMatMul&model_6/enc_outer_1/Relu:activations:02model_6/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_1/MatMul?
+model_6/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_1/BiasAdd/ReadVariableOp?
model_6/enc_middle_1/BiasAddBiasAdd%model_6/enc_middle_1/MatMul:product:03model_6/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_1/BiasAdd?
model_6/enc_middle_1/ReluRelu%model_6/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_1/Relu?
*model_6/enc_middle_0/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_0/MatMul/ReadVariableOp?
model_6/enc_middle_0/MatMulMatMul&model_6/enc_outer_0/Relu:activations:02model_6/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_0/MatMul?
+model_6/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_0/BiasAdd/ReadVariableOp?
model_6/enc_middle_0/BiasAddBiasAdd%model_6/enc_middle_0/MatMul:product:03model_6/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_0/BiasAdd?
model_6/enc_middle_0/ReluRelu%model_6/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_0/Relu?
)model_6/enc_inner_3/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_3_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_3/MatMul/ReadVariableOp?
model_6/enc_inner_3/MatMulMatMul'model_6/enc_middle_3/Relu:activations:01model_6/enc_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_3/MatMul?
*model_6/enc_inner_3/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_3/BiasAdd/ReadVariableOp?
model_6/enc_inner_3/BiasAddBiasAdd$model_6/enc_inner_3/MatMul:product:02model_6/enc_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_3/BiasAdd?
model_6/enc_inner_3/ReluRelu$model_6/enc_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_3/Relu?
)model_6/enc_inner_2/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_2/MatMul/ReadVariableOp?
model_6/enc_inner_2/MatMulMatMul'model_6/enc_middle_2/Relu:activations:01model_6/enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_2/MatMul?
*model_6/enc_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_2/BiasAdd/ReadVariableOp?
model_6/enc_inner_2/BiasAddBiasAdd$model_6/enc_inner_2/MatMul:product:02model_6/enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_2/BiasAdd?
model_6/enc_inner_2/ReluRelu$model_6/enc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_2/Relu?
)model_6/enc_inner_1/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_1/MatMul/ReadVariableOp?
model_6/enc_inner_1/MatMulMatMul'model_6/enc_middle_1/Relu:activations:01model_6/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_1/MatMul?
*model_6/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_1/BiasAdd/ReadVariableOp?
model_6/enc_inner_1/BiasAddBiasAdd$model_6/enc_inner_1/MatMul:product:02model_6/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_1/BiasAdd?
model_6/enc_inner_1/ReluRelu$model_6/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_1/Relu?
)model_6/enc_inner_0/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_0/MatMul/ReadVariableOp?
model_6/enc_inner_0/MatMulMatMul'model_6/enc_middle_0/Relu:activations:01model_6/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_0/MatMul?
*model_6/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_0/BiasAdd/ReadVariableOp?
model_6/enc_inner_0/BiasAddBiasAdd$model_6/enc_inner_0/MatMul:product:02model_6/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_0/BiasAdd?
model_6/enc_inner_0/ReluRelu$model_6/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_0/Relu?
'model_6/channel_3/MatMul/ReadVariableOpReadVariableOp0model_6_channel_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_3/MatMul/ReadVariableOp?
model_6/channel_3/MatMulMatMul&model_6/enc_inner_3/Relu:activations:0/model_6/channel_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_3/MatMul?
(model_6/channel_3/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_3/BiasAdd/ReadVariableOp?
model_6/channel_3/BiasAddBiasAdd"model_6/channel_3/MatMul:product:00model_6/channel_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_3/BiasAdd?
model_6/channel_3/SoftsignSoftsign"model_6/channel_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_3/Softsign?
'model_6/channel_2/MatMul/ReadVariableOpReadVariableOp0model_6_channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_2/MatMul/ReadVariableOp?
model_6/channel_2/MatMulMatMul&model_6/enc_inner_2/Relu:activations:0/model_6/channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_2/MatMul?
(model_6/channel_2/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_2/BiasAdd/ReadVariableOp?
model_6/channel_2/BiasAddBiasAdd"model_6/channel_2/MatMul:product:00model_6/channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_2/BiasAdd?
model_6/channel_2/SoftsignSoftsign"model_6/channel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_2/Softsign?
'model_6/channel_1/MatMul/ReadVariableOpReadVariableOp0model_6_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_1/MatMul/ReadVariableOp?
model_6/channel_1/MatMulMatMul&model_6/enc_inner_1/Relu:activations:0/model_6/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_1/MatMul?
(model_6/channel_1/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_1/BiasAdd/ReadVariableOp?
model_6/channel_1/BiasAddBiasAdd"model_6/channel_1/MatMul:product:00model_6/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_1/BiasAdd?
model_6/channel_1/SoftsignSoftsign"model_6/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_1/Softsign?
'model_6/channel_0/MatMul/ReadVariableOpReadVariableOp0model_6_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_0/MatMul/ReadVariableOp?
model_6/channel_0/MatMulMatMul&model_6/enc_inner_0/Relu:activations:0/model_6/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_0/MatMul?
(model_6/channel_0/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_0/BiasAdd/ReadVariableOp?
model_6/channel_0/BiasAddBiasAdd"model_6/channel_0/MatMul:product:00model_6/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_0/BiasAdd?
model_6/channel_0/SoftsignSoftsign"model_6/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_0/Softsign?
)model_7/dec_inner_3/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_3/MatMul/ReadVariableOp?
model_7/dec_inner_3/MatMulMatMul(model_6/channel_3/Softsign:activations:01model_7/dec_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_3/MatMul?
*model_7/dec_inner_3/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_3/BiasAdd/ReadVariableOp?
model_7/dec_inner_3/BiasAddBiasAdd$model_7/dec_inner_3/MatMul:product:02model_7/dec_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_3/BiasAdd?
model_7/dec_inner_3/ReluRelu$model_7/dec_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_3/Relu?
)model_7/dec_inner_2/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_2/MatMul/ReadVariableOp?
model_7/dec_inner_2/MatMulMatMul(model_6/channel_2/Softsign:activations:01model_7/dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_2/MatMul?
*model_7/dec_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_2/BiasAdd/ReadVariableOp?
model_7/dec_inner_2/BiasAddBiasAdd$model_7/dec_inner_2/MatMul:product:02model_7/dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_2/BiasAdd?
model_7/dec_inner_2/ReluRelu$model_7/dec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_2/Relu?
)model_7/dec_inner_1/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_1/MatMul/ReadVariableOp?
model_7/dec_inner_1/MatMulMatMul(model_6/channel_1/Softsign:activations:01model_7/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_1/MatMul?
*model_7/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_1/BiasAdd/ReadVariableOp?
model_7/dec_inner_1/BiasAddBiasAdd$model_7/dec_inner_1/MatMul:product:02model_7/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_1/BiasAdd?
model_7/dec_inner_1/ReluRelu$model_7/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_1/Relu?
)model_7/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_0/MatMul/ReadVariableOp?
model_7/dec_inner_0/MatMulMatMul(model_6/channel_0/Softsign:activations:01model_7/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_0/MatMul?
*model_7/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_0/BiasAdd/ReadVariableOp?
model_7/dec_inner_0/BiasAddBiasAdd$model_7/dec_inner_0/MatMul:product:02model_7/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_0/BiasAdd?
model_7/dec_inner_0/ReluRelu$model_7/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_0/Relu?
*model_7/dec_middle_3/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_3_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_3/MatMul/ReadVariableOp?
model_7/dec_middle_3/MatMulMatMul&model_7/dec_inner_3/Relu:activations:02model_7/dec_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_3/MatMul?
+model_7/dec_middle_3/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_3/BiasAdd/ReadVariableOp?
model_7/dec_middle_3/BiasAddBiasAdd%model_7/dec_middle_3/MatMul:product:03model_7/dec_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_3/BiasAdd?
model_7/dec_middle_3/ReluRelu%model_7/dec_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_3/Relu?
*model_7/dec_middle_2/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_2/MatMul/ReadVariableOp?
model_7/dec_middle_2/MatMulMatMul&model_7/dec_inner_2/Relu:activations:02model_7/dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_2/MatMul?
+model_7/dec_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_2/BiasAdd/ReadVariableOp?
model_7/dec_middle_2/BiasAddBiasAdd%model_7/dec_middle_2/MatMul:product:03model_7/dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_2/BiasAdd?
model_7/dec_middle_2/ReluRelu%model_7/dec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_2/Relu?
*model_7/dec_middle_1/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_1/MatMul/ReadVariableOp?
model_7/dec_middle_1/MatMulMatMul&model_7/dec_inner_1/Relu:activations:02model_7/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_1/MatMul?
+model_7/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_1/BiasAdd/ReadVariableOp?
model_7/dec_middle_1/BiasAddBiasAdd%model_7/dec_middle_1/MatMul:product:03model_7/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_1/BiasAdd?
model_7/dec_middle_1/ReluRelu%model_7/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_1/Relu?
*model_7/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_0/MatMul/ReadVariableOp?
model_7/dec_middle_0/MatMulMatMul&model_7/dec_inner_0/Relu:activations:02model_7/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_0/MatMul?
+model_7/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_0/BiasAdd/ReadVariableOp?
model_7/dec_middle_0/BiasAddBiasAdd%model_7/dec_middle_0/MatMul:product:03model_7/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_0/BiasAdd?
model_7/dec_middle_0/ReluRelu%model_7/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_0/Relu?
)model_7/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_0/MatMul/ReadVariableOp?
model_7/dec_outer_0/MatMulMatMul'model_7/dec_middle_0/Relu:activations:01model_7/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_0/MatMul?
*model_7/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_0/BiasAdd/ReadVariableOp?
model_7/dec_outer_0/BiasAddBiasAdd$model_7/dec_outer_0/MatMul:product:02model_7/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_0/BiasAdd?
model_7/dec_outer_0/ReluRelu$model_7/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_0/Relu?
)model_7/dec_outer_1/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_1/MatMul/ReadVariableOp?
model_7/dec_outer_1/MatMulMatMul'model_7/dec_middle_1/Relu:activations:01model_7/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_1/MatMul?
*model_7/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_1/BiasAdd/ReadVariableOp?
model_7/dec_outer_1/BiasAddBiasAdd$model_7/dec_outer_1/MatMul:product:02model_7/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_1/BiasAdd?
model_7/dec_outer_1/ReluRelu$model_7/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_1/Relu?
)model_7/dec_outer_2/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_2/MatMul/ReadVariableOp?
model_7/dec_outer_2/MatMulMatMul'model_7/dec_middle_2/Relu:activations:01model_7/dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_2/MatMul?
*model_7/dec_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_2/BiasAdd/ReadVariableOp?
model_7/dec_outer_2/BiasAddBiasAdd$model_7/dec_outer_2/MatMul:product:02model_7/dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_2/BiasAdd?
model_7/dec_outer_2/ReluRelu$model_7/dec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_2/Relu?
)model_7/dec_outer_3/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_3_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_3/MatMul/ReadVariableOp?
model_7/dec_outer_3/MatMulMatMul'model_7/dec_middle_3/Relu:activations:01model_7/dec_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_3/MatMul?
*model_7/dec_outer_3/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_3/BiasAdd/ReadVariableOp?
model_7/dec_outer_3/BiasAddBiasAdd$model_7/dec_outer_3/MatMul:product:02model_7/dec_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_3/BiasAdd?
model_7/dec_outer_3/ReluRelu$model_7/dec_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_3/Relu?
model_7/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model_7/tf.concat_2/concat/axis?
model_7/tf.concat_2/concatConcatV2&model_7/dec_outer_0/Relu:activations:0&model_7/dec_outer_1/Relu:activations:0&model_7/dec_outer_2/Relu:activations:0&model_7/dec_outer_3/Relu:activations:0(model_7/tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_7/tf.concat_2/concat?
(model_7/dec_output/MatMul/ReadVariableOpReadVariableOp1model_7_dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_7/dec_output/MatMul/ReadVariableOp?
model_7/dec_output/MatMulMatMul#model_7/tf.concat_2/concat:output:00model_7/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_7/dec_output/MatMul?
)model_7/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_7_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_7/dec_output/BiasAdd/ReadVariableOp?
model_7/dec_output/BiasAddBiasAdd#model_7/dec_output/MatMul:product:01model_7/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_7/dec_output/BiasAdd?
model_7/dec_output/SigmoidSigmoid#model_7/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_7/dec_output/Sigmoid?
IdentityIdentitymodel_7/dec_output/Sigmoid:y:0)^model_6/channel_0/BiasAdd/ReadVariableOp(^model_6/channel_0/MatMul/ReadVariableOp)^model_6/channel_1/BiasAdd/ReadVariableOp(^model_6/channel_1/MatMul/ReadVariableOp)^model_6/channel_2/BiasAdd/ReadVariableOp(^model_6/channel_2/MatMul/ReadVariableOp)^model_6/channel_3/BiasAdd/ReadVariableOp(^model_6/channel_3/MatMul/ReadVariableOp+^model_6/enc_inner_0/BiasAdd/ReadVariableOp*^model_6/enc_inner_0/MatMul/ReadVariableOp+^model_6/enc_inner_1/BiasAdd/ReadVariableOp*^model_6/enc_inner_1/MatMul/ReadVariableOp+^model_6/enc_inner_2/BiasAdd/ReadVariableOp*^model_6/enc_inner_2/MatMul/ReadVariableOp+^model_6/enc_inner_3/BiasAdd/ReadVariableOp*^model_6/enc_inner_3/MatMul/ReadVariableOp,^model_6/enc_middle_0/BiasAdd/ReadVariableOp+^model_6/enc_middle_0/MatMul/ReadVariableOp,^model_6/enc_middle_1/BiasAdd/ReadVariableOp+^model_6/enc_middle_1/MatMul/ReadVariableOp,^model_6/enc_middle_2/BiasAdd/ReadVariableOp+^model_6/enc_middle_2/MatMul/ReadVariableOp,^model_6/enc_middle_3/BiasAdd/ReadVariableOp+^model_6/enc_middle_3/MatMul/ReadVariableOp+^model_6/enc_outer_0/BiasAdd/ReadVariableOp*^model_6/enc_outer_0/MatMul/ReadVariableOp+^model_6/enc_outer_1/BiasAdd/ReadVariableOp*^model_6/enc_outer_1/MatMul/ReadVariableOp+^model_6/enc_outer_2/BiasAdd/ReadVariableOp*^model_6/enc_outer_2/MatMul/ReadVariableOp+^model_6/enc_outer_3/BiasAdd/ReadVariableOp*^model_6/enc_outer_3/MatMul/ReadVariableOp+^model_7/dec_inner_0/BiasAdd/ReadVariableOp*^model_7/dec_inner_0/MatMul/ReadVariableOp+^model_7/dec_inner_1/BiasAdd/ReadVariableOp*^model_7/dec_inner_1/MatMul/ReadVariableOp+^model_7/dec_inner_2/BiasAdd/ReadVariableOp*^model_7/dec_inner_2/MatMul/ReadVariableOp+^model_7/dec_inner_3/BiasAdd/ReadVariableOp*^model_7/dec_inner_3/MatMul/ReadVariableOp,^model_7/dec_middle_0/BiasAdd/ReadVariableOp+^model_7/dec_middle_0/MatMul/ReadVariableOp,^model_7/dec_middle_1/BiasAdd/ReadVariableOp+^model_7/dec_middle_1/MatMul/ReadVariableOp,^model_7/dec_middle_2/BiasAdd/ReadVariableOp+^model_7/dec_middle_2/MatMul/ReadVariableOp,^model_7/dec_middle_3/BiasAdd/ReadVariableOp+^model_7/dec_middle_3/MatMul/ReadVariableOp+^model_7/dec_outer_0/BiasAdd/ReadVariableOp*^model_7/dec_outer_0/MatMul/ReadVariableOp+^model_7/dec_outer_1/BiasAdd/ReadVariableOp*^model_7/dec_outer_1/MatMul/ReadVariableOp+^model_7/dec_outer_2/BiasAdd/ReadVariableOp*^model_7/dec_outer_2/MatMul/ReadVariableOp+^model_7/dec_outer_3/BiasAdd/ReadVariableOp*^model_7/dec_outer_3/MatMul/ReadVariableOp*^model_7/dec_output/BiasAdd/ReadVariableOp)^model_7/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2T
(model_6/channel_0/BiasAdd/ReadVariableOp(model_6/channel_0/BiasAdd/ReadVariableOp2R
'model_6/channel_0/MatMul/ReadVariableOp'model_6/channel_0/MatMul/ReadVariableOp2T
(model_6/channel_1/BiasAdd/ReadVariableOp(model_6/channel_1/BiasAdd/ReadVariableOp2R
'model_6/channel_1/MatMul/ReadVariableOp'model_6/channel_1/MatMul/ReadVariableOp2T
(model_6/channel_2/BiasAdd/ReadVariableOp(model_6/channel_2/BiasAdd/ReadVariableOp2R
'model_6/channel_2/MatMul/ReadVariableOp'model_6/channel_2/MatMul/ReadVariableOp2T
(model_6/channel_3/BiasAdd/ReadVariableOp(model_6/channel_3/BiasAdd/ReadVariableOp2R
'model_6/channel_3/MatMul/ReadVariableOp'model_6/channel_3/MatMul/ReadVariableOp2X
*model_6/enc_inner_0/BiasAdd/ReadVariableOp*model_6/enc_inner_0/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_0/MatMul/ReadVariableOp)model_6/enc_inner_0/MatMul/ReadVariableOp2X
*model_6/enc_inner_1/BiasAdd/ReadVariableOp*model_6/enc_inner_1/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_1/MatMul/ReadVariableOp)model_6/enc_inner_1/MatMul/ReadVariableOp2X
*model_6/enc_inner_2/BiasAdd/ReadVariableOp*model_6/enc_inner_2/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_2/MatMul/ReadVariableOp)model_6/enc_inner_2/MatMul/ReadVariableOp2X
*model_6/enc_inner_3/BiasAdd/ReadVariableOp*model_6/enc_inner_3/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_3/MatMul/ReadVariableOp)model_6/enc_inner_3/MatMul/ReadVariableOp2Z
+model_6/enc_middle_0/BiasAdd/ReadVariableOp+model_6/enc_middle_0/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_0/MatMul/ReadVariableOp*model_6/enc_middle_0/MatMul/ReadVariableOp2Z
+model_6/enc_middle_1/BiasAdd/ReadVariableOp+model_6/enc_middle_1/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_1/MatMul/ReadVariableOp*model_6/enc_middle_1/MatMul/ReadVariableOp2Z
+model_6/enc_middle_2/BiasAdd/ReadVariableOp+model_6/enc_middle_2/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_2/MatMul/ReadVariableOp*model_6/enc_middle_2/MatMul/ReadVariableOp2Z
+model_6/enc_middle_3/BiasAdd/ReadVariableOp+model_6/enc_middle_3/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_3/MatMul/ReadVariableOp*model_6/enc_middle_3/MatMul/ReadVariableOp2X
*model_6/enc_outer_0/BiasAdd/ReadVariableOp*model_6/enc_outer_0/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_0/MatMul/ReadVariableOp)model_6/enc_outer_0/MatMul/ReadVariableOp2X
*model_6/enc_outer_1/BiasAdd/ReadVariableOp*model_6/enc_outer_1/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_1/MatMul/ReadVariableOp)model_6/enc_outer_1/MatMul/ReadVariableOp2X
*model_6/enc_outer_2/BiasAdd/ReadVariableOp*model_6/enc_outer_2/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_2/MatMul/ReadVariableOp)model_6/enc_outer_2/MatMul/ReadVariableOp2X
*model_6/enc_outer_3/BiasAdd/ReadVariableOp*model_6/enc_outer_3/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_3/MatMul/ReadVariableOp)model_6/enc_outer_3/MatMul/ReadVariableOp2X
*model_7/dec_inner_0/BiasAdd/ReadVariableOp*model_7/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_0/MatMul/ReadVariableOp)model_7/dec_inner_0/MatMul/ReadVariableOp2X
*model_7/dec_inner_1/BiasAdd/ReadVariableOp*model_7/dec_inner_1/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_1/MatMul/ReadVariableOp)model_7/dec_inner_1/MatMul/ReadVariableOp2X
*model_7/dec_inner_2/BiasAdd/ReadVariableOp*model_7/dec_inner_2/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_2/MatMul/ReadVariableOp)model_7/dec_inner_2/MatMul/ReadVariableOp2X
*model_7/dec_inner_3/BiasAdd/ReadVariableOp*model_7/dec_inner_3/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_3/MatMul/ReadVariableOp)model_7/dec_inner_3/MatMul/ReadVariableOp2Z
+model_7/dec_middle_0/BiasAdd/ReadVariableOp+model_7/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_0/MatMul/ReadVariableOp*model_7/dec_middle_0/MatMul/ReadVariableOp2Z
+model_7/dec_middle_1/BiasAdd/ReadVariableOp+model_7/dec_middle_1/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_1/MatMul/ReadVariableOp*model_7/dec_middle_1/MatMul/ReadVariableOp2Z
+model_7/dec_middle_2/BiasAdd/ReadVariableOp+model_7/dec_middle_2/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_2/MatMul/ReadVariableOp*model_7/dec_middle_2/MatMul/ReadVariableOp2Z
+model_7/dec_middle_3/BiasAdd/ReadVariableOp+model_7/dec_middle_3/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_3/MatMul/ReadVariableOp*model_7/dec_middle_3/MatMul/ReadVariableOp2X
*model_7/dec_outer_0/BiasAdd/ReadVariableOp*model_7/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_0/MatMul/ReadVariableOp)model_7/dec_outer_0/MatMul/ReadVariableOp2X
*model_7/dec_outer_1/BiasAdd/ReadVariableOp*model_7/dec_outer_1/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_1/MatMul/ReadVariableOp)model_7/dec_outer_1/MatMul/ReadVariableOp2X
*model_7/dec_outer_2/BiasAdd/ReadVariableOp*model_7/dec_outer_2/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_2/MatMul/ReadVariableOp)model_7/dec_outer_2/MatMul/ReadVariableOp2X
*model_7/dec_outer_3/BiasAdd/ReadVariableOp*model_7/dec_outer_3/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_3/MatMul/ReadVariableOp)model_7/dec_outer_3/MatMul/ReadVariableOp2V
)model_7/dec_output/BiasAdd/ReadVariableOp)model_7/dec_output/BiasAdd/ReadVariableOp2T
(model_7/dec_output/MatMul/ReadVariableOp(model_7/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_309125

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
,__inference_enc_outer_0_layer_call_fn_308674

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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_3046812
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
?	
?
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_305725

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
?
?
(__inference_model_6_layer_call_fn_305275
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30
identity

identity_1

identity_2

identity_3??StatefulPartitionedCall?
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3052022
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

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?M
?

C__inference_model_7_layer_call_and_return_conditional_losses_305872
decoder_input_0
decoder_input_1
decoder_input_2
decoder_input_3
dec_inner_3_305804
dec_inner_3_305806
dec_inner_2_305809
dec_inner_2_305811
dec_inner_1_305814
dec_inner_1_305816
dec_inner_0_305819
dec_inner_0_305821
dec_middle_3_305824
dec_middle_3_305826
dec_middle_2_305829
dec_middle_2_305831
dec_middle_1_305834
dec_middle_1_305836
dec_middle_0_305839
dec_middle_0_305841
dec_outer_0_305844
dec_outer_0_305846
dec_outer_1_305849
dec_outer_1_305851
dec_outer_2_305854
dec_outer_2_305856
dec_outer_3_305859
dec_outer_3_305861
dec_output_305866
dec_output_305868
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?#dec_inner_3/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?$dec_middle_3/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?#dec_outer_3/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_3/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_3dec_inner_3_305804dec_inner_3_305806*
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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_3054552%
#dec_inner_3/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_2dec_inner_2_305809dec_inner_2_305811*
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
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_3054822%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_1dec_inner_1_305814dec_inner_1_305816*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_3055092%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_305819dec_inner_0_305821*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_3055362%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_3/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_3/StatefulPartitionedCall:output:0dec_middle_3_305824dec_middle_3_305826*
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
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_3055632&
$dec_middle_3/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_305829dec_middle_2_305831*
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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_3055902&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_305834dec_middle_1_305836*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_3056172&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_305839dec_middle_0_305841*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_3056442&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_305844dec_outer_0_305846*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_3056712%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_305849dec_outer_1_305851*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_3056982%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_305854dec_outer_2_305856*
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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_3057252%
#dec_outer_2/StatefulPartitionedCall?
#dec_outer_3/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_3/StatefulPartitionedCall:output:0dec_outer_3_305859dec_outer_3_305861*
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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_3057522%
#dec_outer_3/StatefulPartitionedCallt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0,dec_outer_3/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dec_output_305866dec_output_305868*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_3057812$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall$^dec_inner_3/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall%^dec_middle_3/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall$^dec_outer_3/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2J
#dec_inner_3/StatefulPartitionedCall#dec_inner_3/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2L
$dec_middle_3/StatefulPartitionedCall$dec_middle_3/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2J
#dec_outer_3/StatefulPartitionedCall#dec_outer_3/StatefulPartitionedCall2H
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
_user_specified_namedecoder_input_2:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_3
?M
?

C__inference_model_7_layer_call_and_return_conditional_losses_305798
decoder_input_0
decoder_input_1
decoder_input_2
decoder_input_3
dec_inner_3_305466
dec_inner_3_305468
dec_inner_2_305493
dec_inner_2_305495
dec_inner_1_305520
dec_inner_1_305522
dec_inner_0_305547
dec_inner_0_305549
dec_middle_3_305574
dec_middle_3_305576
dec_middle_2_305601
dec_middle_2_305603
dec_middle_1_305628
dec_middle_1_305630
dec_middle_0_305655
dec_middle_0_305657
dec_outer_0_305682
dec_outer_0_305684
dec_outer_1_305709
dec_outer_1_305711
dec_outer_2_305736
dec_outer_2_305738
dec_outer_3_305763
dec_outer_3_305765
dec_output_305792
dec_output_305794
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?#dec_inner_3/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?$dec_middle_3/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?#dec_outer_3/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_3/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_3dec_inner_3_305466dec_inner_3_305468*
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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_3054552%
#dec_inner_3/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_2dec_inner_2_305493dec_inner_2_305495*
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
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_3054822%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_1dec_inner_1_305520dec_inner_1_305522*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_3055092%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_305547dec_inner_0_305549*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_3055362%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_3/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_3/StatefulPartitionedCall:output:0dec_middle_3_305574dec_middle_3_305576*
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
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_3055632&
$dec_middle_3/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_305601dec_middle_2_305603*
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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_3055902&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_305628dec_middle_1_305630*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_3056172&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_305655dec_middle_0_305657*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_3056442&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_305682dec_outer_0_305684*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_3056712%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_305709dec_outer_1_305711*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_3056982%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_305736dec_outer_2_305738*
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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_3057252%
#dec_outer_2/StatefulPartitionedCall?
#dec_outer_3/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_3/StatefulPartitionedCall:output:0dec_outer_3_305763dec_outer_3_305765*
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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_3057522%
#dec_outer_3/StatefulPartitionedCallt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0,dec_outer_3/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dec_output_305792dec_output_305794*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_3057812$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall$^dec_inner_3/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall%^dec_middle_3/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall$^dec_outer_3/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2J
#dec_inner_3/StatefulPartitionedCall#dec_inner_3/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2L
$dec_middle_3/StatefulPartitionedCall$dec_middle_3/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2J
#dec_outer_3/StatefulPartitionedCall#dec_outer_3/StatefulPartitionedCall2H
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
_user_specified_namedecoder_input_2:XT
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_3
?

*__inference_channel_3_layer_call_fn_308974

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
E__inference_channel_3_layer_call_and_return_conditional_losses_3049242
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
?
?
-__inference_dec_middle_1_layer_call_fn_309094

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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_3056172
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
?h
?
C__inference_model_6_layer_call_and_return_conditional_losses_305364

inputs
enc_outer_3_305280
enc_outer_3_305282
enc_outer_2_305285
enc_outer_2_305287
enc_outer_1_305290
enc_outer_1_305292
enc_outer_0_305295
enc_outer_0_305297
enc_middle_3_305300
enc_middle_3_305302
enc_middle_2_305305
enc_middle_2_305307
enc_middle_1_305310
enc_middle_1_305312
enc_middle_0_305315
enc_middle_0_305317
enc_inner_3_305320
enc_inner_3_305322
enc_inner_2_305325
enc_inner_2_305327
enc_inner_1_305330
enc_inner_1_305332
enc_inner_0_305335
enc_inner_0_305337
channel_3_305340
channel_3_305342
channel_2_305345
channel_2_305347
channel_1_305350
channel_1_305352
channel_0_305355
channel_0_305357
identity

identity_1

identity_2

identity_3??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?!channel_3/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?#enc_inner_3/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?$enc_middle_3/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?#enc_outer_3/StatefulPartitionedCall?
#enc_outer_3/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_3_305280enc_outer_3_305282*
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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_3046002%
#enc_outer_3/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_2_305285enc_outer_2_305287*
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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_3046272%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_1_305290enc_outer_1_305292*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_3046542%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_305295enc_outer_0_305297*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_3046812%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_3/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_3/StatefulPartitionedCall:output:0enc_middle_3_305300enc_middle_3_305302*
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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_3047082&
$enc_middle_3/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_305305enc_middle_2_305307*
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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_3047352&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_305310enc_middle_1_305312*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_3047622&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_305315enc_middle_0_305317*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_3047892&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_3/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_3/StatefulPartitionedCall:output:0enc_inner_3_305320enc_inner_3_305322*
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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_3048162%
#enc_inner_3/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_305325enc_inner_2_305327*
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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_3048432%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_305330enc_inner_1_305332*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_3048702%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_305335enc_inner_0_305337*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_3048972%
#enc_inner_0/StatefulPartitionedCall?
!channel_3/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_3/StatefulPartitionedCall:output:0channel_3_305340channel_3_305342*
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
E__inference_channel_3_layer_call_and_return_conditional_losses_3049242#
!channel_3/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_305345channel_2_305347*
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
E__inference_channel_2_layer_call_and_return_conditional_losses_3049512#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_305350channel_1_305352*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_3049782#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_305355channel_0_305357*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_3050052#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity*channel_3/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2F
!channel_3/StatefulPartitionedCall!channel_3/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2J
#enc_inner_3/StatefulPartitionedCall#enc_inner_3/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2L
$enc_middle_3/StatefulPartitionedCall$enc_middle_3/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall2J
#enc_outer_3/StatefulPartitionedCall#enc_outer_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_304762

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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_304627

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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_305752

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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_304897

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
?
?
(__inference_model_6_layer_call_fn_308259

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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30
identity

identity_1

identity_2

identity_3??StatefulPartitionedCall?
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3052022
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

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_307286
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

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56
identity??StatefulPartitionedCall?
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
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_3045852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_308765

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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_309145

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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_309205

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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_304735

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
؆
?
C__inference_model_7_layer_call_and_return_conditional_losses_308534
inputs_0
inputs_1
inputs_2
inputs_3.
*dec_inner_3_matmul_readvariableop_resource/
+dec_inner_3_biasadd_readvariableop_resource.
*dec_inner_2_matmul_readvariableop_resource/
+dec_inner_2_biasadd_readvariableop_resource.
*dec_inner_1_matmul_readvariableop_resource/
+dec_inner_1_biasadd_readvariableop_resource.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_3_matmul_readvariableop_resource0
,dec_middle_3_biasadd_readvariableop_resource/
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
+dec_outer_2_biasadd_readvariableop_resource.
*dec_outer_3_matmul_readvariableop_resource/
+dec_outer_3_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?"dec_inner_1/BiasAdd/ReadVariableOp?!dec_inner_1/MatMul/ReadVariableOp?"dec_inner_2/BiasAdd/ReadVariableOp?!dec_inner_2/MatMul/ReadVariableOp?"dec_inner_3/BiasAdd/ReadVariableOp?!dec_inner_3/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?#dec_middle_1/BiasAdd/ReadVariableOp?"dec_middle_1/MatMul/ReadVariableOp?#dec_middle_2/BiasAdd/ReadVariableOp?"dec_middle_2/MatMul/ReadVariableOp?#dec_middle_3/BiasAdd/ReadVariableOp?"dec_middle_3/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?"dec_outer_1/BiasAdd/ReadVariableOp?!dec_outer_1/MatMul/ReadVariableOp?"dec_outer_2/BiasAdd/ReadVariableOp?!dec_outer_2/MatMul/ReadVariableOp?"dec_outer_3/BiasAdd/ReadVariableOp?!dec_outer_3/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
!dec_inner_3/MatMul/ReadVariableOpReadVariableOp*dec_inner_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_3/MatMul/ReadVariableOp?
dec_inner_3/MatMulMatMulinputs_3)dec_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_3/MatMul?
"dec_inner_3/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_3/BiasAdd/ReadVariableOp?
dec_inner_3/BiasAddBiasAdddec_inner_3/MatMul:product:0*dec_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_3/BiasAdd|
dec_inner_3/ReluReludec_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_3/Relu?
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
"dec_middle_3/MatMul/ReadVariableOpReadVariableOp+dec_middle_3_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_3/MatMul/ReadVariableOp?
dec_middle_3/MatMulMatMuldec_inner_3/Relu:activations:0*dec_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_3/MatMul?
#dec_middle_3/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_3/BiasAdd/ReadVariableOp?
dec_middle_3/BiasAddBiasAdddec_middle_3/MatMul:product:0+dec_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_3/BiasAdd
dec_middle_3/ReluReludec_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_3/Relu?
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
dec_outer_2/Relu?
!dec_outer_3/MatMul/ReadVariableOpReadVariableOp*dec_outer_3_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_3/MatMul/ReadVariableOp?
dec_outer_3/MatMulMatMuldec_middle_3/Relu:activations:0)dec_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_3/MatMul?
"dec_outer_3/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_3/BiasAdd/ReadVariableOp?
dec_outer_3/BiasAddBiasAdddec_outer_3/MatMul:product:0*dec_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_3/BiasAdd|
dec_outer_3/ReluReludec_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_3/Relut
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2dec_outer_0/Relu:activations:0dec_outer_1/Relu:activations:0dec_outer_2/Relu:activations:0dec_outer_3/Relu:activations:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.concat_2/concat:output:0(dec_output/MatMul/ReadVariableOp:value:0*
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
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp#^dec_inner_1/BiasAdd/ReadVariableOp"^dec_inner_1/MatMul/ReadVariableOp#^dec_inner_2/BiasAdd/ReadVariableOp"^dec_inner_2/MatMul/ReadVariableOp#^dec_inner_3/BiasAdd/ReadVariableOp"^dec_inner_3/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp$^dec_middle_1/BiasAdd/ReadVariableOp#^dec_middle_1/MatMul/ReadVariableOp$^dec_middle_2/BiasAdd/ReadVariableOp#^dec_middle_2/MatMul/ReadVariableOp$^dec_middle_3/BiasAdd/ReadVariableOp#^dec_middle_3/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp#^dec_outer_1/BiasAdd/ReadVariableOp"^dec_outer_1/MatMul/ReadVariableOp#^dec_outer_2/BiasAdd/ReadVariableOp"^dec_outer_2/MatMul/ReadVariableOp#^dec_outer_3/BiasAdd/ReadVariableOp"^dec_outer_3/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2H
"dec_inner_1/BiasAdd/ReadVariableOp"dec_inner_1/BiasAdd/ReadVariableOp2F
!dec_inner_1/MatMul/ReadVariableOp!dec_inner_1/MatMul/ReadVariableOp2H
"dec_inner_2/BiasAdd/ReadVariableOp"dec_inner_2/BiasAdd/ReadVariableOp2F
!dec_inner_2/MatMul/ReadVariableOp!dec_inner_2/MatMul/ReadVariableOp2H
"dec_inner_3/BiasAdd/ReadVariableOp"dec_inner_3/BiasAdd/ReadVariableOp2F
!dec_inner_3/MatMul/ReadVariableOp!dec_inner_3/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2J
#dec_middle_1/BiasAdd/ReadVariableOp#dec_middle_1/BiasAdd/ReadVariableOp2H
"dec_middle_1/MatMul/ReadVariableOp"dec_middle_1/MatMul/ReadVariableOp2J
#dec_middle_2/BiasAdd/ReadVariableOp#dec_middle_2/BiasAdd/ReadVariableOp2H
"dec_middle_2/MatMul/ReadVariableOp"dec_middle_2/MatMul/ReadVariableOp2J
#dec_middle_3/BiasAdd/ReadVariableOp#dec_middle_3/BiasAdd/ReadVariableOp2H
"dec_middle_3/MatMul/ReadVariableOp"dec_middle_3/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2H
"dec_outer_1/BiasAdd/ReadVariableOp"dec_outer_1/BiasAdd/ReadVariableOp2F
!dec_outer_1/MatMul/ReadVariableOp!dec_outer_1/MatMul/ReadVariableOp2H
"dec_outer_2/BiasAdd/ReadVariableOp"dec_outer_2/BiasAdd/ReadVariableOp2F
!dec_outer_2/MatMul/ReadVariableOp!dec_outer_2/MatMul/ReadVariableOp2H
"dec_outer_3/BiasAdd/ReadVariableOp"dec_outer_3/BiasAdd/ReadVariableOp2F
!dec_outer_3/MatMul/ReadVariableOp!dec_outer_3/MatMul/ReadVariableOp2F
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
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?
?

I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306537
input_1
model_6_306295
model_6_306297
model_6_306299
model_6_306301
model_6_306303
model_6_306305
model_6_306307
model_6_306309
model_6_306311
model_6_306313
model_6_306315
model_6_306317
model_6_306319
model_6_306321
model_6_306323
model_6_306325
model_6_306327
model_6_306329
model_6_306331
model_6_306333
model_6_306335
model_6_306337
model_6_306339
model_6_306341
model_6_306343
model_6_306345
model_6_306347
model_6_306349
model_6_306351
model_6_306353
model_6_306355
model_6_306357
model_7_306483
model_7_306485
model_7_306487
model_7_306489
model_7_306491
model_7_306493
model_7_306495
model_7_306497
model_7_306499
model_7_306501
model_7_306503
model_7_306505
model_7_306507
model_7_306509
model_7_306511
model_7_306513
model_7_306515
model_7_306517
model_7_306519
model_7_306521
model_7_306523
model_7_306525
model_7_306527
model_7_306529
model_7_306531
model_7_306533
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallinput_1model_6_306295model_6_306297model_6_306299model_6_306301model_6_306303model_6_306305model_6_306307model_6_306309model_6_306311model_6_306313model_6_306315model_6_306317model_6_306319model_6_306321model_6_306323model_6_306325model_6_306327model_6_306329model_6_306331model_6_306333model_6_306335model_6_306337model_6_306339model_6_306341model_6_306343model_6_306345model_6_306347model_6_306349model_6_306351model_6_306353model_6_306355model_6_306357*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3052022!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0(model_6/StatefulPartitionedCall:output:1(model_6/StatefulPartitionedCall:output:2(model_6/StatefulPartitionedCall:output:3model_7_306483model_7_306485model_7_306487model_7_306489model_7_306491model_7_306493model_7_306495model_7_306497model_7_306499model_7_306501model_7_306503model_7_306505model_7_306507model_7_306509model_7_306511model_7_306513model_7_306515model_7_306517model_7_306519model_7_306521model_7_306523model_7_306525model_7_306527model_7_306529model_7_306531model_7_306533*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3059522!
model_7/StatefulPartitionedCall?
IdentityIdentity(model_7/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_308985

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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_305698

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
??
?.
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307704
x6
2model_6_enc_outer_3_matmul_readvariableop_resource7
3model_6_enc_outer_3_biasadd_readvariableop_resource6
2model_6_enc_outer_2_matmul_readvariableop_resource7
3model_6_enc_outer_2_biasadd_readvariableop_resource6
2model_6_enc_outer_1_matmul_readvariableop_resource7
3model_6_enc_outer_1_biasadd_readvariableop_resource6
2model_6_enc_outer_0_matmul_readvariableop_resource7
3model_6_enc_outer_0_biasadd_readvariableop_resource7
3model_6_enc_middle_3_matmul_readvariableop_resource8
4model_6_enc_middle_3_biasadd_readvariableop_resource7
3model_6_enc_middle_2_matmul_readvariableop_resource8
4model_6_enc_middle_2_biasadd_readvariableop_resource7
3model_6_enc_middle_1_matmul_readvariableop_resource8
4model_6_enc_middle_1_biasadd_readvariableop_resource7
3model_6_enc_middle_0_matmul_readvariableop_resource8
4model_6_enc_middle_0_biasadd_readvariableop_resource6
2model_6_enc_inner_3_matmul_readvariableop_resource7
3model_6_enc_inner_3_biasadd_readvariableop_resource6
2model_6_enc_inner_2_matmul_readvariableop_resource7
3model_6_enc_inner_2_biasadd_readvariableop_resource6
2model_6_enc_inner_1_matmul_readvariableop_resource7
3model_6_enc_inner_1_biasadd_readvariableop_resource6
2model_6_enc_inner_0_matmul_readvariableop_resource7
3model_6_enc_inner_0_biasadd_readvariableop_resource4
0model_6_channel_3_matmul_readvariableop_resource5
1model_6_channel_3_biasadd_readvariableop_resource4
0model_6_channel_2_matmul_readvariableop_resource5
1model_6_channel_2_biasadd_readvariableop_resource4
0model_6_channel_1_matmul_readvariableop_resource5
1model_6_channel_1_biasadd_readvariableop_resource4
0model_6_channel_0_matmul_readvariableop_resource5
1model_6_channel_0_biasadd_readvariableop_resource6
2model_7_dec_inner_3_matmul_readvariableop_resource7
3model_7_dec_inner_3_biasadd_readvariableop_resource6
2model_7_dec_inner_2_matmul_readvariableop_resource7
3model_7_dec_inner_2_biasadd_readvariableop_resource6
2model_7_dec_inner_1_matmul_readvariableop_resource7
3model_7_dec_inner_1_biasadd_readvariableop_resource6
2model_7_dec_inner_0_matmul_readvariableop_resource7
3model_7_dec_inner_0_biasadd_readvariableop_resource7
3model_7_dec_middle_3_matmul_readvariableop_resource8
4model_7_dec_middle_3_biasadd_readvariableop_resource7
3model_7_dec_middle_2_matmul_readvariableop_resource8
4model_7_dec_middle_2_biasadd_readvariableop_resource7
3model_7_dec_middle_1_matmul_readvariableop_resource8
4model_7_dec_middle_1_biasadd_readvariableop_resource7
3model_7_dec_middle_0_matmul_readvariableop_resource8
4model_7_dec_middle_0_biasadd_readvariableop_resource6
2model_7_dec_outer_0_matmul_readvariableop_resource7
3model_7_dec_outer_0_biasadd_readvariableop_resource6
2model_7_dec_outer_1_matmul_readvariableop_resource7
3model_7_dec_outer_1_biasadd_readvariableop_resource6
2model_7_dec_outer_2_matmul_readvariableop_resource7
3model_7_dec_outer_2_biasadd_readvariableop_resource6
2model_7_dec_outer_3_matmul_readvariableop_resource7
3model_7_dec_outer_3_biasadd_readvariableop_resource5
1model_7_dec_output_matmul_readvariableop_resource6
2model_7_dec_output_biasadd_readvariableop_resource
identity??(model_6/channel_0/BiasAdd/ReadVariableOp?'model_6/channel_0/MatMul/ReadVariableOp?(model_6/channel_1/BiasAdd/ReadVariableOp?'model_6/channel_1/MatMul/ReadVariableOp?(model_6/channel_2/BiasAdd/ReadVariableOp?'model_6/channel_2/MatMul/ReadVariableOp?(model_6/channel_3/BiasAdd/ReadVariableOp?'model_6/channel_3/MatMul/ReadVariableOp?*model_6/enc_inner_0/BiasAdd/ReadVariableOp?)model_6/enc_inner_0/MatMul/ReadVariableOp?*model_6/enc_inner_1/BiasAdd/ReadVariableOp?)model_6/enc_inner_1/MatMul/ReadVariableOp?*model_6/enc_inner_2/BiasAdd/ReadVariableOp?)model_6/enc_inner_2/MatMul/ReadVariableOp?*model_6/enc_inner_3/BiasAdd/ReadVariableOp?)model_6/enc_inner_3/MatMul/ReadVariableOp?+model_6/enc_middle_0/BiasAdd/ReadVariableOp?*model_6/enc_middle_0/MatMul/ReadVariableOp?+model_6/enc_middle_1/BiasAdd/ReadVariableOp?*model_6/enc_middle_1/MatMul/ReadVariableOp?+model_6/enc_middle_2/BiasAdd/ReadVariableOp?*model_6/enc_middle_2/MatMul/ReadVariableOp?+model_6/enc_middle_3/BiasAdd/ReadVariableOp?*model_6/enc_middle_3/MatMul/ReadVariableOp?*model_6/enc_outer_0/BiasAdd/ReadVariableOp?)model_6/enc_outer_0/MatMul/ReadVariableOp?*model_6/enc_outer_1/BiasAdd/ReadVariableOp?)model_6/enc_outer_1/MatMul/ReadVariableOp?*model_6/enc_outer_2/BiasAdd/ReadVariableOp?)model_6/enc_outer_2/MatMul/ReadVariableOp?*model_6/enc_outer_3/BiasAdd/ReadVariableOp?)model_6/enc_outer_3/MatMul/ReadVariableOp?*model_7/dec_inner_0/BiasAdd/ReadVariableOp?)model_7/dec_inner_0/MatMul/ReadVariableOp?*model_7/dec_inner_1/BiasAdd/ReadVariableOp?)model_7/dec_inner_1/MatMul/ReadVariableOp?*model_7/dec_inner_2/BiasAdd/ReadVariableOp?)model_7/dec_inner_2/MatMul/ReadVariableOp?*model_7/dec_inner_3/BiasAdd/ReadVariableOp?)model_7/dec_inner_3/MatMul/ReadVariableOp?+model_7/dec_middle_0/BiasAdd/ReadVariableOp?*model_7/dec_middle_0/MatMul/ReadVariableOp?+model_7/dec_middle_1/BiasAdd/ReadVariableOp?*model_7/dec_middle_1/MatMul/ReadVariableOp?+model_7/dec_middle_2/BiasAdd/ReadVariableOp?*model_7/dec_middle_2/MatMul/ReadVariableOp?+model_7/dec_middle_3/BiasAdd/ReadVariableOp?*model_7/dec_middle_3/MatMul/ReadVariableOp?*model_7/dec_outer_0/BiasAdd/ReadVariableOp?)model_7/dec_outer_0/MatMul/ReadVariableOp?*model_7/dec_outer_1/BiasAdd/ReadVariableOp?)model_7/dec_outer_1/MatMul/ReadVariableOp?*model_7/dec_outer_2/BiasAdd/ReadVariableOp?)model_7/dec_outer_2/MatMul/ReadVariableOp?*model_7/dec_outer_3/BiasAdd/ReadVariableOp?)model_7/dec_outer_3/MatMul/ReadVariableOp?)model_7/dec_output/BiasAdd/ReadVariableOp?(model_7/dec_output/MatMul/ReadVariableOp?
)model_6/enc_outer_3/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_3_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_3/MatMul/ReadVariableOp?
model_6/enc_outer_3/MatMulMatMulx1model_6/enc_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_3/MatMul?
*model_6/enc_outer_3/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_3/BiasAdd/ReadVariableOp?
model_6/enc_outer_3/BiasAddBiasAdd$model_6/enc_outer_3/MatMul:product:02model_6/enc_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_3/BiasAdd?
model_6/enc_outer_3/ReluRelu$model_6/enc_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_3/Relu?
)model_6/enc_outer_2/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_2_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_2/MatMul/ReadVariableOp?
model_6/enc_outer_2/MatMulMatMulx1model_6/enc_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_2/MatMul?
*model_6/enc_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_2/BiasAdd/ReadVariableOp?
model_6/enc_outer_2/BiasAddBiasAdd$model_6/enc_outer_2/MatMul:product:02model_6/enc_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_2/BiasAdd?
model_6/enc_outer_2/ReluRelu$model_6/enc_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_2/Relu?
)model_6/enc_outer_1/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_1/MatMul/ReadVariableOp?
model_6/enc_outer_1/MatMulMatMulx1model_6/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_1/MatMul?
*model_6/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_1/BiasAdd/ReadVariableOp?
model_6/enc_outer_1/BiasAddBiasAdd$model_6/enc_outer_1/MatMul:product:02model_6/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_1/BiasAdd?
model_6/enc_outer_1/ReluRelu$model_6/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_1/Relu?
)model_6/enc_outer_0/MatMul/ReadVariableOpReadVariableOp2model_6_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_6/enc_outer_0/MatMul/ReadVariableOp?
model_6/enc_outer_0/MatMulMatMulx1model_6/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_0/MatMul?
*model_6/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_6/enc_outer_0/BiasAdd/ReadVariableOp?
model_6/enc_outer_0/BiasAddBiasAdd$model_6/enc_outer_0/MatMul:product:02model_6/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_0/BiasAdd?
model_6/enc_outer_0/ReluRelu$model_6/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_6/enc_outer_0/Relu?
*model_6/enc_middle_3/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_3_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_3/MatMul/ReadVariableOp?
model_6/enc_middle_3/MatMulMatMul&model_6/enc_outer_3/Relu:activations:02model_6/enc_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_3/MatMul?
+model_6/enc_middle_3/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_3/BiasAdd/ReadVariableOp?
model_6/enc_middle_3/BiasAddBiasAdd%model_6/enc_middle_3/MatMul:product:03model_6/enc_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_3/BiasAdd?
model_6/enc_middle_3/ReluRelu%model_6/enc_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_3/Relu?
*model_6/enc_middle_2/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_2_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_2/MatMul/ReadVariableOp?
model_6/enc_middle_2/MatMulMatMul&model_6/enc_outer_2/Relu:activations:02model_6/enc_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_2/MatMul?
+model_6/enc_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_2/BiasAdd/ReadVariableOp?
model_6/enc_middle_2/BiasAddBiasAdd%model_6/enc_middle_2/MatMul:product:03model_6/enc_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_2/BiasAdd?
model_6/enc_middle_2/ReluRelu%model_6/enc_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_2/Relu?
*model_6/enc_middle_1/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_1/MatMul/ReadVariableOp?
model_6/enc_middle_1/MatMulMatMul&model_6/enc_outer_1/Relu:activations:02model_6/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_1/MatMul?
+model_6/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_1/BiasAdd/ReadVariableOp?
model_6/enc_middle_1/BiasAddBiasAdd%model_6/enc_middle_1/MatMul:product:03model_6/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_1/BiasAdd?
model_6/enc_middle_1/ReluRelu%model_6/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_1/Relu?
*model_6/enc_middle_0/MatMul/ReadVariableOpReadVariableOp3model_6_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_6/enc_middle_0/MatMul/ReadVariableOp?
model_6/enc_middle_0/MatMulMatMul&model_6/enc_outer_0/Relu:activations:02model_6/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_0/MatMul?
+model_6/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_6_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_6/enc_middle_0/BiasAdd/ReadVariableOp?
model_6/enc_middle_0/BiasAddBiasAdd%model_6/enc_middle_0/MatMul:product:03model_6/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_0/BiasAdd?
model_6/enc_middle_0/ReluRelu%model_6/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_6/enc_middle_0/Relu?
)model_6/enc_inner_3/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_3_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_3/MatMul/ReadVariableOp?
model_6/enc_inner_3/MatMulMatMul'model_6/enc_middle_3/Relu:activations:01model_6/enc_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_3/MatMul?
*model_6/enc_inner_3/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_3/BiasAdd/ReadVariableOp?
model_6/enc_inner_3/BiasAddBiasAdd$model_6/enc_inner_3/MatMul:product:02model_6/enc_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_3/BiasAdd?
model_6/enc_inner_3/ReluRelu$model_6/enc_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_3/Relu?
)model_6/enc_inner_2/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_2_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_2/MatMul/ReadVariableOp?
model_6/enc_inner_2/MatMulMatMul'model_6/enc_middle_2/Relu:activations:01model_6/enc_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_2/MatMul?
*model_6/enc_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_2/BiasAdd/ReadVariableOp?
model_6/enc_inner_2/BiasAddBiasAdd$model_6/enc_inner_2/MatMul:product:02model_6/enc_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_2/BiasAdd?
model_6/enc_inner_2/ReluRelu$model_6/enc_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_2/Relu?
)model_6/enc_inner_1/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_1/MatMul/ReadVariableOp?
model_6/enc_inner_1/MatMulMatMul'model_6/enc_middle_1/Relu:activations:01model_6/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_1/MatMul?
*model_6/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_1/BiasAdd/ReadVariableOp?
model_6/enc_inner_1/BiasAddBiasAdd$model_6/enc_inner_1/MatMul:product:02model_6/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_1/BiasAdd?
model_6/enc_inner_1/ReluRelu$model_6/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_1/Relu?
)model_6/enc_inner_0/MatMul/ReadVariableOpReadVariableOp2model_6_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_6/enc_inner_0/MatMul/ReadVariableOp?
model_6/enc_inner_0/MatMulMatMul'model_6/enc_middle_0/Relu:activations:01model_6/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_0/MatMul?
*model_6/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_6_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_6/enc_inner_0/BiasAdd/ReadVariableOp?
model_6/enc_inner_0/BiasAddBiasAdd$model_6/enc_inner_0/MatMul:product:02model_6/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_0/BiasAdd?
model_6/enc_inner_0/ReluRelu$model_6/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_6/enc_inner_0/Relu?
'model_6/channel_3/MatMul/ReadVariableOpReadVariableOp0model_6_channel_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_3/MatMul/ReadVariableOp?
model_6/channel_3/MatMulMatMul&model_6/enc_inner_3/Relu:activations:0/model_6/channel_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_3/MatMul?
(model_6/channel_3/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_3/BiasAdd/ReadVariableOp?
model_6/channel_3/BiasAddBiasAdd"model_6/channel_3/MatMul:product:00model_6/channel_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_3/BiasAdd?
model_6/channel_3/SoftsignSoftsign"model_6/channel_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_3/Softsign?
'model_6/channel_2/MatMul/ReadVariableOpReadVariableOp0model_6_channel_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_2/MatMul/ReadVariableOp?
model_6/channel_2/MatMulMatMul&model_6/enc_inner_2/Relu:activations:0/model_6/channel_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_2/MatMul?
(model_6/channel_2/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_2/BiasAdd/ReadVariableOp?
model_6/channel_2/BiasAddBiasAdd"model_6/channel_2/MatMul:product:00model_6/channel_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_2/BiasAdd?
model_6/channel_2/SoftsignSoftsign"model_6/channel_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_2/Softsign?
'model_6/channel_1/MatMul/ReadVariableOpReadVariableOp0model_6_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_1/MatMul/ReadVariableOp?
model_6/channel_1/MatMulMatMul&model_6/enc_inner_1/Relu:activations:0/model_6/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_1/MatMul?
(model_6/channel_1/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_1/BiasAdd/ReadVariableOp?
model_6/channel_1/BiasAddBiasAdd"model_6/channel_1/MatMul:product:00model_6/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_1/BiasAdd?
model_6/channel_1/SoftsignSoftsign"model_6/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_1/Softsign?
'model_6/channel_0/MatMul/ReadVariableOpReadVariableOp0model_6_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_6/channel_0/MatMul/ReadVariableOp?
model_6/channel_0/MatMulMatMul&model_6/enc_inner_0/Relu:activations:0/model_6/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_0/MatMul?
(model_6/channel_0/BiasAdd/ReadVariableOpReadVariableOp1model_6_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_6/channel_0/BiasAdd/ReadVariableOp?
model_6/channel_0/BiasAddBiasAdd"model_6/channel_0/MatMul:product:00model_6/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/channel_0/BiasAdd?
model_6/channel_0/SoftsignSoftsign"model_6/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/channel_0/Softsign?
)model_7/dec_inner_3/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_3/MatMul/ReadVariableOp?
model_7/dec_inner_3/MatMulMatMul(model_6/channel_3/Softsign:activations:01model_7/dec_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_3/MatMul?
*model_7/dec_inner_3/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_3/BiasAdd/ReadVariableOp?
model_7/dec_inner_3/BiasAddBiasAdd$model_7/dec_inner_3/MatMul:product:02model_7/dec_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_3/BiasAdd?
model_7/dec_inner_3/ReluRelu$model_7/dec_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_3/Relu?
)model_7/dec_inner_2/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_2/MatMul/ReadVariableOp?
model_7/dec_inner_2/MatMulMatMul(model_6/channel_2/Softsign:activations:01model_7/dec_inner_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_2/MatMul?
*model_7/dec_inner_2/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_2/BiasAdd/ReadVariableOp?
model_7/dec_inner_2/BiasAddBiasAdd$model_7/dec_inner_2/MatMul:product:02model_7/dec_inner_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_2/BiasAdd?
model_7/dec_inner_2/ReluRelu$model_7/dec_inner_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_2/Relu?
)model_7/dec_inner_1/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_1/MatMul/ReadVariableOp?
model_7/dec_inner_1/MatMulMatMul(model_6/channel_1/Softsign:activations:01model_7/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_1/MatMul?
*model_7/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_1/BiasAdd/ReadVariableOp?
model_7/dec_inner_1/BiasAddBiasAdd$model_7/dec_inner_1/MatMul:product:02model_7/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_1/BiasAdd?
model_7/dec_inner_1/ReluRelu$model_7/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_1/Relu?
)model_7/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_7_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_7/dec_inner_0/MatMul/ReadVariableOp?
model_7/dec_inner_0/MatMulMatMul(model_6/channel_0/Softsign:activations:01model_7/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_0/MatMul?
*model_7/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_7/dec_inner_0/BiasAdd/ReadVariableOp?
model_7/dec_inner_0/BiasAddBiasAdd$model_7/dec_inner_0/MatMul:product:02model_7/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_0/BiasAdd?
model_7/dec_inner_0/ReluRelu$model_7/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_7/dec_inner_0/Relu?
*model_7/dec_middle_3/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_3_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_3/MatMul/ReadVariableOp?
model_7/dec_middle_3/MatMulMatMul&model_7/dec_inner_3/Relu:activations:02model_7/dec_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_3/MatMul?
+model_7/dec_middle_3/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_3/BiasAdd/ReadVariableOp?
model_7/dec_middle_3/BiasAddBiasAdd%model_7/dec_middle_3/MatMul:product:03model_7/dec_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_3/BiasAdd?
model_7/dec_middle_3/ReluRelu%model_7/dec_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_3/Relu?
*model_7/dec_middle_2/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_2_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_2/MatMul/ReadVariableOp?
model_7/dec_middle_2/MatMulMatMul&model_7/dec_inner_2/Relu:activations:02model_7/dec_middle_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_2/MatMul?
+model_7/dec_middle_2/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_2/BiasAdd/ReadVariableOp?
model_7/dec_middle_2/BiasAddBiasAdd%model_7/dec_middle_2/MatMul:product:03model_7/dec_middle_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_2/BiasAdd?
model_7/dec_middle_2/ReluRelu%model_7/dec_middle_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_2/Relu?
*model_7/dec_middle_1/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_1/MatMul/ReadVariableOp?
model_7/dec_middle_1/MatMulMatMul&model_7/dec_inner_1/Relu:activations:02model_7/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_1/MatMul?
+model_7/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_1/BiasAdd/ReadVariableOp?
model_7/dec_middle_1/BiasAddBiasAdd%model_7/dec_middle_1/MatMul:product:03model_7/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_1/BiasAdd?
model_7/dec_middle_1/ReluRelu%model_7/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_1/Relu?
*model_7/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_7_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_7/dec_middle_0/MatMul/ReadVariableOp?
model_7/dec_middle_0/MatMulMatMul&model_7/dec_inner_0/Relu:activations:02model_7/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_0/MatMul?
+model_7/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_7_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_7/dec_middle_0/BiasAdd/ReadVariableOp?
model_7/dec_middle_0/BiasAddBiasAdd%model_7/dec_middle_0/MatMul:product:03model_7/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_0/BiasAdd?
model_7/dec_middle_0/ReluRelu%model_7/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_middle_0/Relu?
)model_7/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_0/MatMul/ReadVariableOp?
model_7/dec_outer_0/MatMulMatMul'model_7/dec_middle_0/Relu:activations:01model_7/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_0/MatMul?
*model_7/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_0/BiasAdd/ReadVariableOp?
model_7/dec_outer_0/BiasAddBiasAdd$model_7/dec_outer_0/MatMul:product:02model_7/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_0/BiasAdd?
model_7/dec_outer_0/ReluRelu$model_7/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_0/Relu?
)model_7/dec_outer_1/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_1/MatMul/ReadVariableOp?
model_7/dec_outer_1/MatMulMatMul'model_7/dec_middle_1/Relu:activations:01model_7/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_1/MatMul?
*model_7/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_1/BiasAdd/ReadVariableOp?
model_7/dec_outer_1/BiasAddBiasAdd$model_7/dec_outer_1/MatMul:product:02model_7/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_1/BiasAdd?
model_7/dec_outer_1/ReluRelu$model_7/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_1/Relu?
)model_7/dec_outer_2/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_2_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_2/MatMul/ReadVariableOp?
model_7/dec_outer_2/MatMulMatMul'model_7/dec_middle_2/Relu:activations:01model_7/dec_outer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_2/MatMul?
*model_7/dec_outer_2/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_2/BiasAdd/ReadVariableOp?
model_7/dec_outer_2/BiasAddBiasAdd$model_7/dec_outer_2/MatMul:product:02model_7/dec_outer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_2/BiasAdd?
model_7/dec_outer_2/ReluRelu$model_7/dec_outer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_2/Relu?
)model_7/dec_outer_3/MatMul/ReadVariableOpReadVariableOp2model_7_dec_outer_3_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_7/dec_outer_3/MatMul/ReadVariableOp?
model_7/dec_outer_3/MatMulMatMul'model_7/dec_middle_3/Relu:activations:01model_7/dec_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_3/MatMul?
*model_7/dec_outer_3/BiasAdd/ReadVariableOpReadVariableOp3model_7_dec_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_7/dec_outer_3/BiasAdd/ReadVariableOp?
model_7/dec_outer_3/BiasAddBiasAdd$model_7/dec_outer_3/MatMul:product:02model_7/dec_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_3/BiasAdd?
model_7/dec_outer_3/ReluRelu$model_7/dec_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_7/dec_outer_3/Relu?
model_7/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model_7/tf.concat_2/concat/axis?
model_7/tf.concat_2/concatConcatV2&model_7/dec_outer_0/Relu:activations:0&model_7/dec_outer_1/Relu:activations:0&model_7/dec_outer_2/Relu:activations:0&model_7/dec_outer_3/Relu:activations:0(model_7/tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_7/tf.concat_2/concat?
(model_7/dec_output/MatMul/ReadVariableOpReadVariableOp1model_7_dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_7/dec_output/MatMul/ReadVariableOp?
model_7/dec_output/MatMulMatMul#model_7/tf.concat_2/concat:output:00model_7/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_7/dec_output/MatMul?
)model_7/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_7_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_7/dec_output/BiasAdd/ReadVariableOp?
model_7/dec_output/BiasAddBiasAdd#model_7/dec_output/MatMul:product:01model_7/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_7/dec_output/BiasAdd?
model_7/dec_output/SigmoidSigmoid#model_7/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_7/dec_output/Sigmoid?
IdentityIdentitymodel_7/dec_output/Sigmoid:y:0)^model_6/channel_0/BiasAdd/ReadVariableOp(^model_6/channel_0/MatMul/ReadVariableOp)^model_6/channel_1/BiasAdd/ReadVariableOp(^model_6/channel_1/MatMul/ReadVariableOp)^model_6/channel_2/BiasAdd/ReadVariableOp(^model_6/channel_2/MatMul/ReadVariableOp)^model_6/channel_3/BiasAdd/ReadVariableOp(^model_6/channel_3/MatMul/ReadVariableOp+^model_6/enc_inner_0/BiasAdd/ReadVariableOp*^model_6/enc_inner_0/MatMul/ReadVariableOp+^model_6/enc_inner_1/BiasAdd/ReadVariableOp*^model_6/enc_inner_1/MatMul/ReadVariableOp+^model_6/enc_inner_2/BiasAdd/ReadVariableOp*^model_6/enc_inner_2/MatMul/ReadVariableOp+^model_6/enc_inner_3/BiasAdd/ReadVariableOp*^model_6/enc_inner_3/MatMul/ReadVariableOp,^model_6/enc_middle_0/BiasAdd/ReadVariableOp+^model_6/enc_middle_0/MatMul/ReadVariableOp,^model_6/enc_middle_1/BiasAdd/ReadVariableOp+^model_6/enc_middle_1/MatMul/ReadVariableOp,^model_6/enc_middle_2/BiasAdd/ReadVariableOp+^model_6/enc_middle_2/MatMul/ReadVariableOp,^model_6/enc_middle_3/BiasAdd/ReadVariableOp+^model_6/enc_middle_3/MatMul/ReadVariableOp+^model_6/enc_outer_0/BiasAdd/ReadVariableOp*^model_6/enc_outer_0/MatMul/ReadVariableOp+^model_6/enc_outer_1/BiasAdd/ReadVariableOp*^model_6/enc_outer_1/MatMul/ReadVariableOp+^model_6/enc_outer_2/BiasAdd/ReadVariableOp*^model_6/enc_outer_2/MatMul/ReadVariableOp+^model_6/enc_outer_3/BiasAdd/ReadVariableOp*^model_6/enc_outer_3/MatMul/ReadVariableOp+^model_7/dec_inner_0/BiasAdd/ReadVariableOp*^model_7/dec_inner_0/MatMul/ReadVariableOp+^model_7/dec_inner_1/BiasAdd/ReadVariableOp*^model_7/dec_inner_1/MatMul/ReadVariableOp+^model_7/dec_inner_2/BiasAdd/ReadVariableOp*^model_7/dec_inner_2/MatMul/ReadVariableOp+^model_7/dec_inner_3/BiasAdd/ReadVariableOp*^model_7/dec_inner_3/MatMul/ReadVariableOp,^model_7/dec_middle_0/BiasAdd/ReadVariableOp+^model_7/dec_middle_0/MatMul/ReadVariableOp,^model_7/dec_middle_1/BiasAdd/ReadVariableOp+^model_7/dec_middle_1/MatMul/ReadVariableOp,^model_7/dec_middle_2/BiasAdd/ReadVariableOp+^model_7/dec_middle_2/MatMul/ReadVariableOp,^model_7/dec_middle_3/BiasAdd/ReadVariableOp+^model_7/dec_middle_3/MatMul/ReadVariableOp+^model_7/dec_outer_0/BiasAdd/ReadVariableOp*^model_7/dec_outer_0/MatMul/ReadVariableOp+^model_7/dec_outer_1/BiasAdd/ReadVariableOp*^model_7/dec_outer_1/MatMul/ReadVariableOp+^model_7/dec_outer_2/BiasAdd/ReadVariableOp*^model_7/dec_outer_2/MatMul/ReadVariableOp+^model_7/dec_outer_3/BiasAdd/ReadVariableOp*^model_7/dec_outer_3/MatMul/ReadVariableOp*^model_7/dec_output/BiasAdd/ReadVariableOp)^model_7/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2T
(model_6/channel_0/BiasAdd/ReadVariableOp(model_6/channel_0/BiasAdd/ReadVariableOp2R
'model_6/channel_0/MatMul/ReadVariableOp'model_6/channel_0/MatMul/ReadVariableOp2T
(model_6/channel_1/BiasAdd/ReadVariableOp(model_6/channel_1/BiasAdd/ReadVariableOp2R
'model_6/channel_1/MatMul/ReadVariableOp'model_6/channel_1/MatMul/ReadVariableOp2T
(model_6/channel_2/BiasAdd/ReadVariableOp(model_6/channel_2/BiasAdd/ReadVariableOp2R
'model_6/channel_2/MatMul/ReadVariableOp'model_6/channel_2/MatMul/ReadVariableOp2T
(model_6/channel_3/BiasAdd/ReadVariableOp(model_6/channel_3/BiasAdd/ReadVariableOp2R
'model_6/channel_3/MatMul/ReadVariableOp'model_6/channel_3/MatMul/ReadVariableOp2X
*model_6/enc_inner_0/BiasAdd/ReadVariableOp*model_6/enc_inner_0/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_0/MatMul/ReadVariableOp)model_6/enc_inner_0/MatMul/ReadVariableOp2X
*model_6/enc_inner_1/BiasAdd/ReadVariableOp*model_6/enc_inner_1/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_1/MatMul/ReadVariableOp)model_6/enc_inner_1/MatMul/ReadVariableOp2X
*model_6/enc_inner_2/BiasAdd/ReadVariableOp*model_6/enc_inner_2/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_2/MatMul/ReadVariableOp)model_6/enc_inner_2/MatMul/ReadVariableOp2X
*model_6/enc_inner_3/BiasAdd/ReadVariableOp*model_6/enc_inner_3/BiasAdd/ReadVariableOp2V
)model_6/enc_inner_3/MatMul/ReadVariableOp)model_6/enc_inner_3/MatMul/ReadVariableOp2Z
+model_6/enc_middle_0/BiasAdd/ReadVariableOp+model_6/enc_middle_0/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_0/MatMul/ReadVariableOp*model_6/enc_middle_0/MatMul/ReadVariableOp2Z
+model_6/enc_middle_1/BiasAdd/ReadVariableOp+model_6/enc_middle_1/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_1/MatMul/ReadVariableOp*model_6/enc_middle_1/MatMul/ReadVariableOp2Z
+model_6/enc_middle_2/BiasAdd/ReadVariableOp+model_6/enc_middle_2/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_2/MatMul/ReadVariableOp*model_6/enc_middle_2/MatMul/ReadVariableOp2Z
+model_6/enc_middle_3/BiasAdd/ReadVariableOp+model_6/enc_middle_3/BiasAdd/ReadVariableOp2X
*model_6/enc_middle_3/MatMul/ReadVariableOp*model_6/enc_middle_3/MatMul/ReadVariableOp2X
*model_6/enc_outer_0/BiasAdd/ReadVariableOp*model_6/enc_outer_0/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_0/MatMul/ReadVariableOp)model_6/enc_outer_0/MatMul/ReadVariableOp2X
*model_6/enc_outer_1/BiasAdd/ReadVariableOp*model_6/enc_outer_1/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_1/MatMul/ReadVariableOp)model_6/enc_outer_1/MatMul/ReadVariableOp2X
*model_6/enc_outer_2/BiasAdd/ReadVariableOp*model_6/enc_outer_2/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_2/MatMul/ReadVariableOp)model_6/enc_outer_2/MatMul/ReadVariableOp2X
*model_6/enc_outer_3/BiasAdd/ReadVariableOp*model_6/enc_outer_3/BiasAdd/ReadVariableOp2V
)model_6/enc_outer_3/MatMul/ReadVariableOp)model_6/enc_outer_3/MatMul/ReadVariableOp2X
*model_7/dec_inner_0/BiasAdd/ReadVariableOp*model_7/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_0/MatMul/ReadVariableOp)model_7/dec_inner_0/MatMul/ReadVariableOp2X
*model_7/dec_inner_1/BiasAdd/ReadVariableOp*model_7/dec_inner_1/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_1/MatMul/ReadVariableOp)model_7/dec_inner_1/MatMul/ReadVariableOp2X
*model_7/dec_inner_2/BiasAdd/ReadVariableOp*model_7/dec_inner_2/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_2/MatMul/ReadVariableOp)model_7/dec_inner_2/MatMul/ReadVariableOp2X
*model_7/dec_inner_3/BiasAdd/ReadVariableOp*model_7/dec_inner_3/BiasAdd/ReadVariableOp2V
)model_7/dec_inner_3/MatMul/ReadVariableOp)model_7/dec_inner_3/MatMul/ReadVariableOp2Z
+model_7/dec_middle_0/BiasAdd/ReadVariableOp+model_7/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_0/MatMul/ReadVariableOp*model_7/dec_middle_0/MatMul/ReadVariableOp2Z
+model_7/dec_middle_1/BiasAdd/ReadVariableOp+model_7/dec_middle_1/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_1/MatMul/ReadVariableOp*model_7/dec_middle_1/MatMul/ReadVariableOp2Z
+model_7/dec_middle_2/BiasAdd/ReadVariableOp+model_7/dec_middle_2/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_2/MatMul/ReadVariableOp*model_7/dec_middle_2/MatMul/ReadVariableOp2Z
+model_7/dec_middle_3/BiasAdd/ReadVariableOp+model_7/dec_middle_3/BiasAdd/ReadVariableOp2X
*model_7/dec_middle_3/MatMul/ReadVariableOp*model_7/dec_middle_3/MatMul/ReadVariableOp2X
*model_7/dec_outer_0/BiasAdd/ReadVariableOp*model_7/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_0/MatMul/ReadVariableOp)model_7/dec_outer_0/MatMul/ReadVariableOp2X
*model_7/dec_outer_1/BiasAdd/ReadVariableOp*model_7/dec_outer_1/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_1/MatMul/ReadVariableOp)model_7/dec_outer_1/MatMul/ReadVariableOp2X
*model_7/dec_outer_2/BiasAdd/ReadVariableOp*model_7/dec_outer_2/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_2/MatMul/ReadVariableOp)model_7/dec_outer_2/MatMul/ReadVariableOp2X
*model_7/dec_outer_3/BiasAdd/ReadVariableOp*model_7/dec_outer_3/BiasAdd/ReadVariableOp2V
)model_7/dec_outer_3/MatMul/ReadVariableOp)model_7/dec_outer_3/MatMul/ReadVariableOp2V
)model_7/dec_output/BiasAdd/ReadVariableOp)model_7/dec_output/BiasAdd/ReadVariableOp2T
(model_7/dec_output/MatMul/ReadVariableOp(model_7/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_305590

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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_309045

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
?h
?
C__inference_model_6_layer_call_and_return_conditional_losses_305202

inputs
enc_outer_3_305118
enc_outer_3_305120
enc_outer_2_305123
enc_outer_2_305125
enc_outer_1_305128
enc_outer_1_305130
enc_outer_0_305133
enc_outer_0_305135
enc_middle_3_305138
enc_middle_3_305140
enc_middle_2_305143
enc_middle_2_305145
enc_middle_1_305148
enc_middle_1_305150
enc_middle_0_305153
enc_middle_0_305155
enc_inner_3_305158
enc_inner_3_305160
enc_inner_2_305163
enc_inner_2_305165
enc_inner_1_305168
enc_inner_1_305170
enc_inner_0_305173
enc_inner_0_305175
channel_3_305178
channel_3_305180
channel_2_305183
channel_2_305185
channel_1_305188
channel_1_305190
channel_0_305193
channel_0_305195
identity

identity_1

identity_2

identity_3??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?!channel_3/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?#enc_inner_3/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?$enc_middle_3/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?#enc_outer_3/StatefulPartitionedCall?
#enc_outer_3/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_3_305118enc_outer_3_305120*
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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_3046002%
#enc_outer_3/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_2_305123enc_outer_2_305125*
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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_3046272%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_1_305128enc_outer_1_305130*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_3046542%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_305133enc_outer_0_305135*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_3046812%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_3/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_3/StatefulPartitionedCall:output:0enc_middle_3_305138enc_middle_3_305140*
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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_3047082&
$enc_middle_3/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_305143enc_middle_2_305145*
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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_3047352&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_305148enc_middle_1_305150*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_3047622&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_305153enc_middle_0_305155*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_3047892&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_3/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_3/StatefulPartitionedCall:output:0enc_inner_3_305158enc_inner_3_305160*
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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_3048162%
#enc_inner_3/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_305163enc_inner_2_305165*
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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_3048432%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_305168enc_inner_1_305170*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_3048702%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_305173enc_inner_0_305175*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_3048972%
#enc_inner_0/StatefulPartitionedCall?
!channel_3/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_3/StatefulPartitionedCall:output:0channel_3_305178channel_3_305180*
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
E__inference_channel_3_layer_call_and_return_conditional_losses_3049242#
!channel_3/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_305183channel_2_305185*
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
E__inference_channel_2_layer_call_and_return_conditional_losses_3049512#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_305188channel_1_305190*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_3049782#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_305193channel_0_305195*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_3050052#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity*channel_3/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2F
!channel_3/StatefulPartitionedCall!channel_3/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2J
#enc_inner_3/StatefulPartitionedCall#enc_inner_3/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2L
$enc_middle_3/StatefulPartitionedCall$enc_middle_3/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall2J
#enc_outer_3/StatefulPartitionedCall#enc_outer_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_309165

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
?L
?	
C__inference_model_7_layer_call_and_return_conditional_losses_306086

inputs
inputs_1
inputs_2
inputs_3
dec_inner_3_306018
dec_inner_3_306020
dec_inner_2_306023
dec_inner_2_306025
dec_inner_1_306028
dec_inner_1_306030
dec_inner_0_306033
dec_inner_0_306035
dec_middle_3_306038
dec_middle_3_306040
dec_middle_2_306043
dec_middle_2_306045
dec_middle_1_306048
dec_middle_1_306050
dec_middle_0_306053
dec_middle_0_306055
dec_outer_0_306058
dec_outer_0_306060
dec_outer_1_306063
dec_outer_1_306065
dec_outer_2_306068
dec_outer_2_306070
dec_outer_3_306073
dec_outer_3_306075
dec_output_306080
dec_output_306082
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?#dec_inner_2/StatefulPartitionedCall?#dec_inner_3/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?$dec_middle_2/StatefulPartitionedCall?$dec_middle_3/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?#dec_outer_2/StatefulPartitionedCall?#dec_outer_3/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_3/StatefulPartitionedCallStatefulPartitionedCallinputs_3dec_inner_3_306018dec_inner_3_306020*
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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_3054552%
#dec_inner_3/StatefulPartitionedCall?
#dec_inner_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2dec_inner_2_306023dec_inner_2_306025*
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
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_3054822%
#dec_inner_2/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1dec_inner_1_306028dec_inner_1_306030*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_3055092%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_306033dec_inner_0_306035*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_3055362%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_3/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_3/StatefulPartitionedCall:output:0dec_middle_3_306038dec_middle_3_306040*
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
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_3055632&
$dec_middle_3/StatefulPartitionedCall?
$dec_middle_2/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_2/StatefulPartitionedCall:output:0dec_middle_2_306043dec_middle_2_306045*
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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_3055902&
$dec_middle_2/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_306048dec_middle_1_306050*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_3056172&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_306053dec_middle_0_306055*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_3056442&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_306058dec_outer_0_306060*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_3056712%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_306063dec_outer_1_306065*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_3056982%
#dec_outer_1/StatefulPartitionedCall?
#dec_outer_2/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_2/StatefulPartitionedCall:output:0dec_outer_2_306068dec_outer_2_306070*
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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_3057252%
#dec_outer_2/StatefulPartitionedCall?
#dec_outer_3/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_3/StatefulPartitionedCall:output:0dec_outer_3_306073dec_outer_3_306075*
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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_3057522%
#dec_outer_3/StatefulPartitionedCallt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0,dec_outer_2/StatefulPartitionedCall:output:0,dec_outer_3/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dec_output_306080dec_output_306082*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_3057812$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall$^dec_inner_2/StatefulPartitionedCall$^dec_inner_3/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall%^dec_middle_2/StatefulPartitionedCall%^dec_middle_3/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall$^dec_outer_2/StatefulPartitionedCall$^dec_outer_3/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2J
#dec_inner_2/StatefulPartitionedCall#dec_inner_2/StatefulPartitionedCall2J
#dec_inner_3/StatefulPartitionedCall#dec_inner_3/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2L
$dec_middle_2/StatefulPartitionedCall$dec_middle_2/StatefulPartitionedCall2L
$dec_middle_3/StatefulPartitionedCall$dec_middle_3/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2J
#dec_outer_2/StatefulPartitionedCall#dec_outer_2/StatefulPartitionedCall2J
#dec_outer_3/StatefulPartitionedCall#dec_outer_3/StatefulPartitionedCall2H
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_308805

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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_305455

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
F__inference_dec_output_layer_call_and_return_conditional_losses_309225

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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_308685

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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_308845

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
??
?c
"__inference__traced_restore_310353
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
$assignvariableop_10_enc_outer_2_bias*
&assignvariableop_11_enc_outer_3_kernel(
$assignvariableop_12_enc_outer_3_bias+
'assignvariableop_13_enc_middle_0_kernel)
%assignvariableop_14_enc_middle_0_bias+
'assignvariableop_15_enc_middle_1_kernel)
%assignvariableop_16_enc_middle_1_bias+
'assignvariableop_17_enc_middle_2_kernel)
%assignvariableop_18_enc_middle_2_bias+
'assignvariableop_19_enc_middle_3_kernel)
%assignvariableop_20_enc_middle_3_bias*
&assignvariableop_21_enc_inner_0_kernel(
$assignvariableop_22_enc_inner_0_bias*
&assignvariableop_23_enc_inner_1_kernel(
$assignvariableop_24_enc_inner_1_bias*
&assignvariableop_25_enc_inner_2_kernel(
$assignvariableop_26_enc_inner_2_bias*
&assignvariableop_27_enc_inner_3_kernel(
$assignvariableop_28_enc_inner_3_bias(
$assignvariableop_29_channel_0_kernel&
"assignvariableop_30_channel_0_bias(
$assignvariableop_31_channel_1_kernel&
"assignvariableop_32_channel_1_bias(
$assignvariableop_33_channel_2_kernel&
"assignvariableop_34_channel_2_bias(
$assignvariableop_35_channel_3_kernel&
"assignvariableop_36_channel_3_bias*
&assignvariableop_37_dec_inner_0_kernel(
$assignvariableop_38_dec_inner_0_bias*
&assignvariableop_39_dec_inner_1_kernel(
$assignvariableop_40_dec_inner_1_bias*
&assignvariableop_41_dec_inner_2_kernel(
$assignvariableop_42_dec_inner_2_bias*
&assignvariableop_43_dec_inner_3_kernel(
$assignvariableop_44_dec_inner_3_bias+
'assignvariableop_45_dec_middle_0_kernel)
%assignvariableop_46_dec_middle_0_bias+
'assignvariableop_47_dec_middle_1_kernel)
%assignvariableop_48_dec_middle_1_bias+
'assignvariableop_49_dec_middle_2_kernel)
%assignvariableop_50_dec_middle_2_bias+
'assignvariableop_51_dec_middle_3_kernel)
%assignvariableop_52_dec_middle_3_bias*
&assignvariableop_53_dec_outer_0_kernel(
$assignvariableop_54_dec_outer_0_bias*
&assignvariableop_55_dec_outer_1_kernel(
$assignvariableop_56_dec_outer_1_bias*
&assignvariableop_57_dec_outer_2_kernel(
$assignvariableop_58_dec_outer_2_bias*
&assignvariableop_59_dec_outer_3_kernel(
$assignvariableop_60_dec_outer_3_bias)
%assignvariableop_61_dec_output_kernel'
#assignvariableop_62_dec_output_bias
assignvariableop_63_total
assignvariableop_64_count1
-assignvariableop_65_adam_enc_outer_0_kernel_m/
+assignvariableop_66_adam_enc_outer_0_bias_m1
-assignvariableop_67_adam_enc_outer_1_kernel_m/
+assignvariableop_68_adam_enc_outer_1_bias_m1
-assignvariableop_69_adam_enc_outer_2_kernel_m/
+assignvariableop_70_adam_enc_outer_2_bias_m1
-assignvariableop_71_adam_enc_outer_3_kernel_m/
+assignvariableop_72_adam_enc_outer_3_bias_m2
.assignvariableop_73_adam_enc_middle_0_kernel_m0
,assignvariableop_74_adam_enc_middle_0_bias_m2
.assignvariableop_75_adam_enc_middle_1_kernel_m0
,assignvariableop_76_adam_enc_middle_1_bias_m2
.assignvariableop_77_adam_enc_middle_2_kernel_m0
,assignvariableop_78_adam_enc_middle_2_bias_m2
.assignvariableop_79_adam_enc_middle_3_kernel_m0
,assignvariableop_80_adam_enc_middle_3_bias_m1
-assignvariableop_81_adam_enc_inner_0_kernel_m/
+assignvariableop_82_adam_enc_inner_0_bias_m1
-assignvariableop_83_adam_enc_inner_1_kernel_m/
+assignvariableop_84_adam_enc_inner_1_bias_m1
-assignvariableop_85_adam_enc_inner_2_kernel_m/
+assignvariableop_86_adam_enc_inner_2_bias_m1
-assignvariableop_87_adam_enc_inner_3_kernel_m/
+assignvariableop_88_adam_enc_inner_3_bias_m/
+assignvariableop_89_adam_channel_0_kernel_m-
)assignvariableop_90_adam_channel_0_bias_m/
+assignvariableop_91_adam_channel_1_kernel_m-
)assignvariableop_92_adam_channel_1_bias_m/
+assignvariableop_93_adam_channel_2_kernel_m-
)assignvariableop_94_adam_channel_2_bias_m/
+assignvariableop_95_adam_channel_3_kernel_m-
)assignvariableop_96_adam_channel_3_bias_m1
-assignvariableop_97_adam_dec_inner_0_kernel_m/
+assignvariableop_98_adam_dec_inner_0_bias_m1
-assignvariableop_99_adam_dec_inner_1_kernel_m0
,assignvariableop_100_adam_dec_inner_1_bias_m2
.assignvariableop_101_adam_dec_inner_2_kernel_m0
,assignvariableop_102_adam_dec_inner_2_bias_m2
.assignvariableop_103_adam_dec_inner_3_kernel_m0
,assignvariableop_104_adam_dec_inner_3_bias_m3
/assignvariableop_105_adam_dec_middle_0_kernel_m1
-assignvariableop_106_adam_dec_middle_0_bias_m3
/assignvariableop_107_adam_dec_middle_1_kernel_m1
-assignvariableop_108_adam_dec_middle_1_bias_m3
/assignvariableop_109_adam_dec_middle_2_kernel_m1
-assignvariableop_110_adam_dec_middle_2_bias_m3
/assignvariableop_111_adam_dec_middle_3_kernel_m1
-assignvariableop_112_adam_dec_middle_3_bias_m2
.assignvariableop_113_adam_dec_outer_0_kernel_m0
,assignvariableop_114_adam_dec_outer_0_bias_m2
.assignvariableop_115_adam_dec_outer_1_kernel_m0
,assignvariableop_116_adam_dec_outer_1_bias_m2
.assignvariableop_117_adam_dec_outer_2_kernel_m0
,assignvariableop_118_adam_dec_outer_2_bias_m2
.assignvariableop_119_adam_dec_outer_3_kernel_m0
,assignvariableop_120_adam_dec_outer_3_bias_m1
-assignvariableop_121_adam_dec_output_kernel_m/
+assignvariableop_122_adam_dec_output_bias_m2
.assignvariableop_123_adam_enc_outer_0_kernel_v0
,assignvariableop_124_adam_enc_outer_0_bias_v2
.assignvariableop_125_adam_enc_outer_1_kernel_v0
,assignvariableop_126_adam_enc_outer_1_bias_v2
.assignvariableop_127_adam_enc_outer_2_kernel_v0
,assignvariableop_128_adam_enc_outer_2_bias_v2
.assignvariableop_129_adam_enc_outer_3_kernel_v0
,assignvariableop_130_adam_enc_outer_3_bias_v3
/assignvariableop_131_adam_enc_middle_0_kernel_v1
-assignvariableop_132_adam_enc_middle_0_bias_v3
/assignvariableop_133_adam_enc_middle_1_kernel_v1
-assignvariableop_134_adam_enc_middle_1_bias_v3
/assignvariableop_135_adam_enc_middle_2_kernel_v1
-assignvariableop_136_adam_enc_middle_2_bias_v3
/assignvariableop_137_adam_enc_middle_3_kernel_v1
-assignvariableop_138_adam_enc_middle_3_bias_v2
.assignvariableop_139_adam_enc_inner_0_kernel_v0
,assignvariableop_140_adam_enc_inner_0_bias_v2
.assignvariableop_141_adam_enc_inner_1_kernel_v0
,assignvariableop_142_adam_enc_inner_1_bias_v2
.assignvariableop_143_adam_enc_inner_2_kernel_v0
,assignvariableop_144_adam_enc_inner_2_bias_v2
.assignvariableop_145_adam_enc_inner_3_kernel_v0
,assignvariableop_146_adam_enc_inner_3_bias_v0
,assignvariableop_147_adam_channel_0_kernel_v.
*assignvariableop_148_adam_channel_0_bias_v0
,assignvariableop_149_adam_channel_1_kernel_v.
*assignvariableop_150_adam_channel_1_bias_v0
,assignvariableop_151_adam_channel_2_kernel_v.
*assignvariableop_152_adam_channel_2_bias_v0
,assignvariableop_153_adam_channel_3_kernel_v.
*assignvariableop_154_adam_channel_3_bias_v2
.assignvariableop_155_adam_dec_inner_0_kernel_v0
,assignvariableop_156_adam_dec_inner_0_bias_v2
.assignvariableop_157_adam_dec_inner_1_kernel_v0
,assignvariableop_158_adam_dec_inner_1_bias_v2
.assignvariableop_159_adam_dec_inner_2_kernel_v0
,assignvariableop_160_adam_dec_inner_2_bias_v2
.assignvariableop_161_adam_dec_inner_3_kernel_v0
,assignvariableop_162_adam_dec_inner_3_bias_v3
/assignvariableop_163_adam_dec_middle_0_kernel_v1
-assignvariableop_164_adam_dec_middle_0_bias_v3
/assignvariableop_165_adam_dec_middle_1_kernel_v1
-assignvariableop_166_adam_dec_middle_1_bias_v3
/assignvariableop_167_adam_dec_middle_2_kernel_v1
-assignvariableop_168_adam_dec_middle_2_bias_v3
/assignvariableop_169_adam_dec_middle_3_kernel_v1
-assignvariableop_170_adam_dec_middle_3_bias_v2
.assignvariableop_171_adam_dec_outer_0_kernel_v0
,assignvariableop_172_adam_dec_outer_0_bias_v2
.assignvariableop_173_adam_dec_outer_1_kernel_v0
,assignvariableop_174_adam_dec_outer_1_bias_v2
.assignvariableop_175_adam_dec_outer_2_kernel_v0
,assignvariableop_176_adam_dec_outer_2_bias_v2
.assignvariableop_177_adam_dec_outer_3_kernel_v0
,assignvariableop_178_adam_dec_outer_3_bias_v1
-assignvariableop_179_adam_dec_output_kernel_v/
+assignvariableop_180_adam_dec_output_bias_v
identity_182??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_153?AssignVariableOp_154?AssignVariableOp_155?AssignVariableOp_156?AssignVariableOp_157?AssignVariableOp_158?AssignVariableOp_159?AssignVariableOp_16?AssignVariableOp_160?AssignVariableOp_161?AssignVariableOp_162?AssignVariableOp_163?AssignVariableOp_164?AssignVariableOp_165?AssignVariableOp_166?AssignVariableOp_167?AssignVariableOp_168?AssignVariableOp_169?AssignVariableOp_17?AssignVariableOp_170?AssignVariableOp_171?AssignVariableOp_172?AssignVariableOp_173?AssignVariableOp_174?AssignVariableOp_175?AssignVariableOp_176?AssignVariableOp_177?AssignVariableOp_178?AssignVariableOp_179?AssignVariableOp_18?AssignVariableOp_180?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?T
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?S
value?SB?S?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/54/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/55/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/56/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/57/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/54/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/55/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/56/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/57/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
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
AssignVariableOp_11AssignVariableOp&assignvariableop_11_enc_outer_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_enc_outer_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_enc_middle_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_enc_middle_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_enc_middle_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_enc_middle_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_enc_middle_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_enc_middle_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_enc_middle_3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_enc_middle_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_enc_inner_0_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_enc_inner_0_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp&assignvariableop_23_enc_inner_1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_enc_inner_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp&assignvariableop_25_enc_inner_2_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_enc_inner_2_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp&assignvariableop_27_enc_inner_3_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp$assignvariableop_28_enc_inner_3_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp$assignvariableop_29_channel_0_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_channel_0_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_channel_1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_channel_1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_channel_2_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_channel_2_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp$assignvariableop_35_channel_3_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp"assignvariableop_36_channel_3_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp&assignvariableop_37_dec_inner_0_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_dec_inner_0_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp&assignvariableop_39_dec_inner_1_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_dec_inner_1_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp&assignvariableop_41_dec_inner_2_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_dec_inner_2_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp&assignvariableop_43_dec_inner_3_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp$assignvariableop_44_dec_inner_3_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_dec_middle_0_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp%assignvariableop_46_dec_middle_0_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp'assignvariableop_47_dec_middle_1_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp%assignvariableop_48_dec_middle_1_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp'assignvariableop_49_dec_middle_2_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp%assignvariableop_50_dec_middle_2_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp'assignvariableop_51_dec_middle_3_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp%assignvariableop_52_dec_middle_3_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp&assignvariableop_53_dec_outer_0_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp$assignvariableop_54_dec_outer_0_biasIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp&assignvariableop_55_dec_outer_1_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_dec_outer_1_biasIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp&assignvariableop_57_dec_outer_2_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp$assignvariableop_58_dec_outer_2_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp&assignvariableop_59_dec_outer_3_kernelIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp$assignvariableop_60_dec_outer_3_biasIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp%assignvariableop_61_dec_output_kernelIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp#assignvariableop_62_dec_output_biasIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpassignvariableop_63_totalIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpassignvariableop_64_countIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp-assignvariableop_65_adam_enc_outer_0_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_enc_outer_0_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp-assignvariableop_67_adam_enc_outer_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_enc_outer_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_enc_outer_2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_enc_outer_2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp-assignvariableop_71_adam_enc_outer_3_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_enc_outer_3_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp.assignvariableop_73_adam_enc_middle_0_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp,assignvariableop_74_adam_enc_middle_0_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp.assignvariableop_75_adam_enc_middle_1_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp,assignvariableop_76_adam_enc_middle_1_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp.assignvariableop_77_adam_enc_middle_2_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp,assignvariableop_78_adam_enc_middle_2_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_enc_middle_3_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_enc_middle_3_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp-assignvariableop_81_adam_enc_inner_0_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp+assignvariableop_82_adam_enc_inner_0_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp-assignvariableop_83_adam_enc_inner_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp+assignvariableop_84_adam_enc_inner_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp-assignvariableop_85_adam_enc_inner_2_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adam_enc_inner_2_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp-assignvariableop_87_adam_enc_inner_3_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp+assignvariableop_88_adam_enc_inner_3_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_channel_0_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_channel_0_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_channel_1_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_channel_1_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_channel_2_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_channel_2_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp+assignvariableop_95_adam_channel_3_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp)assignvariableop_96_adam_channel_3_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp-assignvariableop_97_adam_dec_inner_0_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp+assignvariableop_98_adam_dec_inner_0_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp-assignvariableop_99_adam_dec_inner_1_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp,assignvariableop_100_adam_dec_inner_1_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp.assignvariableop_101_adam_dec_inner_2_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp,assignvariableop_102_adam_dec_inner_2_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp.assignvariableop_103_adam_dec_inner_3_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp,assignvariableop_104_adam_dec_inner_3_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp/assignvariableop_105_adam_dec_middle_0_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp-assignvariableop_106_adam_dec_middle_0_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp/assignvariableop_107_adam_dec_middle_1_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp-assignvariableop_108_adam_dec_middle_1_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp/assignvariableop_109_adam_dec_middle_2_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp-assignvariableop_110_adam_dec_middle_2_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp/assignvariableop_111_adam_dec_middle_3_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp-assignvariableop_112_adam_dec_middle_3_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp.assignvariableop_113_adam_dec_outer_0_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp,assignvariableop_114_adam_dec_outer_0_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp.assignvariableop_115_adam_dec_outer_1_kernel_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp,assignvariableop_116_adam_dec_outer_1_bias_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp.assignvariableop_117_adam_dec_outer_2_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp,assignvariableop_118_adam_dec_outer_2_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp.assignvariableop_119_adam_dec_outer_3_kernel_mIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp,assignvariableop_120_adam_dec_outer_3_bias_mIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp-assignvariableop_121_adam_dec_output_kernel_mIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp+assignvariableop_122_adam_dec_output_bias_mIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp.assignvariableop_123_adam_enc_outer_0_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp,assignvariableop_124_adam_enc_outer_0_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp.assignvariableop_125_adam_enc_outer_1_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp,assignvariableop_126_adam_enc_outer_1_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp.assignvariableop_127_adam_enc_outer_2_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp,assignvariableop_128_adam_enc_outer_2_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp.assignvariableop_129_adam_enc_outer_3_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp,assignvariableop_130_adam_enc_outer_3_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp/assignvariableop_131_adam_enc_middle_0_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp-assignvariableop_132_adam_enc_middle_0_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp/assignvariableop_133_adam_enc_middle_1_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp-assignvariableop_134_adam_enc_middle_1_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOp/assignvariableop_135_adam_enc_middle_2_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOp-assignvariableop_136_adam_enc_middle_2_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOp/assignvariableop_137_adam_enc_middle_3_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138?
AssignVariableOp_138AssignVariableOp-assignvariableop_138_adam_enc_middle_3_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139?
AssignVariableOp_139AssignVariableOp.assignvariableop_139_adam_enc_inner_0_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140?
AssignVariableOp_140AssignVariableOp,assignvariableop_140_adam_enc_inner_0_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141?
AssignVariableOp_141AssignVariableOp.assignvariableop_141_adam_enc_inner_1_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142?
AssignVariableOp_142AssignVariableOp,assignvariableop_142_adam_enc_inner_1_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143?
AssignVariableOp_143AssignVariableOp.assignvariableop_143_adam_enc_inner_2_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144?
AssignVariableOp_144AssignVariableOp,assignvariableop_144_adam_enc_inner_2_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145?
AssignVariableOp_145AssignVariableOp.assignvariableop_145_adam_enc_inner_3_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146?
AssignVariableOp_146AssignVariableOp,assignvariableop_146_adam_enc_inner_3_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147?
AssignVariableOp_147AssignVariableOp,assignvariableop_147_adam_channel_0_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148?
AssignVariableOp_148AssignVariableOp*assignvariableop_148_adam_channel_0_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149?
AssignVariableOp_149AssignVariableOp,assignvariableop_149_adam_channel_1_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150?
AssignVariableOp_150AssignVariableOp*assignvariableop_150_adam_channel_1_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151?
AssignVariableOp_151AssignVariableOp,assignvariableop_151_adam_channel_2_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152?
AssignVariableOp_152AssignVariableOp*assignvariableop_152_adam_channel_2_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_152q
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:2
Identity_153?
AssignVariableOp_153AssignVariableOp,assignvariableop_153_adam_channel_3_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_153q
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:2
Identity_154?
AssignVariableOp_154AssignVariableOp*assignvariableop_154_adam_channel_3_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_154q
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:2
Identity_155?
AssignVariableOp_155AssignVariableOp.assignvariableop_155_adam_dec_inner_0_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_155q
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:2
Identity_156?
AssignVariableOp_156AssignVariableOp,assignvariableop_156_adam_dec_inner_0_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_156q
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:2
Identity_157?
AssignVariableOp_157AssignVariableOp.assignvariableop_157_adam_dec_inner_1_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_157q
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:2
Identity_158?
AssignVariableOp_158AssignVariableOp,assignvariableop_158_adam_dec_inner_1_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_158q
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:2
Identity_159?
AssignVariableOp_159AssignVariableOp.assignvariableop_159_adam_dec_inner_2_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159q
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:2
Identity_160?
AssignVariableOp_160AssignVariableOp,assignvariableop_160_adam_dec_inner_2_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_160q
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:2
Identity_161?
AssignVariableOp_161AssignVariableOp.assignvariableop_161_adam_dec_inner_3_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_161q
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:2
Identity_162?
AssignVariableOp_162AssignVariableOp,assignvariableop_162_adam_dec_inner_3_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_162q
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:2
Identity_163?
AssignVariableOp_163AssignVariableOp/assignvariableop_163_adam_dec_middle_0_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_163q
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:2
Identity_164?
AssignVariableOp_164AssignVariableOp-assignvariableop_164_adam_dec_middle_0_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_164q
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:2
Identity_165?
AssignVariableOp_165AssignVariableOp/assignvariableop_165_adam_dec_middle_1_kernel_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_165q
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:2
Identity_166?
AssignVariableOp_166AssignVariableOp-assignvariableop_166_adam_dec_middle_1_bias_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_166q
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:2
Identity_167?
AssignVariableOp_167AssignVariableOp/assignvariableop_167_adam_dec_middle_2_kernel_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_167q
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:2
Identity_168?
AssignVariableOp_168AssignVariableOp-assignvariableop_168_adam_dec_middle_2_bias_vIdentity_168:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_168q
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:2
Identity_169?
AssignVariableOp_169AssignVariableOp/assignvariableop_169_adam_dec_middle_3_kernel_vIdentity_169:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169q
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:2
Identity_170?
AssignVariableOp_170AssignVariableOp-assignvariableop_170_adam_dec_middle_3_bias_vIdentity_170:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_170q
Identity_171IdentityRestoreV2:tensors:171"/device:CPU:0*
T0*
_output_shapes
:2
Identity_171?
AssignVariableOp_171AssignVariableOp.assignvariableop_171_adam_dec_outer_0_kernel_vIdentity_171:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_171q
Identity_172IdentityRestoreV2:tensors:172"/device:CPU:0*
T0*
_output_shapes
:2
Identity_172?
AssignVariableOp_172AssignVariableOp,assignvariableop_172_adam_dec_outer_0_bias_vIdentity_172:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_172q
Identity_173IdentityRestoreV2:tensors:173"/device:CPU:0*
T0*
_output_shapes
:2
Identity_173?
AssignVariableOp_173AssignVariableOp.assignvariableop_173_adam_dec_outer_1_kernel_vIdentity_173:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_173q
Identity_174IdentityRestoreV2:tensors:174"/device:CPU:0*
T0*
_output_shapes
:2
Identity_174?
AssignVariableOp_174AssignVariableOp,assignvariableop_174_adam_dec_outer_1_bias_vIdentity_174:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_174q
Identity_175IdentityRestoreV2:tensors:175"/device:CPU:0*
T0*
_output_shapes
:2
Identity_175?
AssignVariableOp_175AssignVariableOp.assignvariableop_175_adam_dec_outer_2_kernel_vIdentity_175:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_175q
Identity_176IdentityRestoreV2:tensors:176"/device:CPU:0*
T0*
_output_shapes
:2
Identity_176?
AssignVariableOp_176AssignVariableOp,assignvariableop_176_adam_dec_outer_2_bias_vIdentity_176:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_176q
Identity_177IdentityRestoreV2:tensors:177"/device:CPU:0*
T0*
_output_shapes
:2
Identity_177?
AssignVariableOp_177AssignVariableOp.assignvariableop_177_adam_dec_outer_3_kernel_vIdentity_177:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_177q
Identity_178IdentityRestoreV2:tensors:178"/device:CPU:0*
T0*
_output_shapes
:2
Identity_178?
AssignVariableOp_178AssignVariableOp,assignvariableop_178_adam_dec_outer_3_bias_vIdentity_178:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_178q
Identity_179IdentityRestoreV2:tensors:179"/device:CPU:0*
T0*
_output_shapes
:2
Identity_179?
AssignVariableOp_179AssignVariableOp-assignvariableop_179_adam_dec_output_kernel_vIdentity_179:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179q
Identity_180IdentityRestoreV2:tensors:180"/device:CPU:0*
T0*
_output_shapes
:2
Identity_180?
AssignVariableOp_180AssignVariableOp+assignvariableop_180_adam_dec_output_bias_vIdentity_180:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1809
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp? 
Identity_181Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_181? 
Identity_182IdentityIdentity_181:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_182"%
identity_182Identity_182:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_170AssignVariableOp_1702,
AssignVariableOp_171AssignVariableOp_1712,
AssignVariableOp_172AssignVariableOp_1722,
AssignVariableOp_173AssignVariableOp_1732,
AssignVariableOp_174AssignVariableOp_1742,
AssignVariableOp_175AssignVariableOp_1752,
AssignVariableOp_176AssignVariableOp_1762,
AssignVariableOp_177AssignVariableOp_1772,
AssignVariableOp_178AssignVariableOp_1782,
AssignVariableOp_179AssignVariableOp_1792*
AssignVariableOp_18AssignVariableOp_182,
AssignVariableOp_180AssignVariableOp_1802*
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
?	
?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_309025

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
?h
?
C__inference_model_6_layer_call_and_return_conditional_losses_305112
encoder_input
enc_outer_3_305028
enc_outer_3_305030
enc_outer_2_305033
enc_outer_2_305035
enc_outer_1_305038
enc_outer_1_305040
enc_outer_0_305043
enc_outer_0_305045
enc_middle_3_305048
enc_middle_3_305050
enc_middle_2_305053
enc_middle_2_305055
enc_middle_1_305058
enc_middle_1_305060
enc_middle_0_305063
enc_middle_0_305065
enc_inner_3_305068
enc_inner_3_305070
enc_inner_2_305073
enc_inner_2_305075
enc_inner_1_305078
enc_inner_1_305080
enc_inner_0_305083
enc_inner_0_305085
channel_3_305088
channel_3_305090
channel_2_305093
channel_2_305095
channel_1_305098
channel_1_305100
channel_0_305103
channel_0_305105
identity

identity_1

identity_2

identity_3??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?!channel_2/StatefulPartitionedCall?!channel_3/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?#enc_inner_2/StatefulPartitionedCall?#enc_inner_3/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?$enc_middle_2/StatefulPartitionedCall?$enc_middle_3/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?#enc_outer_2/StatefulPartitionedCall?#enc_outer_3/StatefulPartitionedCall?
#enc_outer_3/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_3_305028enc_outer_3_305030*
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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_3046002%
#enc_outer_3/StatefulPartitionedCall?
#enc_outer_2/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_2_305033enc_outer_2_305035*
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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_3046272%
#enc_outer_2/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_1_305038enc_outer_1_305040*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_3046542%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_305043enc_outer_0_305045*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_3046812%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_3/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_3/StatefulPartitionedCall:output:0enc_middle_3_305048enc_middle_3_305050*
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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_3047082&
$enc_middle_3/StatefulPartitionedCall?
$enc_middle_2/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_2/StatefulPartitionedCall:output:0enc_middle_2_305053enc_middle_2_305055*
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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_3047352&
$enc_middle_2/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_305058enc_middle_1_305060*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_3047622&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_305063enc_middle_0_305065*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_3047892&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_3/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_3/StatefulPartitionedCall:output:0enc_inner_3_305068enc_inner_3_305070*
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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_3048162%
#enc_inner_3/StatefulPartitionedCall?
#enc_inner_2/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_2/StatefulPartitionedCall:output:0enc_inner_2_305073enc_inner_2_305075*
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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_3048432%
#enc_inner_2/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_305078enc_inner_1_305080*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_3048702%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_305083enc_inner_0_305085*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_3048972%
#enc_inner_0/StatefulPartitionedCall?
!channel_3/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_3/StatefulPartitionedCall:output:0channel_3_305088channel_3_305090*
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
E__inference_channel_3_layer_call_and_return_conditional_losses_3049242#
!channel_3/StatefulPartitionedCall?
!channel_2/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_2/StatefulPartitionedCall:output:0channel_2_305093channel_2_305095*
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
E__inference_channel_2_layer_call_and_return_conditional_losses_3049512#
!channel_2/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_305098channel_1_305100*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_3049782#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_305103channel_0_305105*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_3050052#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity*channel_2/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity*channel_3/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall"^channel_2/StatefulPartitionedCall"^channel_3/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall$^enc_inner_2/StatefulPartitionedCall$^enc_inner_3/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall%^enc_middle_2/StatefulPartitionedCall%^enc_middle_3/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall$^enc_outer_2/StatefulPartitionedCall$^enc_outer_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2F
!channel_2/StatefulPartitionedCall!channel_2/StatefulPartitionedCall2F
!channel_3/StatefulPartitionedCall!channel_3/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2J
#enc_inner_2/StatefulPartitionedCall#enc_inner_2/StatefulPartitionedCall2J
#enc_inner_3/StatefulPartitionedCall#enc_inner_3/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2L
$enc_middle_2/StatefulPartitionedCall$enc_middle_2/StatefulPartitionedCall2L
$enc_middle_3/StatefulPartitionedCall$enc_middle_3/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall2J
#enc_outer_2/StatefulPartitionedCall#enc_outer_2/StatefulPartitionedCall2J
#enc_outer_3/StatefulPartitionedCall#enc_outer_3/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?	
?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_304789

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
?
?
.__inference_autoencoder_3_layer_call_fn_306909
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

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56
identity??StatefulPartitionedCall?	
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
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_3067902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
E__inference_channel_1_layer_call_and_return_conditional_losses_304978

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
?
?
(__inference_model_7_layer_call_fn_308594
inputs_0
inputs_1
inputs_2
inputs_3
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

unknown_24
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3059522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::22
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
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?	
?
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_308705

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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_308665

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
?

I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306790
x
model_6_306668
model_6_306670
model_6_306672
model_6_306674
model_6_306676
model_6_306678
model_6_306680
model_6_306682
model_6_306684
model_6_306686
model_6_306688
model_6_306690
model_6_306692
model_6_306694
model_6_306696
model_6_306698
model_6_306700
model_6_306702
model_6_306704
model_6_306706
model_6_306708
model_6_306710
model_6_306712
model_6_306714
model_6_306716
model_6_306718
model_6_306720
model_6_306722
model_6_306724
model_6_306726
model_6_306728
model_6_306730
model_7_306736
model_7_306738
model_7_306740
model_7_306742
model_7_306744
model_7_306746
model_7_306748
model_7_306750
model_7_306752
model_7_306754
model_7_306756
model_7_306758
model_7_306760
model_7_306762
model_7_306764
model_7_306766
model_7_306768
model_7_306770
model_7_306772
model_7_306774
model_7_306776
model_7_306778
model_7_306780
model_7_306782
model_7_306784
model_7_306786
identity??model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallxmodel_6_306668model_6_306670model_6_306672model_6_306674model_6_306676model_6_306678model_6_306680model_6_306682model_6_306684model_6_306686model_6_306688model_6_306690model_6_306692model_6_306694model_6_306696model_6_306698model_6_306700model_6_306702model_6_306704model_6_306706model_6_306708model_6_306710model_6_306712model_6_306714model_6_306716model_6_306718model_6_306720model_6_306722model_6_306724model_6_306726model_6_306728model_6_306730*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????:?????????:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_3052022!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0(model_6/StatefulPartitionedCall:output:1(model_6/StatefulPartitionedCall:output:2(model_6/StatefulPartitionedCall:output:3model_7_306736model_7_306738model_7_306740model_7_306742model_7_306744model_7_306746model_7_306748model_7_306750model_7_306752model_7_306754model_7_306756model_7_306758model_7_306760model_7_306762model_7_306764model_7_306766model_7_306768model_7_306770model_7_306772model_7_306774model_7_306776model_7_306778model_7_306780model_7_306782model_7_306784model_7_306786*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_3059522!
model_7/StatefulPartitionedCall?
IdentityIdentity(model_7/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
.__inference_autoencoder_3_layer_call_fn_307946
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

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56
identity??StatefulPartitionedCall?	
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
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_3070362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
؆
?
C__inference_model_7_layer_call_and_return_conditional_losses_308434
inputs_0
inputs_1
inputs_2
inputs_3.
*dec_inner_3_matmul_readvariableop_resource/
+dec_inner_3_biasadd_readvariableop_resource.
*dec_inner_2_matmul_readvariableop_resource/
+dec_inner_2_biasadd_readvariableop_resource.
*dec_inner_1_matmul_readvariableop_resource/
+dec_inner_1_biasadd_readvariableop_resource.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_3_matmul_readvariableop_resource0
,dec_middle_3_biasadd_readvariableop_resource/
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
+dec_outer_2_biasadd_readvariableop_resource.
*dec_outer_3_matmul_readvariableop_resource/
+dec_outer_3_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?"dec_inner_1/BiasAdd/ReadVariableOp?!dec_inner_1/MatMul/ReadVariableOp?"dec_inner_2/BiasAdd/ReadVariableOp?!dec_inner_2/MatMul/ReadVariableOp?"dec_inner_3/BiasAdd/ReadVariableOp?!dec_inner_3/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?#dec_middle_1/BiasAdd/ReadVariableOp?"dec_middle_1/MatMul/ReadVariableOp?#dec_middle_2/BiasAdd/ReadVariableOp?"dec_middle_2/MatMul/ReadVariableOp?#dec_middle_3/BiasAdd/ReadVariableOp?"dec_middle_3/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?"dec_outer_1/BiasAdd/ReadVariableOp?!dec_outer_1/MatMul/ReadVariableOp?"dec_outer_2/BiasAdd/ReadVariableOp?!dec_outer_2/MatMul/ReadVariableOp?"dec_outer_3/BiasAdd/ReadVariableOp?!dec_outer_3/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
!dec_inner_3/MatMul/ReadVariableOpReadVariableOp*dec_inner_3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_3/MatMul/ReadVariableOp?
dec_inner_3/MatMulMatMulinputs_3)dec_inner_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_3/MatMul?
"dec_inner_3/BiasAdd/ReadVariableOpReadVariableOp+dec_inner_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"dec_inner_3/BiasAdd/ReadVariableOp?
dec_inner_3/BiasAddBiasAdddec_inner_3/MatMul:product:0*dec_inner_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dec_inner_3/BiasAdd|
dec_inner_3/ReluReludec_inner_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
dec_inner_3/Relu?
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
"dec_middle_3/MatMul/ReadVariableOpReadVariableOp+dec_middle_3_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02$
"dec_middle_3/MatMul/ReadVariableOp?
dec_middle_3/MatMulMatMuldec_inner_3/Relu:activations:0*dec_middle_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_3/MatMul?
#dec_middle_3/BiasAdd/ReadVariableOpReadVariableOp,dec_middle_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#dec_middle_3/BiasAdd/ReadVariableOp?
dec_middle_3/BiasAddBiasAdddec_middle_3/MatMul:product:0+dec_middle_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_middle_3/BiasAdd
dec_middle_3/ReluReludec_middle_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_middle_3/Relu?
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
dec_outer_2/Relu?
!dec_outer_3/MatMul/ReadVariableOpReadVariableOp*dec_outer_3_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02#
!dec_outer_3/MatMul/ReadVariableOp?
dec_outer_3/MatMulMatMuldec_middle_3/Relu:activations:0)dec_outer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_3/MatMul?
"dec_outer_3/BiasAdd/ReadVariableOpReadVariableOp+dec_outer_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dec_outer_3/BiasAdd/ReadVariableOp?
dec_outer_3/BiasAddBiasAdddec_outer_3/MatMul:product:0*dec_outer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dec_outer_3/BiasAdd|
dec_outer_3/ReluReludec_outer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dec_outer_3/Relut
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2dec_outer_0/Relu:activations:0dec_outer_1/Relu:activations:0dec_outer_2/Relu:activations:0dec_outer_3/Relu:activations:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.concat_2/concat:output:0(dec_output/MatMul/ReadVariableOp:value:0*
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
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp#^dec_inner_1/BiasAdd/ReadVariableOp"^dec_inner_1/MatMul/ReadVariableOp#^dec_inner_2/BiasAdd/ReadVariableOp"^dec_inner_2/MatMul/ReadVariableOp#^dec_inner_3/BiasAdd/ReadVariableOp"^dec_inner_3/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp$^dec_middle_1/BiasAdd/ReadVariableOp#^dec_middle_1/MatMul/ReadVariableOp$^dec_middle_2/BiasAdd/ReadVariableOp#^dec_middle_2/MatMul/ReadVariableOp$^dec_middle_3/BiasAdd/ReadVariableOp#^dec_middle_3/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp#^dec_outer_1/BiasAdd/ReadVariableOp"^dec_outer_1/MatMul/ReadVariableOp#^dec_outer_2/BiasAdd/ReadVariableOp"^dec_outer_2/MatMul/ReadVariableOp#^dec_outer_3/BiasAdd/ReadVariableOp"^dec_outer_3/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2H
"dec_inner_1/BiasAdd/ReadVariableOp"dec_inner_1/BiasAdd/ReadVariableOp2F
!dec_inner_1/MatMul/ReadVariableOp!dec_inner_1/MatMul/ReadVariableOp2H
"dec_inner_2/BiasAdd/ReadVariableOp"dec_inner_2/BiasAdd/ReadVariableOp2F
!dec_inner_2/MatMul/ReadVariableOp!dec_inner_2/MatMul/ReadVariableOp2H
"dec_inner_3/BiasAdd/ReadVariableOp"dec_inner_3/BiasAdd/ReadVariableOp2F
!dec_inner_3/MatMul/ReadVariableOp!dec_inner_3/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2J
#dec_middle_1/BiasAdd/ReadVariableOp#dec_middle_1/BiasAdd/ReadVariableOp2H
"dec_middle_1/MatMul/ReadVariableOp"dec_middle_1/MatMul/ReadVariableOp2J
#dec_middle_2/BiasAdd/ReadVariableOp#dec_middle_2/BiasAdd/ReadVariableOp2H
"dec_middle_2/MatMul/ReadVariableOp"dec_middle_2/MatMul/ReadVariableOp2J
#dec_middle_3/BiasAdd/ReadVariableOp#dec_middle_3/BiasAdd/ReadVariableOp2H
"dec_middle_3/MatMul/ReadVariableOp"dec_middle_3/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2H
"dec_outer_1/BiasAdd/ReadVariableOp"dec_outer_1/BiasAdd/ReadVariableOp2F
!dec_outer_1/MatMul/ReadVariableOp!dec_outer_1/MatMul/ReadVariableOp2H
"dec_outer_2/BiasAdd/ReadVariableOp"dec_outer_2/BiasAdd/ReadVariableOp2F
!dec_outer_2/MatMul/ReadVariableOp!dec_outer_2/MatMul/ReadVariableOp2H
"dec_outer_3/BiasAdd/ReadVariableOp"dec_outer_3/BiasAdd/ReadVariableOp2F
!dec_outer_3/MatMul/ReadVariableOp!dec_outer_3/MatMul/ReadVariableOp2F
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
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?
?
,__inference_dec_inner_2_layer_call_fn_309034

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
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_3054822
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_309085

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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_309105

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
,__inference_dec_outer_2_layer_call_fn_309194

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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_3057252
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
?

*__inference_channel_1_layer_call_fn_308934

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
E__inference_channel_1_layer_call_and_return_conditional_losses_3049782
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
?	
?
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_308865

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
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
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
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ǎ
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
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_3", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_1", "inbound_nodes": [[["enc_outer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_2", "inbound_nodes": [[["enc_outer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_3", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_3", "inbound_nodes": [[["enc_outer_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_1", "inbound_nodes": [[["enc_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_2", "inbound_nodes": [[["enc_middle_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_3", "inbound_nodes": [[["enc_middle_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_1", "inbound_nodes": [[["enc_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_2", "inbound_nodes": [[["enc_inner_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_3", "inbound_nodes": [[["enc_inner_3", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0], ["channel_1", 0, 0], ["channel_2", 0, 0], ["channel_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 784]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_3", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_1", "inbound_nodes": [[["enc_outer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_2", "inbound_nodes": [[["enc_outer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_3", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_3", "inbound_nodes": [[["enc_outer_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_1", "inbound_nodes": [[["enc_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_2", "inbound_nodes": [[["enc_middle_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_3", "inbound_nodes": [[["enc_middle_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_1", "inbound_nodes": [[["enc_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_2", "inbound_nodes": [[["enc_inner_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_3", "inbound_nodes": [[["enc_inner_3", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0], ["channel_1", 0, 0], ["channel_2", 0, 0], ["channel_3", 0, 0]]}}}
Ԍ
layer-0
layer-1
 layer-2
!layer-3
"layer_with_weights-0
"layer-4
#layer_with_weights-1
#layer-5
$layer_with_weights-2
$layer-6
%layer_with_weights-3
%layer-7
&layer_with_weights-4
&layer-8
'layer_with_weights-5
'layer-9
(layer_with_weights-6
(layer-10
)layer_with_weights-7
)layer-11
*layer_with_weights-8
*layer-12
+layer_with_weights-9
+layer-13
,layer_with_weights-10
,layer-14
-layer_with_weights-11
-layer-15
.layer-16
/layer_with_weights-12
/layer-17
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_networkކ{"class_name": "Functional", "name": "model_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}, "name": "decoder_input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_2"}, "name": "decoder_input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_3"}, "name": "decoder_input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_1", "inbound_nodes": [[["decoder_input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_2", "inbound_nodes": [[["decoder_input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_3", "inbound_nodes": [[["decoder_input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_1", "inbound_nodes": [[["dec_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_2", "inbound_nodes": [[["dec_inner_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_3", "inbound_nodes": [[["dec_inner_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_1", "inbound_nodes": [[["dec_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_2", "inbound_nodes": [[["dec_middle_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_3", "inbound_nodes": [[["dec_middle_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_2", "inbound_nodes": [[["dec_outer_0", 0, 0, {"axis": 1}], ["dec_outer_1", 0, 0, {"axis": 1}], ["dec_outer_2", 0, 0, {"axis": 1}], ["dec_outer_3", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.concat_2", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0], ["decoder_input_1", 0, 0], ["decoder_input_2", 0, 0], ["decoder_input_3", 0, 0]], "output_layers": [["dec_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}, "name": "decoder_input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_2"}, "name": "decoder_input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_3"}, "name": "decoder_input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_1", "inbound_nodes": [[["decoder_input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_2", "inbound_nodes": [[["decoder_input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_3", "inbound_nodes": [[["decoder_input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_1", "inbound_nodes": [[["dec_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_2", "inbound_nodes": [[["dec_inner_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_3", "inbound_nodes": [[["dec_inner_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_1", "inbound_nodes": [[["dec_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_2", "inbound_nodes": [[["dec_middle_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_3", "inbound_nodes": [[["dec_middle_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_2", "inbound_nodes": [[["dec_outer_0", 0, 0, {"axis": 1}], ["dec_outer_1", 0, 0, {"axis": 1}], ["dec_outer_2", 0, 0, {"axis": 1}], ["dec_outer_3", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.concat_2", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0], ["decoder_input_1", 0, 0], ["decoder_input_2", 0, 0], ["decoder_input_3", 0, 0]], "output_layers": [["dec_output", 0, 0]]}}}
?	
4iter

5beta_1

6beta_2
	7decay
8learning_rate9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?"
	optimizer
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
[34
\35
]36
^37
_38
`39
a40
b41
c42
d43
e44
f45
g46
h47
i48
j49
k50
l51
m52
n53
o54
p55
q56
r57"
trackable_list_wrapper
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
[34
\35
]36
^37
_38
`39
a40
b41
c42
d43
e44
f45
g46
h47
i48
j49
k50
l51
m52
n53
o54
p55
q56
r57"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
smetrics
tnon_trainable_variables
ulayer_regularization_losses
trainable_variables
vlayer_metrics
regularization_losses

wlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
?

9kernel
:bias
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

;kernel
<bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

=kernel
>bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

?kernel
@bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Ckernel
Dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Gkernel
Hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_3", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Ikernel
Jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

Kkernel
Lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

Mkernel
Nbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

Qkernel
Rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Skernel
Tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Ukernel
Vbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Wkernel
Xbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31"
trackable_list_wrapper
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
trainable_variables
?layer_metrics
regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_2"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_3"}}
?

Ykernel
Zbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

[kernel
\bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

]kernel
^bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

_kernel
`bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

akernel
bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

ckernel
dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

ekernel
fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

gkernel
hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

ikernel
jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

kkernel
lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

mkernel
nbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

okernel
pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_3", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}}
?

qkernel
rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 240}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 240]}}
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25"
trackable_list_wrapper
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
1trainable_variables
?layer_metrics
2regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
%:#	?<2enc_outer_3/kernel
:<2enc_outer_3/bias
%:#<22enc_middle_0/kernel
:22enc_middle_0/bias
%:#<22enc_middle_1/kernel
:22enc_middle_1/bias
%:#<22enc_middle_2/kernel
:22enc_middle_2/bias
%:#<22enc_middle_3/kernel
:22enc_middle_3/bias
$:"2(2enc_inner_0/kernel
:(2enc_inner_0/bias
$:"2(2enc_inner_1/kernel
:(2enc_inner_1/bias
$:"2(2enc_inner_2/kernel
:(2enc_inner_2/bias
$:"2(2enc_inner_3/kernel
:(2enc_inner_3/bias
": (2channel_0/kernel
:2channel_0/bias
": (2channel_1/kernel
:2channel_1/bias
": (2channel_2/kernel
:2channel_2/bias
": (2channel_3/kernel
:2channel_3/bias
$:"(2dec_inner_0/kernel
:(2dec_inner_0/bias
$:"(2dec_inner_1/kernel
:(2dec_inner_1/bias
$:"(2dec_inner_2/kernel
:(2dec_inner_2/bias
$:"(2dec_inner_3/kernel
:(2dec_inner_3/bias
%:#(<2dec_middle_0/kernel
:<2dec_middle_0/bias
%:#(<2dec_middle_1/kernel
:<2dec_middle_1/bias
%:#(<2dec_middle_2/kernel
:<2dec_middle_2/bias
%:#(<2dec_middle_3/kernel
:<2dec_middle_3/bias
$:"<<2dec_outer_0/kernel
:<2dec_outer_0/bias
$:"<<2dec_outer_1/kernel
:<2dec_outer_1/bias
$:"<<2dec_outer_2/kernel
:<2dec_outer_2/bias
$:"<<2dec_outer_3/kernel
:<2dec_outer_3/bias
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
xtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
y	variables
?layer_metrics
zregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
|trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
}	variables
?layer_metrics
~regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
13
14
15
16"
trackable_list_wrapper
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?	variables
?layer_metrics
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
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
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(	?<2Adam/enc_outer_0/kernel/m
#:!<2Adam/enc_outer_0/bias/m
*:(	?<2Adam/enc_outer_1/kernel/m
#:!<2Adam/enc_outer_1/bias/m
*:(	?<2Adam/enc_outer_2/kernel/m
#:!<2Adam/enc_outer_2/bias/m
*:(	?<2Adam/enc_outer_3/kernel/m
#:!<2Adam/enc_outer_3/bias/m
*:(<22Adam/enc_middle_0/kernel/m
$:"22Adam/enc_middle_0/bias/m
*:(<22Adam/enc_middle_1/kernel/m
$:"22Adam/enc_middle_1/bias/m
*:(<22Adam/enc_middle_2/kernel/m
$:"22Adam/enc_middle_2/bias/m
*:(<22Adam/enc_middle_3/kernel/m
$:"22Adam/enc_middle_3/bias/m
):'2(2Adam/enc_inner_0/kernel/m
#:!(2Adam/enc_inner_0/bias/m
):'2(2Adam/enc_inner_1/kernel/m
#:!(2Adam/enc_inner_1/bias/m
):'2(2Adam/enc_inner_2/kernel/m
#:!(2Adam/enc_inner_2/bias/m
):'2(2Adam/enc_inner_3/kernel/m
#:!(2Adam/enc_inner_3/bias/m
':%(2Adam/channel_0/kernel/m
!:2Adam/channel_0/bias/m
':%(2Adam/channel_1/kernel/m
!:2Adam/channel_1/bias/m
':%(2Adam/channel_2/kernel/m
!:2Adam/channel_2/bias/m
':%(2Adam/channel_3/kernel/m
!:2Adam/channel_3/bias/m
):'(2Adam/dec_inner_0/kernel/m
#:!(2Adam/dec_inner_0/bias/m
):'(2Adam/dec_inner_1/kernel/m
#:!(2Adam/dec_inner_1/bias/m
):'(2Adam/dec_inner_2/kernel/m
#:!(2Adam/dec_inner_2/bias/m
):'(2Adam/dec_inner_3/kernel/m
#:!(2Adam/dec_inner_3/bias/m
*:((<2Adam/dec_middle_0/kernel/m
$:"<2Adam/dec_middle_0/bias/m
*:((<2Adam/dec_middle_1/kernel/m
$:"<2Adam/dec_middle_1/bias/m
*:((<2Adam/dec_middle_2/kernel/m
$:"<2Adam/dec_middle_2/bias/m
*:((<2Adam/dec_middle_3/kernel/m
$:"<2Adam/dec_middle_3/bias/m
):'<<2Adam/dec_outer_0/kernel/m
#:!<2Adam/dec_outer_0/bias/m
):'<<2Adam/dec_outer_1/kernel/m
#:!<2Adam/dec_outer_1/bias/m
):'<<2Adam/dec_outer_2/kernel/m
#:!<2Adam/dec_outer_2/bias/m
):'<<2Adam/dec_outer_3/kernel/m
#:!<2Adam/dec_outer_3/bias/m
*:(
??2Adam/dec_output/kernel/m
#:!?2Adam/dec_output/bias/m
*:(	?<2Adam/enc_outer_0/kernel/v
#:!<2Adam/enc_outer_0/bias/v
*:(	?<2Adam/enc_outer_1/kernel/v
#:!<2Adam/enc_outer_1/bias/v
*:(	?<2Adam/enc_outer_2/kernel/v
#:!<2Adam/enc_outer_2/bias/v
*:(	?<2Adam/enc_outer_3/kernel/v
#:!<2Adam/enc_outer_3/bias/v
*:(<22Adam/enc_middle_0/kernel/v
$:"22Adam/enc_middle_0/bias/v
*:(<22Adam/enc_middle_1/kernel/v
$:"22Adam/enc_middle_1/bias/v
*:(<22Adam/enc_middle_2/kernel/v
$:"22Adam/enc_middle_2/bias/v
*:(<22Adam/enc_middle_3/kernel/v
$:"22Adam/enc_middle_3/bias/v
):'2(2Adam/enc_inner_0/kernel/v
#:!(2Adam/enc_inner_0/bias/v
):'2(2Adam/enc_inner_1/kernel/v
#:!(2Adam/enc_inner_1/bias/v
):'2(2Adam/enc_inner_2/kernel/v
#:!(2Adam/enc_inner_2/bias/v
):'2(2Adam/enc_inner_3/kernel/v
#:!(2Adam/enc_inner_3/bias/v
':%(2Adam/channel_0/kernel/v
!:2Adam/channel_0/bias/v
':%(2Adam/channel_1/kernel/v
!:2Adam/channel_1/bias/v
':%(2Adam/channel_2/kernel/v
!:2Adam/channel_2/bias/v
':%(2Adam/channel_3/kernel/v
!:2Adam/channel_3/bias/v
):'(2Adam/dec_inner_0/kernel/v
#:!(2Adam/dec_inner_0/bias/v
):'(2Adam/dec_inner_1/kernel/v
#:!(2Adam/dec_inner_1/bias/v
):'(2Adam/dec_inner_2/kernel/v
#:!(2Adam/dec_inner_2/bias/v
):'(2Adam/dec_inner_3/kernel/v
#:!(2Adam/dec_inner_3/bias/v
*:((<2Adam/dec_middle_0/kernel/v
$:"<2Adam/dec_middle_0/bias/v
*:((<2Adam/dec_middle_1/kernel/v
$:"<2Adam/dec_middle_1/bias/v
*:((<2Adam/dec_middle_2/kernel/v
$:"<2Adam/dec_middle_2/bias/v
*:((<2Adam/dec_middle_3/kernel/v
$:"<2Adam/dec_middle_3/bias/v
):'<<2Adam/dec_outer_0/kernel/v
#:!<2Adam/dec_outer_0/bias/v
):'<<2Adam/dec_outer_1/kernel/v
#:!<2Adam/dec_outer_1/bias/v
):'<<2Adam/dec_outer_2/kernel/v
#:!<2Adam/dec_outer_2/bias/v
):'<<2Adam/dec_outer_3/kernel/v
#:!<2Adam/dec_outer_3/bias/v
*:(
??2Adam/dec_output/kernel/v
#:!?2Adam/dec_output/bias/v
?2?
!__inference__wrapped_model_304585?
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
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307704
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307495
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306537
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306662?
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
.__inference_autoencoder_3_layer_call_fn_307155
.__inference_autoencoder_3_layer_call_fn_306909
.__inference_autoencoder_3_layer_call_fn_307825
.__inference_autoencoder_3_layer_call_fn_307946?
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
C__inference_model_6_layer_call_and_return_conditional_losses_308065
C__inference_model_6_layer_call_and_return_conditional_losses_305025
C__inference_model_6_layer_call_and_return_conditional_losses_305112
C__inference_model_6_layer_call_and_return_conditional_losses_308184?
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
(__inference_model_6_layer_call_fn_305437
(__inference_model_6_layer_call_fn_308259
(__inference_model_6_layer_call_fn_308334
(__inference_model_6_layer_call_fn_305275?
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
C__inference_model_7_layer_call_and_return_conditional_losses_305872
C__inference_model_7_layer_call_and_return_conditional_losses_305798
C__inference_model_7_layer_call_and_return_conditional_losses_308434
C__inference_model_7_layer_call_and_return_conditional_losses_308534?
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
(__inference_model_7_layer_call_fn_306141
(__inference_model_7_layer_call_fn_308654
(__inference_model_7_layer_call_fn_308594
(__inference_model_7_layer_call_fn_306007?
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
$__inference_signature_wrapper_307286input_1"?
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_308665?
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
,__inference_enc_outer_0_layer_call_fn_308674?
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_308685?
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
,__inference_enc_outer_1_layer_call_fn_308694?
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
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_308705?
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
,__inference_enc_outer_2_layer_call_fn_308714?
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
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_308725?
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
,__inference_enc_outer_3_layer_call_fn_308734?
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_308745?
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
-__inference_enc_middle_0_layer_call_fn_308754?
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_308765?
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
-__inference_enc_middle_1_layer_call_fn_308774?
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
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_308785?
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
-__inference_enc_middle_2_layer_call_fn_308794?
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
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_308805?
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
-__inference_enc_middle_3_layer_call_fn_308814?
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_308825?
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
,__inference_enc_inner_0_layer_call_fn_308834?
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_308845?
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
,__inference_enc_inner_1_layer_call_fn_308854?
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
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_308865?
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
,__inference_enc_inner_2_layer_call_fn_308874?
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
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_308885?
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
,__inference_enc_inner_3_layer_call_fn_308894?
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
E__inference_channel_0_layer_call_and_return_conditional_losses_308905?
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
*__inference_channel_0_layer_call_fn_308914?
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
E__inference_channel_1_layer_call_and_return_conditional_losses_308925?
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
*__inference_channel_1_layer_call_fn_308934?
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
E__inference_channel_2_layer_call_and_return_conditional_losses_308945?
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
*__inference_channel_2_layer_call_fn_308954?
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
E__inference_channel_3_layer_call_and_return_conditional_losses_308965?
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
*__inference_channel_3_layer_call_fn_308974?
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_308985?
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
,__inference_dec_inner_0_layer_call_fn_308994?
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_309005?
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
,__inference_dec_inner_1_layer_call_fn_309014?
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
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_309025?
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
,__inference_dec_inner_2_layer_call_fn_309034?
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
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_309045?
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
,__inference_dec_inner_3_layer_call_fn_309054?
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_309065?
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
-__inference_dec_middle_0_layer_call_fn_309074?
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_309085?
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
-__inference_dec_middle_1_layer_call_fn_309094?
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
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_309105?
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
-__inference_dec_middle_2_layer_call_fn_309114?
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
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_309125?
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
-__inference_dec_middle_3_layer_call_fn_309134?
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_309145?
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
,__inference_dec_outer_0_layer_call_fn_309154?
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_309165?
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
,__inference_dec_outer_1_layer_call_fn_309174?
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
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_309185?
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
,__inference_dec_outer_2_layer_call_fn_309194?
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
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_309205?
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
,__inference_dec_outer_3_layer_call_fn_309214?
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
F__inference_dec_output_layer_call_and_return_conditional_losses_309225?
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
+__inference_dec_output_layer_call_fn_309234?
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
!__inference__wrapped_model_304585?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqr1?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306537?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqrA?>
'?$
"?
input_1??????????
?

trainingp"&?#
?
0??????????
? ?
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_306662?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqrA?>
'?$
"?
input_1??????????
?

trainingp "&?#
?
0??????????
? ?
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307495?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqr;?8
!?
?
x??????????
?

trainingp"&?#
?
0??????????
? ?
I__inference_autoencoder_3_layer_call_and_return_conditional_losses_307704?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqr;?8
!?
?
x??????????
?

trainingp "&?#
?
0??????????
? ?
.__inference_autoencoder_3_layer_call_fn_306909?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqrA?>
'?$
"?
input_1??????????
?

trainingp"????????????
.__inference_autoencoder_3_layer_call_fn_307155?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqrA?>
'?$
"?
input_1??????????
?

trainingp "????????????
.__inference_autoencoder_3_layer_call_fn_307825?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqr;?8
!?
?
x??????????
?

trainingp"????????????
.__inference_autoencoder_3_layer_call_fn_307946?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqr;?8
!?
?
x??????????
?

trainingp "????????????
E__inference_channel_0_layer_call_and_return_conditional_losses_308905\QR/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_0_layer_call_fn_308914OQR/?,
%?"
 ?
inputs?????????(
? "???????????
E__inference_channel_1_layer_call_and_return_conditional_losses_308925\ST/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_1_layer_call_fn_308934OST/?,
%?"
 ?
inputs?????????(
? "???????????
E__inference_channel_2_layer_call_and_return_conditional_losses_308945\UV/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_2_layer_call_fn_308954OUV/?,
%?"
 ?
inputs?????????(
? "???????????
E__inference_channel_3_layer_call_and_return_conditional_losses_308965\WX/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_3_layer_call_fn_308974OWX/?,
%?"
 ?
inputs?????????(
? "???????????
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_308985\YZ/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_0_layer_call_fn_308994OYZ/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_309005\[\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_1_layer_call_fn_309014O[\/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_inner_2_layer_call_and_return_conditional_losses_309025\]^/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_2_layer_call_fn_309034O]^/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_inner_3_layer_call_and_return_conditional_losses_309045\_`/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_3_layer_call_fn_309054O_`/?,
%?"
 ?
inputs?????????
? "??????????(?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_309065\ab/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_0_layer_call_fn_309074Oab/?,
%?"
 ?
inputs?????????(
? "??????????<?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_309085\cd/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_1_layer_call_fn_309094Ocd/?,
%?"
 ?
inputs?????????(
? "??????????<?
H__inference_dec_middle_2_layer_call_and_return_conditional_losses_309105\ef/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_2_layer_call_fn_309114Oef/?,
%?"
 ?
inputs?????????(
? "??????????<?
H__inference_dec_middle_3_layer_call_and_return_conditional_losses_309125\gh/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_3_layer_call_fn_309134Ogh/?,
%?"
 ?
inputs?????????(
? "??????????<?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_309145\ij/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_0_layer_call_fn_309154Oij/?,
%?"
 ?
inputs?????????<
? "??????????<?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_309165\kl/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_1_layer_call_fn_309174Okl/?,
%?"
 ?
inputs?????????<
? "??????????<?
G__inference_dec_outer_2_layer_call_and_return_conditional_losses_309185\mn/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_2_layer_call_fn_309194Omn/?,
%?"
 ?
inputs?????????<
? "??????????<?
G__inference_dec_outer_3_layer_call_and_return_conditional_losses_309205\op/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_3_layer_call_fn_309214Oop/?,
%?"
 ?
inputs?????????<
? "??????????<?
F__inference_dec_output_layer_call_and_return_conditional_losses_309225^qr0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dec_output_layer_call_fn_309234Qqr0?-
&?#
!?
inputs??????????
? "????????????
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_308825\IJ/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_0_layer_call_fn_308834OIJ/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_308845\KL/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_1_layer_call_fn_308854OKL/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_inner_2_layer_call_and_return_conditional_losses_308865\MN/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_2_layer_call_fn_308874OMN/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_inner_3_layer_call_and_return_conditional_losses_308885\OP/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_3_layer_call_fn_308894OOP/?,
%?"
 ?
inputs?????????2
? "??????????(?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_308745\AB/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_0_layer_call_fn_308754OAB/?,
%?"
 ?
inputs?????????<
? "??????????2?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_308765\CD/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_1_layer_call_fn_308774OCD/?,
%?"
 ?
inputs?????????<
? "??????????2?
H__inference_enc_middle_2_layer_call_and_return_conditional_losses_308785\EF/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_2_layer_call_fn_308794OEF/?,
%?"
 ?
inputs?????????<
? "??????????2?
H__inference_enc_middle_3_layer_call_and_return_conditional_losses_308805\GH/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_3_layer_call_fn_308814OGH/?,
%?"
 ?
inputs?????????<
? "??????????2?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_308665]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_0_layer_call_fn_308674P9:0?-
&?#
!?
inputs??????????
? "??????????<?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_308685];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_1_layer_call_fn_308694P;<0?-
&?#
!?
inputs??????????
? "??????????<?
G__inference_enc_outer_2_layer_call_and_return_conditional_losses_308705]=>0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_2_layer_call_fn_308714P=>0?-
&?#
!?
inputs??????????
? "??????????<?
G__inference_enc_outer_3_layer_call_and_return_conditional_losses_308725]?@0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_3_layer_call_fn_308734P?@0?-
&?#
!?
inputs??????????
? "??????????<?
C__inference_model_6_layer_call_and_return_conditional_losses_305025? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR??<
5?2
(?%
encoder_input??????????
p

 
? "???
?|
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_305112? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR??<
5?2
(?%
encoder_input??????????
p 

 
? "???
?|
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_308065? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR8?5
.?+
!?
inputs??????????
p

 
? "???
?|
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
? ?
C__inference_model_6_layer_call_and_return_conditional_losses_308184? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR8?5
.?+
!?
inputs??????????
p 

 
? "???
?|
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
? ?
(__inference_model_6_layer_call_fn_305275? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR??<
5?2
(?%
encoder_input??????????
p

 
? "w?t
?
0?????????
?
1?????????
?
2?????????
?
3??????????
(__inference_model_6_layer_call_fn_305437? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR??<
5?2
(?%
encoder_input??????????
p 

 
? "w?t
?
0?????????
?
1?????????
?
2?????????
?
3??????????
(__inference_model_6_layer_call_fn_308259? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR8?5
.?+
!?
inputs??????????
p

 
? "w?t
?
0?????????
?
1?????????
?
2?????????
?
3??????????
(__inference_model_6_layer_call_fn_308334? ?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR8?5
.?+
!?
inputs??????????
p 

 
? "w?t
?
0?????????
?
1?????????
?
2?????????
?
3??????????
C__inference_model_7_layer_call_and_return_conditional_losses_305798?_`]^[\YZghefcdabijklmnopqr???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
)?&
decoder_input_3?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_305872?_`]^[\YZghefcdabijklmnopqr???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
)?&
decoder_input_3?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_308434?_`]^[\YZghefcdabijklmnopqr???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_308534?_`]^[\YZghefcdabijklmnopqr???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
p 

 
? "&?#
?
0??????????
? ?
(__inference_model_7_layer_call_fn_306007?_`]^[\YZghefcdabijklmnopqr???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
)?&
decoder_input_3?????????
p

 
? "????????????
(__inference_model_7_layer_call_fn_306141?_`]^[\YZghefcdabijklmnopqr???
???
???
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
)?&
decoder_input_2?????????
)?&
decoder_input_3?????????
p 

 
? "????????????
(__inference_model_7_layer_call_fn_308594?_`]^[\YZghefcdabijklmnopqr???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
p

 
? "????????????
(__inference_model_7_layer_call_fn_308654?_`]^[\YZghefcdabijklmnopqr???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
p 

 
? "????????????
$__inference_signature_wrapper_307286?:?@=>;<9:GHEFCDABOPMNKLIJWXUVSTQR_`]^[\YZghefcdabijklmnopqr<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????