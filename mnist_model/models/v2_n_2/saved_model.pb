??
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
 ?"serve*2.4.12unknown8??
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

dec_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x?*"
shared_namedec_output/kernel
x
%dec_output/kernel/Read/ReadVariableOpReadVariableOpdec_output/kernel*
_output_shapes
:	x?*
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
Adam/dec_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x?*)
shared_nameAdam/dec_output/kernel/m
?
,Adam/dec_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_output/kernel/m*
_output_shapes
:	x?*
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
Adam/dec_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x?*)
shared_nameAdam/dec_output/kernel/v
?
,Adam/dec_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_output/kernel/v*
_output_shapes
:	x?*
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
?
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
	variables
trainable_variables
regularization_losses
	keras_api
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
layer_with_weights-6
layer-9
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?
$iter

%beta_1

&beta_2
	'decay
(learning_rate)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29
 
?
	variables
Gmetrics
Hnon_trainable_variables
Ilayer_regularization_losses
trainable_variables
Jlayer_metrics
regularization_losses

Klayers
 
 
h

)kernel
*bias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
h

+kernel
,bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
h

-kernel
.bias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
h

/kernel
0bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
h

1kernel
2bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
h

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
h

5kernel
6bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
h

7kernel
8bias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
 
?
	variables
lmetrics
mnon_trainable_variables
nlayer_regularization_losses
trainable_variables
olayer_metrics
regularization_losses

players
 
 
h

9kernel
:bias
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
h

;kernel
<bias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
h

=kernel
>bias
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
i

?kernel
@bias
}trainable_variables
~	variables
regularization_losses
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

?	keras_api
l

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
f
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
f
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
 
?
 	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
!trainable_variables
?layer_metrics
"regularization_losses
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
OM
VARIABLE_VALUEenc_middle_0/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_middle_0/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_middle_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_middle_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_inner_0/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEenc_inner_0/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEenc_inner_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_inner_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_0/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_0/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEchannel_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEchannel_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_0/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_0/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_inner_1/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_inner_1/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_0/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_0/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_1/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_1/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_0/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_0/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_1/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_1/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_output/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdec_output/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE

?0
 
 
 

0
1

)0
*1

)0
*1
 
?
Ltrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
M	variables
?layer_metrics
Nregularization_losses
?layers

+0
,1

+0
,1
 
?
Ptrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
Q	variables
?layer_metrics
Rregularization_losses
?layers

-0
.1

-0
.1
 
?
Ttrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
U	variables
?layer_metrics
Vregularization_losses
?layers

/0
01

/0
01
 
?
Xtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
Y	variables
?layer_metrics
Zregularization_losses
?layers

10
21

10
21
 
?
\trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
]	variables
?layer_metrics
^regularization_losses
?layers

30
41

30
41
 
?
`trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
a	variables
?layer_metrics
bregularization_losses
?layers

50
61

50
61
 
?
dtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
e	variables
?layer_metrics
fregularization_losses
?layers

70
81

70
81
 
?
htrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
i	variables
?layer_metrics
jregularization_losses
?layers
 
 
 
 
?
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

90
:1

90
:1
 
?
qtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
r	variables
?layer_metrics
sregularization_losses
?layers

;0
<1

;0
<1
 
?
utrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
v	variables
?layer_metrics
wregularization_losses
?layers

=0
>1

=0
>1
 
?
ytrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
z	variables
?layer_metrics
{regularization_losses
?layers

?0
@1

?0
@1
 
?
}trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
~	variables
?layer_metrics
regularization_losses
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
 
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
 
 
 
 
F
0
1
2
3
4
5
6
7
8
9
8

?total

?count
?	variables
?	keras_api
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
?0
?1

?	variables
qo
VARIABLE_VALUEAdam/enc_outer_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_inner_0/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_inner_0/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_1/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_1/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_0/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_0/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_1/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_1/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_0/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_0/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_1/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_1/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_1/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_1/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_1/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_1/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_inner_0/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_inner_0/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_inner_1/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_inner_1/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_0/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_0/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/channel_1/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/channel_1/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_0/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_0/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_inner_1/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_inner_1/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_1/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_1/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_1/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_1/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1enc_outer_1/kernelenc_outer_1/biasenc_outer_0/kernelenc_outer_0/biasenc_middle_1/kernelenc_middle_1/biasenc_middle_0/kernelenc_middle_0/biasenc_inner_1/kernelenc_inner_1/biasenc_inner_0/kernelenc_inner_0/biaschannel_1/kernelchannel_1/biaschannel_0/kernelchannel_0/biasdec_inner_1/kerneldec_inner_1/biasdec_inner_0/kerneldec_inner_0/biasdec_middle_1/kerneldec_middle_1/biasdec_middle_0/kerneldec_middle_0/biasdec_outer_0/kerneldec_outer_0/biasdec_outer_1/kerneldec_outer_1/biasdec_output/kerneldec_output/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_168134
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?#
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp&enc_outer_0/kernel/Read/ReadVariableOp$enc_outer_0/bias/Read/ReadVariableOp&enc_outer_1/kernel/Read/ReadVariableOp$enc_outer_1/bias/Read/ReadVariableOp'enc_middle_0/kernel/Read/ReadVariableOp%enc_middle_0/bias/Read/ReadVariableOp'enc_middle_1/kernel/Read/ReadVariableOp%enc_middle_1/bias/Read/ReadVariableOp&enc_inner_0/kernel/Read/ReadVariableOp$enc_inner_0/bias/Read/ReadVariableOp&enc_inner_1/kernel/Read/ReadVariableOp$enc_inner_1/bias/Read/ReadVariableOp$channel_0/kernel/Read/ReadVariableOp"channel_0/bias/Read/ReadVariableOp$channel_1/kernel/Read/ReadVariableOp"channel_1/bias/Read/ReadVariableOp&dec_inner_0/kernel/Read/ReadVariableOp$dec_inner_0/bias/Read/ReadVariableOp&dec_inner_1/kernel/Read/ReadVariableOp$dec_inner_1/bias/Read/ReadVariableOp'dec_middle_0/kernel/Read/ReadVariableOp%dec_middle_0/bias/Read/ReadVariableOp'dec_middle_1/kernel/Read/ReadVariableOp%dec_middle_1/bias/Read/ReadVariableOp&dec_outer_0/kernel/Read/ReadVariableOp$dec_outer_0/bias/Read/ReadVariableOp&dec_outer_1/kernel/Read/ReadVariableOp$dec_outer_1/bias/Read/ReadVariableOp%dec_output/kernel/Read/ReadVariableOp#dec_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/enc_outer_0/kernel/m/Read/ReadVariableOp+Adam/enc_outer_0/bias/m/Read/ReadVariableOp-Adam/enc_outer_1/kernel/m/Read/ReadVariableOp+Adam/enc_outer_1/bias/m/Read/ReadVariableOp.Adam/enc_middle_0/kernel/m/Read/ReadVariableOp,Adam/enc_middle_0/bias/m/Read/ReadVariableOp.Adam/enc_middle_1/kernel/m/Read/ReadVariableOp,Adam/enc_middle_1/bias/m/Read/ReadVariableOp-Adam/enc_inner_0/kernel/m/Read/ReadVariableOp+Adam/enc_inner_0/bias/m/Read/ReadVariableOp-Adam/enc_inner_1/kernel/m/Read/ReadVariableOp+Adam/enc_inner_1/bias/m/Read/ReadVariableOp+Adam/channel_0/kernel/m/Read/ReadVariableOp)Adam/channel_0/bias/m/Read/ReadVariableOp+Adam/channel_1/kernel/m/Read/ReadVariableOp)Adam/channel_1/bias/m/Read/ReadVariableOp-Adam/dec_inner_0/kernel/m/Read/ReadVariableOp+Adam/dec_inner_0/bias/m/Read/ReadVariableOp-Adam/dec_inner_1/kernel/m/Read/ReadVariableOp+Adam/dec_inner_1/bias/m/Read/ReadVariableOp.Adam/dec_middle_0/kernel/m/Read/ReadVariableOp,Adam/dec_middle_0/bias/m/Read/ReadVariableOp.Adam/dec_middle_1/kernel/m/Read/ReadVariableOp,Adam/dec_middle_1/bias/m/Read/ReadVariableOp-Adam/dec_outer_0/kernel/m/Read/ReadVariableOp+Adam/dec_outer_0/bias/m/Read/ReadVariableOp-Adam/dec_outer_1/kernel/m/Read/ReadVariableOp+Adam/dec_outer_1/bias/m/Read/ReadVariableOp,Adam/dec_output/kernel/m/Read/ReadVariableOp*Adam/dec_output/bias/m/Read/ReadVariableOp-Adam/enc_outer_0/kernel/v/Read/ReadVariableOp+Adam/enc_outer_0/bias/v/Read/ReadVariableOp-Adam/enc_outer_1/kernel/v/Read/ReadVariableOp+Adam/enc_outer_1/bias/v/Read/ReadVariableOp.Adam/enc_middle_0/kernel/v/Read/ReadVariableOp,Adam/enc_middle_0/bias/v/Read/ReadVariableOp.Adam/enc_middle_1/kernel/v/Read/ReadVariableOp,Adam/enc_middle_1/bias/v/Read/ReadVariableOp-Adam/enc_inner_0/kernel/v/Read/ReadVariableOp+Adam/enc_inner_0/bias/v/Read/ReadVariableOp-Adam/enc_inner_1/kernel/v/Read/ReadVariableOp+Adam/enc_inner_1/bias/v/Read/ReadVariableOp+Adam/channel_0/kernel/v/Read/ReadVariableOp)Adam/channel_0/bias/v/Read/ReadVariableOp+Adam/channel_1/kernel/v/Read/ReadVariableOp)Adam/channel_1/bias/v/Read/ReadVariableOp-Adam/dec_inner_0/kernel/v/Read/ReadVariableOp+Adam/dec_inner_0/bias/v/Read/ReadVariableOp-Adam/dec_inner_1/kernel/v/Read/ReadVariableOp+Adam/dec_inner_1/bias/v/Read/ReadVariableOp.Adam/dec_middle_0/kernel/v/Read/ReadVariableOp,Adam/dec_middle_0/bias/v/Read/ReadVariableOp.Adam/dec_middle_1/kernel/v/Read/ReadVariableOp,Adam/dec_middle_1/bias/v/Read/ReadVariableOp-Adam/dec_outer_0/kernel/v/Read/ReadVariableOp+Adam/dec_outer_0/bias/v/Read/ReadVariableOp-Adam/dec_outer_1/kernel/v/Read/ReadVariableOp+Adam/dec_outer_1/bias/v/Read/ReadVariableOp,Adam/dec_output/kernel/v/Read/ReadVariableOp*Adam/dec_output/bias/v/Read/ReadVariableOpConst*n
Ting
e2c	*
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
__inference__traced_save_169480
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateenc_outer_0/kernelenc_outer_0/biasenc_outer_1/kernelenc_outer_1/biasenc_middle_0/kernelenc_middle_0/biasenc_middle_1/kernelenc_middle_1/biasenc_inner_0/kernelenc_inner_0/biasenc_inner_1/kernelenc_inner_1/biaschannel_0/kernelchannel_0/biaschannel_1/kernelchannel_1/biasdec_inner_0/kerneldec_inner_0/biasdec_inner_1/kerneldec_inner_1/biasdec_middle_0/kerneldec_middle_0/biasdec_middle_1/kerneldec_middle_1/biasdec_outer_0/kerneldec_outer_0/biasdec_outer_1/kerneldec_outer_1/biasdec_output/kerneldec_output/biastotalcountAdam/enc_outer_0/kernel/mAdam/enc_outer_0/bias/mAdam/enc_outer_1/kernel/mAdam/enc_outer_1/bias/mAdam/enc_middle_0/kernel/mAdam/enc_middle_0/bias/mAdam/enc_middle_1/kernel/mAdam/enc_middle_1/bias/mAdam/enc_inner_0/kernel/mAdam/enc_inner_0/bias/mAdam/enc_inner_1/kernel/mAdam/enc_inner_1/bias/mAdam/channel_0/kernel/mAdam/channel_0/bias/mAdam/channel_1/kernel/mAdam/channel_1/bias/mAdam/dec_inner_0/kernel/mAdam/dec_inner_0/bias/mAdam/dec_inner_1/kernel/mAdam/dec_inner_1/bias/mAdam/dec_middle_0/kernel/mAdam/dec_middle_0/bias/mAdam/dec_middle_1/kernel/mAdam/dec_middle_1/bias/mAdam/dec_outer_0/kernel/mAdam/dec_outer_0/bias/mAdam/dec_outer_1/kernel/mAdam/dec_outer_1/bias/mAdam/dec_output/kernel/mAdam/dec_output/bias/mAdam/enc_outer_0/kernel/vAdam/enc_outer_0/bias/vAdam/enc_outer_1/kernel/vAdam/enc_outer_1/bias/vAdam/enc_middle_0/kernel/vAdam/enc_middle_0/bias/vAdam/enc_middle_1/kernel/vAdam/enc_middle_1/bias/vAdam/enc_inner_0/kernel/vAdam/enc_inner_0/bias/vAdam/enc_inner_1/kernel/vAdam/enc_inner_1/bias/vAdam/channel_0/kernel/vAdam/channel_0/bias/vAdam/channel_1/kernel/vAdam/channel_1/bias/vAdam/dec_inner_0/kernel/vAdam/dec_inner_0/bias/vAdam/dec_inner_1/kernel/vAdam/dec_inner_1/bias/vAdam/dec_middle_0/kernel/vAdam/dec_middle_0/bias/vAdam/dec_middle_1/kernel/vAdam/dec_middle_1/bias/vAdam/dec_outer_0/kernel/vAdam/dec_outer_0/bias/vAdam/dec_outer_1/kernel/vAdam/dec_outer_1/bias/vAdam/dec_output/kernel/vAdam/dec_output/bias/v*m
Tinf
d2b*
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
"__inference__traced_restore_169781??
?
?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167794
input_1
model_2_167730
model_2_167732
model_2_167734
model_2_167736
model_2_167738
model_2_167740
model_2_167742
model_2_167744
model_2_167746
model_2_167748
model_2_167750
model_2_167752
model_2_167754
model_2_167756
model_2_167758
model_2_167760
model_3_167764
model_3_167766
model_3_167768
model_3_167770
model_3_167772
model_3_167774
model_3_167776
model_3_167778
model_3_167780
model_3_167782
model_3_167784
model_3_167786
model_3_167788
model_3_167790
identity??model_2/StatefulPartitionedCall?model_3/StatefulPartitionedCall?
model_2/StatefulPartitionedCallStatefulPartitionedCallinput_1model_2_167730model_2_167732model_2_167734model_2_167736model_2_167738model_2_167740model_2_167742model_2_167744model_2_167746model_2_167748model_2_167750model_2_167752model_2_167754model_2_167756model_2_167758model_2_167760*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670842!
model_2/StatefulPartitionedCall?
model_3/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0(model_2/StatefulPartitionedCall:output:1model_3_167764model_3_167766model_3_167768model_3_167770model_3_167772model_3_167774model_3_167776model_3_167778model_3_167780model_3_167782model_3_167784model_3_167786model_3_167788model_3_167790*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674822!
model_3/StatefulPartitionedCall?
IdentityIdentity(model_3/StatefulPartitionedCall:output:0 ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167996
x
model_2_167932
model_2_167934
model_2_167936
model_2_167938
model_2_167940
model_2_167942
model_2_167944
model_2_167946
model_2_167948
model_2_167950
model_2_167952
model_2_167954
model_2_167956
model_2_167958
model_2_167960
model_2_167962
model_3_167966
model_3_167968
model_3_167970
model_3_167972
model_3_167974
model_3_167976
model_3_167978
model_3_167980
model_3_167982
model_3_167984
model_3_167986
model_3_167988
model_3_167990
model_3_167992
identity??model_2/StatefulPartitionedCall?model_3/StatefulPartitionedCall?
model_2/StatefulPartitionedCallStatefulPartitionedCallxmodel_2_167932model_2_167934model_2_167936model_2_167938model_2_167940model_2_167942model_2_167944model_2_167946model_2_167948model_2_167950model_2_167952model_2_167954model_2_167956model_2_167958model_2_167960model_2_167962*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670842!
model_2/StatefulPartitionedCall?
model_3/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0(model_2/StatefulPartitionedCall:output:1model_3_167966model_3_167968model_3_167970model_3_167972model_3_167974model_3_167976model_3_167978model_3_167980model_3_167982model_3_167984model_3_167986model_3_167988model_3_167990model_3_167992*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674822!
model_3/StatefulPartitionedCall?
IdentityIdentity(model_3/StatefulPartitionedCall:output:0 ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_168957

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
??
?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_168356
x6
2model_2_enc_outer_1_matmul_readvariableop_resource7
3model_2_enc_outer_1_biasadd_readvariableop_resource6
2model_2_enc_outer_0_matmul_readvariableop_resource7
3model_2_enc_outer_0_biasadd_readvariableop_resource7
3model_2_enc_middle_1_matmul_readvariableop_resource8
4model_2_enc_middle_1_biasadd_readvariableop_resource7
3model_2_enc_middle_0_matmul_readvariableop_resource8
4model_2_enc_middle_0_biasadd_readvariableop_resource6
2model_2_enc_inner_1_matmul_readvariableop_resource7
3model_2_enc_inner_1_biasadd_readvariableop_resource6
2model_2_enc_inner_0_matmul_readvariableop_resource7
3model_2_enc_inner_0_biasadd_readvariableop_resource4
0model_2_channel_1_matmul_readvariableop_resource5
1model_2_channel_1_biasadd_readvariableop_resource4
0model_2_channel_0_matmul_readvariableop_resource5
1model_2_channel_0_biasadd_readvariableop_resource6
2model_3_dec_inner_1_matmul_readvariableop_resource7
3model_3_dec_inner_1_biasadd_readvariableop_resource6
2model_3_dec_inner_0_matmul_readvariableop_resource7
3model_3_dec_inner_0_biasadd_readvariableop_resource7
3model_3_dec_middle_1_matmul_readvariableop_resource8
4model_3_dec_middle_1_biasadd_readvariableop_resource7
3model_3_dec_middle_0_matmul_readvariableop_resource8
4model_3_dec_middle_0_biasadd_readvariableop_resource6
2model_3_dec_outer_0_matmul_readvariableop_resource7
3model_3_dec_outer_0_biasadd_readvariableop_resource6
2model_3_dec_outer_1_matmul_readvariableop_resource7
3model_3_dec_outer_1_biasadd_readvariableop_resource5
1model_3_dec_output_matmul_readvariableop_resource6
2model_3_dec_output_biasadd_readvariableop_resource
identity??(model_2/channel_0/BiasAdd/ReadVariableOp?'model_2/channel_0/MatMul/ReadVariableOp?(model_2/channel_1/BiasAdd/ReadVariableOp?'model_2/channel_1/MatMul/ReadVariableOp?*model_2/enc_inner_0/BiasAdd/ReadVariableOp?)model_2/enc_inner_0/MatMul/ReadVariableOp?*model_2/enc_inner_1/BiasAdd/ReadVariableOp?)model_2/enc_inner_1/MatMul/ReadVariableOp?+model_2/enc_middle_0/BiasAdd/ReadVariableOp?*model_2/enc_middle_0/MatMul/ReadVariableOp?+model_2/enc_middle_1/BiasAdd/ReadVariableOp?*model_2/enc_middle_1/MatMul/ReadVariableOp?*model_2/enc_outer_0/BiasAdd/ReadVariableOp?)model_2/enc_outer_0/MatMul/ReadVariableOp?*model_2/enc_outer_1/BiasAdd/ReadVariableOp?)model_2/enc_outer_1/MatMul/ReadVariableOp?*model_3/dec_inner_0/BiasAdd/ReadVariableOp?)model_3/dec_inner_0/MatMul/ReadVariableOp?*model_3/dec_inner_1/BiasAdd/ReadVariableOp?)model_3/dec_inner_1/MatMul/ReadVariableOp?+model_3/dec_middle_0/BiasAdd/ReadVariableOp?*model_3/dec_middle_0/MatMul/ReadVariableOp?+model_3/dec_middle_1/BiasAdd/ReadVariableOp?*model_3/dec_middle_1/MatMul/ReadVariableOp?*model_3/dec_outer_0/BiasAdd/ReadVariableOp?)model_3/dec_outer_0/MatMul/ReadVariableOp?*model_3/dec_outer_1/BiasAdd/ReadVariableOp?)model_3/dec_outer_1/MatMul/ReadVariableOp?)model_3/dec_output/BiasAdd/ReadVariableOp?(model_3/dec_output/MatMul/ReadVariableOp?
)model_2/enc_outer_1/MatMul/ReadVariableOpReadVariableOp2model_2_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_2/enc_outer_1/MatMul/ReadVariableOp?
model_2/enc_outer_1/MatMulMatMulx1model_2/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_1/MatMul?
*model_2/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_2/enc_outer_1/BiasAdd/ReadVariableOp?
model_2/enc_outer_1/BiasAddBiasAdd$model_2/enc_outer_1/MatMul:product:02model_2/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_1/BiasAdd?
model_2/enc_outer_1/ReluRelu$model_2/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_1/Relu?
)model_2/enc_outer_0/MatMul/ReadVariableOpReadVariableOp2model_2_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_2/enc_outer_0/MatMul/ReadVariableOp?
model_2/enc_outer_0/MatMulMatMulx1model_2/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_0/MatMul?
*model_2/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_2/enc_outer_0/BiasAdd/ReadVariableOp?
model_2/enc_outer_0/BiasAddBiasAdd$model_2/enc_outer_0/MatMul:product:02model_2/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_0/BiasAdd?
model_2/enc_outer_0/ReluRelu$model_2/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_0/Relu?
*model_2/enc_middle_1/MatMul/ReadVariableOpReadVariableOp3model_2_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_2/enc_middle_1/MatMul/ReadVariableOp?
model_2/enc_middle_1/MatMulMatMul&model_2/enc_outer_1/Relu:activations:02model_2/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_1/MatMul?
+model_2/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_2_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_2/enc_middle_1/BiasAdd/ReadVariableOp?
model_2/enc_middle_1/BiasAddBiasAdd%model_2/enc_middle_1/MatMul:product:03model_2/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_1/BiasAdd?
model_2/enc_middle_1/ReluRelu%model_2/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_1/Relu?
*model_2/enc_middle_0/MatMul/ReadVariableOpReadVariableOp3model_2_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_2/enc_middle_0/MatMul/ReadVariableOp?
model_2/enc_middle_0/MatMulMatMul&model_2/enc_outer_0/Relu:activations:02model_2/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_0/MatMul?
+model_2/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_2_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_2/enc_middle_0/BiasAdd/ReadVariableOp?
model_2/enc_middle_0/BiasAddBiasAdd%model_2/enc_middle_0/MatMul:product:03model_2/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_0/BiasAdd?
model_2/enc_middle_0/ReluRelu%model_2/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_0/Relu?
)model_2/enc_inner_1/MatMul/ReadVariableOpReadVariableOp2model_2_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_2/enc_inner_1/MatMul/ReadVariableOp?
model_2/enc_inner_1/MatMulMatMul'model_2/enc_middle_1/Relu:activations:01model_2/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_1/MatMul?
*model_2/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_2/enc_inner_1/BiasAdd/ReadVariableOp?
model_2/enc_inner_1/BiasAddBiasAdd$model_2/enc_inner_1/MatMul:product:02model_2/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_1/BiasAdd?
model_2/enc_inner_1/ReluRelu$model_2/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_1/Relu?
)model_2/enc_inner_0/MatMul/ReadVariableOpReadVariableOp2model_2_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_2/enc_inner_0/MatMul/ReadVariableOp?
model_2/enc_inner_0/MatMulMatMul'model_2/enc_middle_0/Relu:activations:01model_2/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_0/MatMul?
*model_2/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_2/enc_inner_0/BiasAdd/ReadVariableOp?
model_2/enc_inner_0/BiasAddBiasAdd$model_2/enc_inner_0/MatMul:product:02model_2/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_0/BiasAdd?
model_2/enc_inner_0/ReluRelu$model_2/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_0/Relu?
'model_2/channel_1/MatMul/ReadVariableOpReadVariableOp0model_2_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_2/channel_1/MatMul/ReadVariableOp?
model_2/channel_1/MatMulMatMul&model_2/enc_inner_1/Relu:activations:0/model_2/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_1/MatMul?
(model_2/channel_1/BiasAdd/ReadVariableOpReadVariableOp1model_2_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/channel_1/BiasAdd/ReadVariableOp?
model_2/channel_1/BiasAddBiasAdd"model_2/channel_1/MatMul:product:00model_2/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_1/BiasAdd?
model_2/channel_1/SoftsignSoftsign"model_2/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/channel_1/Softsign?
'model_2/channel_0/MatMul/ReadVariableOpReadVariableOp0model_2_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_2/channel_0/MatMul/ReadVariableOp?
model_2/channel_0/MatMulMatMul&model_2/enc_inner_0/Relu:activations:0/model_2/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_0/MatMul?
(model_2/channel_0/BiasAdd/ReadVariableOpReadVariableOp1model_2_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/channel_0/BiasAdd/ReadVariableOp?
model_2/channel_0/BiasAddBiasAdd"model_2/channel_0/MatMul:product:00model_2/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_0/BiasAdd?
model_2/channel_0/SoftsignSoftsign"model_2/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/channel_0/Softsign?
)model_3/dec_inner_1/MatMul/ReadVariableOpReadVariableOp2model_3_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_3/dec_inner_1/MatMul/ReadVariableOp?
model_3/dec_inner_1/MatMulMatMul(model_2/channel_1/Softsign:activations:01model_3/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_1/MatMul?
*model_3/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_3/dec_inner_1/BiasAdd/ReadVariableOp?
model_3/dec_inner_1/BiasAddBiasAdd$model_3/dec_inner_1/MatMul:product:02model_3/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_1/BiasAdd?
model_3/dec_inner_1/ReluRelu$model_3/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_1/Relu?
)model_3/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_3_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_3/dec_inner_0/MatMul/ReadVariableOp?
model_3/dec_inner_0/MatMulMatMul(model_2/channel_0/Softsign:activations:01model_3/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_0/MatMul?
*model_3/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_3/dec_inner_0/BiasAdd/ReadVariableOp?
model_3/dec_inner_0/BiasAddBiasAdd$model_3/dec_inner_0/MatMul:product:02model_3/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_0/BiasAdd?
model_3/dec_inner_0/ReluRelu$model_3/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_0/Relu?
*model_3/dec_middle_1/MatMul/ReadVariableOpReadVariableOp3model_3_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_3/dec_middle_1/MatMul/ReadVariableOp?
model_3/dec_middle_1/MatMulMatMul&model_3/dec_inner_1/Relu:activations:02model_3/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_1/MatMul?
+model_3/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_3_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_3/dec_middle_1/BiasAdd/ReadVariableOp?
model_3/dec_middle_1/BiasAddBiasAdd%model_3/dec_middle_1/MatMul:product:03model_3/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_1/BiasAdd?
model_3/dec_middle_1/ReluRelu%model_3/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_1/Relu?
*model_3/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_3_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_3/dec_middle_0/MatMul/ReadVariableOp?
model_3/dec_middle_0/MatMulMatMul&model_3/dec_inner_0/Relu:activations:02model_3/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_0/MatMul?
+model_3/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_3_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_3/dec_middle_0/BiasAdd/ReadVariableOp?
model_3/dec_middle_0/BiasAddBiasAdd%model_3/dec_middle_0/MatMul:product:03model_3/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_0/BiasAdd?
model_3/dec_middle_0/ReluRelu%model_3/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_0/Relu?
)model_3/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_3_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_3/dec_outer_0/MatMul/ReadVariableOp?
model_3/dec_outer_0/MatMulMatMul'model_3/dec_middle_0/Relu:activations:01model_3/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_0/MatMul?
*model_3/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_3/dec_outer_0/BiasAdd/ReadVariableOp?
model_3/dec_outer_0/BiasAddBiasAdd$model_3/dec_outer_0/MatMul:product:02model_3/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_0/BiasAdd?
model_3/dec_outer_0/ReluRelu$model_3/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_0/Relu?
)model_3/dec_outer_1/MatMul/ReadVariableOpReadVariableOp2model_3_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_3/dec_outer_1/MatMul/ReadVariableOp?
model_3/dec_outer_1/MatMulMatMul'model_3/dec_middle_1/Relu:activations:01model_3/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_1/MatMul?
*model_3/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_3/dec_outer_1/BiasAdd/ReadVariableOp?
model_3/dec_outer_1/BiasAddBiasAdd$model_3/dec_outer_1/MatMul:product:02model_3/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_1/BiasAdd?
model_3/dec_outer_1/ReluRelu$model_3/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_1/Relu?
model_3/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model_3/tf.concat/concat/axis?
model_3/tf.concat/concatConcatV2&model_3/dec_outer_0/Relu:activations:0&model_3/dec_outer_1/Relu:activations:0&model_3/tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
model_3/tf.concat/concat?
(model_3/dec_output/MatMul/ReadVariableOpReadVariableOp1model_3_dec_output_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02*
(model_3/dec_output/MatMul/ReadVariableOp?
model_3/dec_output/MatMulMatMul!model_3/tf.concat/concat:output:00model_3/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dec_output/MatMul?
)model_3/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_3_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_3/dec_output/BiasAdd/ReadVariableOp?
model_3/dec_output/BiasAddBiasAdd#model_3/dec_output/MatMul:product:01model_3/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dec_output/BiasAdd?
model_3/dec_output/SigmoidSigmoid#model_3/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_3/dec_output/Sigmoid?
IdentityIdentitymodel_3/dec_output/Sigmoid:y:0)^model_2/channel_0/BiasAdd/ReadVariableOp(^model_2/channel_0/MatMul/ReadVariableOp)^model_2/channel_1/BiasAdd/ReadVariableOp(^model_2/channel_1/MatMul/ReadVariableOp+^model_2/enc_inner_0/BiasAdd/ReadVariableOp*^model_2/enc_inner_0/MatMul/ReadVariableOp+^model_2/enc_inner_1/BiasAdd/ReadVariableOp*^model_2/enc_inner_1/MatMul/ReadVariableOp,^model_2/enc_middle_0/BiasAdd/ReadVariableOp+^model_2/enc_middle_0/MatMul/ReadVariableOp,^model_2/enc_middle_1/BiasAdd/ReadVariableOp+^model_2/enc_middle_1/MatMul/ReadVariableOp+^model_2/enc_outer_0/BiasAdd/ReadVariableOp*^model_2/enc_outer_0/MatMul/ReadVariableOp+^model_2/enc_outer_1/BiasAdd/ReadVariableOp*^model_2/enc_outer_1/MatMul/ReadVariableOp+^model_3/dec_inner_0/BiasAdd/ReadVariableOp*^model_3/dec_inner_0/MatMul/ReadVariableOp+^model_3/dec_inner_1/BiasAdd/ReadVariableOp*^model_3/dec_inner_1/MatMul/ReadVariableOp,^model_3/dec_middle_0/BiasAdd/ReadVariableOp+^model_3/dec_middle_0/MatMul/ReadVariableOp,^model_3/dec_middle_1/BiasAdd/ReadVariableOp+^model_3/dec_middle_1/MatMul/ReadVariableOp+^model_3/dec_outer_0/BiasAdd/ReadVariableOp*^model_3/dec_outer_0/MatMul/ReadVariableOp+^model_3/dec_outer_1/BiasAdd/ReadVariableOp*^model_3/dec_outer_1/MatMul/ReadVariableOp*^model_3/dec_output/BiasAdd/ReadVariableOp)^model_3/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2T
(model_2/channel_0/BiasAdd/ReadVariableOp(model_2/channel_0/BiasAdd/ReadVariableOp2R
'model_2/channel_0/MatMul/ReadVariableOp'model_2/channel_0/MatMul/ReadVariableOp2T
(model_2/channel_1/BiasAdd/ReadVariableOp(model_2/channel_1/BiasAdd/ReadVariableOp2R
'model_2/channel_1/MatMul/ReadVariableOp'model_2/channel_1/MatMul/ReadVariableOp2X
*model_2/enc_inner_0/BiasAdd/ReadVariableOp*model_2/enc_inner_0/BiasAdd/ReadVariableOp2V
)model_2/enc_inner_0/MatMul/ReadVariableOp)model_2/enc_inner_0/MatMul/ReadVariableOp2X
*model_2/enc_inner_1/BiasAdd/ReadVariableOp*model_2/enc_inner_1/BiasAdd/ReadVariableOp2V
)model_2/enc_inner_1/MatMul/ReadVariableOp)model_2/enc_inner_1/MatMul/ReadVariableOp2Z
+model_2/enc_middle_0/BiasAdd/ReadVariableOp+model_2/enc_middle_0/BiasAdd/ReadVariableOp2X
*model_2/enc_middle_0/MatMul/ReadVariableOp*model_2/enc_middle_0/MatMul/ReadVariableOp2Z
+model_2/enc_middle_1/BiasAdd/ReadVariableOp+model_2/enc_middle_1/BiasAdd/ReadVariableOp2X
*model_2/enc_middle_1/MatMul/ReadVariableOp*model_2/enc_middle_1/MatMul/ReadVariableOp2X
*model_2/enc_outer_0/BiasAdd/ReadVariableOp*model_2/enc_outer_0/BiasAdd/ReadVariableOp2V
)model_2/enc_outer_0/MatMul/ReadVariableOp)model_2/enc_outer_0/MatMul/ReadVariableOp2X
*model_2/enc_outer_1/BiasAdd/ReadVariableOp*model_2/enc_outer_1/BiasAdd/ReadVariableOp2V
)model_2/enc_outer_1/MatMul/ReadVariableOp)model_2/enc_outer_1/MatMul/ReadVariableOp2X
*model_3/dec_inner_0/BiasAdd/ReadVariableOp*model_3/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_3/dec_inner_0/MatMul/ReadVariableOp)model_3/dec_inner_0/MatMul/ReadVariableOp2X
*model_3/dec_inner_1/BiasAdd/ReadVariableOp*model_3/dec_inner_1/BiasAdd/ReadVariableOp2V
)model_3/dec_inner_1/MatMul/ReadVariableOp)model_3/dec_inner_1/MatMul/ReadVariableOp2Z
+model_3/dec_middle_0/BiasAdd/ReadVariableOp+model_3/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_3/dec_middle_0/MatMul/ReadVariableOp*model_3/dec_middle_0/MatMul/ReadVariableOp2Z
+model_3/dec_middle_1/BiasAdd/ReadVariableOp+model_3/dec_middle_1/BiasAdd/ReadVariableOp2X
*model_3/dec_middle_1/MatMul/ReadVariableOp*model_3/dec_middle_1/MatMul/ReadVariableOp2X
*model_3/dec_outer_0/BiasAdd/ReadVariableOp*model_3/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_3/dec_outer_0/MatMul/ReadVariableOp)model_3/dec_outer_0/MatMul/ReadVariableOp2X
*model_3/dec_outer_1/BiasAdd/ReadVariableOp*model_3/dec_outer_1/BiasAdd/ReadVariableOp2V
)model_3/dec_outer_1/MatMul/ReadVariableOp)model_3/dec_outer_1/MatMul/ReadVariableOp2V
)model_3/dec_output/BiasAdd/ReadVariableOp)model_3/dec_output/BiasAdd/ReadVariableOp2T
(model_3/dec_output/MatMul/ReadVariableOp(model_3/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_168877

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
??
?
!__inference__wrapped_model_166685
input_1D
@autoencoder_1_model_2_enc_outer_1_matmul_readvariableop_resourceE
Aautoencoder_1_model_2_enc_outer_1_biasadd_readvariableop_resourceD
@autoencoder_1_model_2_enc_outer_0_matmul_readvariableop_resourceE
Aautoencoder_1_model_2_enc_outer_0_biasadd_readvariableop_resourceE
Aautoencoder_1_model_2_enc_middle_1_matmul_readvariableop_resourceF
Bautoencoder_1_model_2_enc_middle_1_biasadd_readvariableop_resourceE
Aautoencoder_1_model_2_enc_middle_0_matmul_readvariableop_resourceF
Bautoencoder_1_model_2_enc_middle_0_biasadd_readvariableop_resourceD
@autoencoder_1_model_2_enc_inner_1_matmul_readvariableop_resourceE
Aautoencoder_1_model_2_enc_inner_1_biasadd_readvariableop_resourceD
@autoencoder_1_model_2_enc_inner_0_matmul_readvariableop_resourceE
Aautoencoder_1_model_2_enc_inner_0_biasadd_readvariableop_resourceB
>autoencoder_1_model_2_channel_1_matmul_readvariableop_resourceC
?autoencoder_1_model_2_channel_1_biasadd_readvariableop_resourceB
>autoencoder_1_model_2_channel_0_matmul_readvariableop_resourceC
?autoencoder_1_model_2_channel_0_biasadd_readvariableop_resourceD
@autoencoder_1_model_3_dec_inner_1_matmul_readvariableop_resourceE
Aautoencoder_1_model_3_dec_inner_1_biasadd_readvariableop_resourceD
@autoencoder_1_model_3_dec_inner_0_matmul_readvariableop_resourceE
Aautoencoder_1_model_3_dec_inner_0_biasadd_readvariableop_resourceE
Aautoencoder_1_model_3_dec_middle_1_matmul_readvariableop_resourceF
Bautoencoder_1_model_3_dec_middle_1_biasadd_readvariableop_resourceE
Aautoencoder_1_model_3_dec_middle_0_matmul_readvariableop_resourceF
Bautoencoder_1_model_3_dec_middle_0_biasadd_readvariableop_resourceD
@autoencoder_1_model_3_dec_outer_0_matmul_readvariableop_resourceE
Aautoencoder_1_model_3_dec_outer_0_biasadd_readvariableop_resourceD
@autoencoder_1_model_3_dec_outer_1_matmul_readvariableop_resourceE
Aautoencoder_1_model_3_dec_outer_1_biasadd_readvariableop_resourceC
?autoencoder_1_model_3_dec_output_matmul_readvariableop_resourceD
@autoencoder_1_model_3_dec_output_biasadd_readvariableop_resource
identity??6autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOp?5autoencoder_1/model_2/channel_0/MatMul/ReadVariableOp?6autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOp?5autoencoder_1/model_2/channel_1/MatMul/ReadVariableOp?8autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOp?7autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOp?8autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOp?7autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOp?9autoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOp?8autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOp?9autoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOp?8autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOp?8autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOp?7autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOp?8autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOp?7autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOp?8autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOp?7autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOp?8autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOp?7autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOp?9autoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOp?8autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOp?9autoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOp?8autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOp?8autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOp?7autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOp?8autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOp?7autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOp?7autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOp?6autoencoder_1/model_3/dec_output/MatMul/ReadVariableOp?
7autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_2_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOp?
(autoencoder_1/model_2/enc_outer_1/MatMulMatMulinput_1?autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_1/model_2/enc_outer_1/MatMul?
8autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_2_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOp?
)autoencoder_1/model_2/enc_outer_1/BiasAddBiasAdd2autoencoder_1/model_2/enc_outer_1/MatMul:product:0@autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_1/model_2/enc_outer_1/BiasAdd?
&autoencoder_1/model_2/enc_outer_1/ReluRelu2autoencoder_1/model_2/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_1/model_2/enc_outer_1/Relu?
7autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_2_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype029
7autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOp?
(autoencoder_1/model_2/enc_outer_0/MatMulMatMulinput_1?autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_1/model_2/enc_outer_0/MatMul?
8autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_2_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOp?
)autoencoder_1/model_2/enc_outer_0/BiasAddBiasAdd2autoencoder_1/model_2/enc_outer_0/MatMul:product:0@autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_1/model_2/enc_outer_0/BiasAdd?
&autoencoder_1/model_2/enc_outer_0/ReluRelu2autoencoder_1/model_2/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_1/model_2/enc_outer_0/Relu?
8autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOpReadVariableOpAautoencoder_1_model_2_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOp?
)autoencoder_1/model_2/enc_middle_1/MatMulMatMul4autoencoder_1/model_2/enc_outer_1/Relu:activations:0@autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_1/model_2/enc_middle_1/MatMul?
9autoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_1_model_2_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOp?
*autoencoder_1/model_2/enc_middle_1/BiasAddBiasAdd3autoencoder_1/model_2/enc_middle_1/MatMul:product:0Aautoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_1/model_2/enc_middle_1/BiasAdd?
'autoencoder_1/model_2/enc_middle_1/ReluRelu3autoencoder_1/model_2/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_1/model_2/enc_middle_1/Relu?
8autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOpReadVariableOpAautoencoder_1_model_2_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02:
8autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOp?
)autoencoder_1/model_2/enc_middle_0/MatMulMatMul4autoencoder_1/model_2/enc_outer_0/Relu:activations:0@autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)autoencoder_1/model_2/enc_middle_0/MatMul?
9autoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_1_model_2_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9autoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOp?
*autoencoder_1/model_2/enc_middle_0/BiasAddBiasAdd3autoencoder_1/model_2/enc_middle_0/MatMul:product:0Aautoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*autoencoder_1/model_2/enc_middle_0/BiasAdd?
'autoencoder_1/model_2/enc_middle_0/ReluRelu3autoencoder_1/model_2/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22)
'autoencoder_1/model_2/enc_middle_0/Relu?
7autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_2_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOp?
(autoencoder_1/model_2/enc_inner_1/MatMulMatMul5autoencoder_1/model_2/enc_middle_1/Relu:activations:0?autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_1/model_2/enc_inner_1/MatMul?
8autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_2_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOp?
)autoencoder_1/model_2/enc_inner_1/BiasAddBiasAdd2autoencoder_1/model_2/enc_inner_1/MatMul:product:0@autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_1/model_2/enc_inner_1/BiasAdd?
&autoencoder_1/model_2/enc_inner_1/ReluRelu2autoencoder_1/model_2/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_1/model_2/enc_inner_1/Relu?
7autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_2_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype029
7autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOp?
(autoencoder_1/model_2/enc_inner_0/MatMulMatMul5autoencoder_1/model_2/enc_middle_0/Relu:activations:0?autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_1/model_2/enc_inner_0/MatMul?
8autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_2_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOp?
)autoencoder_1/model_2/enc_inner_0/BiasAddBiasAdd2autoencoder_1/model_2/enc_inner_0/MatMul:product:0@autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_1/model_2/enc_inner_0/BiasAdd?
&autoencoder_1/model_2/enc_inner_0/ReluRelu2autoencoder_1/model_2/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_1/model_2/enc_inner_0/Relu?
5autoencoder_1/model_2/channel_1/MatMul/ReadVariableOpReadVariableOp>autoencoder_1_model_2_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_1/model_2/channel_1/MatMul/ReadVariableOp?
&autoencoder_1/model_2/channel_1/MatMulMatMul4autoencoder_1/model_2/enc_inner_1/Relu:activations:0=autoencoder_1/model_2/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_1/model_2/channel_1/MatMul?
6autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_1_model_2_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOp?
'autoencoder_1/model_2/channel_1/BiasAddBiasAdd0autoencoder_1/model_2/channel_1/MatMul:product:0>autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_1/model_2/channel_1/BiasAdd?
(autoencoder_1/model_2/channel_1/SoftsignSoftsign0autoencoder_1/model_2/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_1/model_2/channel_1/Softsign?
5autoencoder_1/model_2/channel_0/MatMul/ReadVariableOpReadVariableOp>autoencoder_1_model_2_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder_1/model_2/channel_0/MatMul/ReadVariableOp?
&autoencoder_1/model_2/channel_0/MatMulMatMul4autoencoder_1/model_2/enc_inner_0/Relu:activations:0=autoencoder_1/model_2/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&autoencoder_1/model_2/channel_0/MatMul?
6autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_1_model_2_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOp?
'autoencoder_1/model_2/channel_0/BiasAddBiasAdd0autoencoder_1/model_2/channel_0/MatMul:product:0>autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'autoencoder_1/model_2/channel_0/BiasAdd?
(autoencoder_1/model_2/channel_0/SoftsignSoftsign0autoencoder_1/model_2/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(autoencoder_1/model_2/channel_0/Softsign?
7autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_3_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOp?
(autoencoder_1/model_3/dec_inner_1/MatMulMatMul6autoencoder_1/model_2/channel_1/Softsign:activations:0?autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_1/model_3/dec_inner_1/MatMul?
8autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_3_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOp?
)autoencoder_1/model_3/dec_inner_1/BiasAddBiasAdd2autoencoder_1/model_3/dec_inner_1/MatMul:product:0@autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_1/model_3/dec_inner_1/BiasAdd?
&autoencoder_1/model_3/dec_inner_1/ReluRelu2autoencoder_1/model_3/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_1/model_3/dec_inner_1/Relu?
7autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_3_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype029
7autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOp?
(autoencoder_1/model_3/dec_inner_0/MatMulMatMul6autoencoder_1/model_2/channel_0/Softsign:activations:0?autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(autoencoder_1/model_3/dec_inner_0/MatMul?
8autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_3_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02:
8autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOp?
)autoencoder_1/model_3/dec_inner_0/BiasAddBiasAdd2autoencoder_1/model_3/dec_inner_0/MatMul:product:0@autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)autoencoder_1/model_3/dec_inner_0/BiasAdd?
&autoencoder_1/model_3/dec_inner_0/ReluRelu2autoencoder_1/model_3/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder_1/model_3/dec_inner_0/Relu?
8autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOpReadVariableOpAautoencoder_1_model_3_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOp?
)autoencoder_1/model_3/dec_middle_1/MatMulMatMul4autoencoder_1/model_3/dec_inner_1/Relu:activations:0@autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_1/model_3/dec_middle_1/MatMul?
9autoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_1_model_3_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOp?
*autoencoder_1/model_3/dec_middle_1/BiasAddBiasAdd3autoencoder_1/model_3/dec_middle_1/MatMul:product:0Aautoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_1/model_3/dec_middle_1/BiasAdd?
'autoencoder_1/model_3/dec_middle_1/ReluRelu3autoencoder_1/model_3/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_1/model_3/dec_middle_1/Relu?
8autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOpReadVariableOpAautoencoder_1_model_3_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02:
8autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOp?
)autoencoder_1/model_3/dec_middle_0/MatMulMatMul4autoencoder_1/model_3/dec_inner_0/Relu:activations:0@autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_1/model_3/dec_middle_0/MatMul?
9autoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_1_model_3_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9autoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOp?
*autoencoder_1/model_3/dec_middle_0/BiasAddBiasAdd3autoencoder_1/model_3/dec_middle_0/MatMul:product:0Aautoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2,
*autoencoder_1/model_3/dec_middle_0/BiasAdd?
'autoencoder_1/model_3/dec_middle_0/ReluRelu3autoencoder_1/model_3/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder_1/model_3/dec_middle_0/Relu?
7autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_3_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOp?
(autoencoder_1/model_3/dec_outer_0/MatMulMatMul5autoencoder_1/model_3/dec_middle_0/Relu:activations:0?autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_1/model_3/dec_outer_0/MatMul?
8autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_3_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOp?
)autoencoder_1/model_3/dec_outer_0/BiasAddBiasAdd2autoencoder_1/model_3/dec_outer_0/MatMul:product:0@autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_1/model_3/dec_outer_0/BiasAdd?
&autoencoder_1/model_3/dec_outer_0/ReluRelu2autoencoder_1/model_3/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_1/model_3/dec_outer_0/Relu?
7autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOpReadVariableOp@autoencoder_1_model_3_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype029
7autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOp?
(autoencoder_1/model_3/dec_outer_1/MatMulMatMul5autoencoder_1/model_3/dec_middle_1/Relu:activations:0?autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder_1/model_3/dec_outer_1/MatMul?
8autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_1_model_3_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02:
8autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOp?
)autoencoder_1/model_3/dec_outer_1/BiasAddBiasAdd2autoencoder_1/model_3/dec_outer_1/MatMul:product:0@autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2+
)autoencoder_1/model_3/dec_outer_1/BiasAdd?
&autoencoder_1/model_3/dec_outer_1/ReluRelu2autoencoder_1/model_3/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder_1/model_3/dec_outer_1/Relu?
+autoencoder_1/model_3/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+autoencoder_1/model_3/tf.concat/concat/axis?
&autoencoder_1/model_3/tf.concat/concatConcatV24autoencoder_1/model_3/dec_outer_0/Relu:activations:04autoencoder_1/model_3/dec_outer_1/Relu:activations:04autoencoder_1/model_3/tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2(
&autoencoder_1/model_3/tf.concat/concat?
6autoencoder_1/model_3/dec_output/MatMul/ReadVariableOpReadVariableOp?autoencoder_1_model_3_dec_output_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype028
6autoencoder_1/model_3/dec_output/MatMul/ReadVariableOp?
'autoencoder_1/model_3/dec_output/MatMulMatMul/autoencoder_1/model_3/tf.concat/concat:output:0>autoencoder_1/model_3/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'autoencoder_1/model_3/dec_output/MatMul?
7autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_1_model_3_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOp?
(autoencoder_1/model_3/dec_output/BiasAddBiasAdd1autoencoder_1/model_3/dec_output/MatMul:product:0?autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(autoencoder_1/model_3/dec_output/BiasAdd?
(autoencoder_1/model_3/dec_output/SigmoidSigmoid1autoencoder_1/model_3/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2*
(autoencoder_1/model_3/dec_output/Sigmoid?
IdentityIdentity,autoencoder_1/model_3/dec_output/Sigmoid:y:07^autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOp6^autoencoder_1/model_2/channel_0/MatMul/ReadVariableOp7^autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOp6^autoencoder_1/model_2/channel_1/MatMul/ReadVariableOp9^autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOp8^autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOp9^autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOp8^autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOp:^autoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOp9^autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOp:^autoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOp9^autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOp9^autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOp8^autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOp9^autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOp8^autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOp9^autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOp8^autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOp9^autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOp8^autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOp:^autoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOp9^autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOp:^autoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOp9^autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOp9^autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOp8^autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOp9^autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOp8^autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOp8^autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOp7^autoencoder_1/model_3/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2p
6autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOp6autoencoder_1/model_2/channel_0/BiasAdd/ReadVariableOp2n
5autoencoder_1/model_2/channel_0/MatMul/ReadVariableOp5autoencoder_1/model_2/channel_0/MatMul/ReadVariableOp2p
6autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOp6autoencoder_1/model_2/channel_1/BiasAdd/ReadVariableOp2n
5autoencoder_1/model_2/channel_1/MatMul/ReadVariableOp5autoencoder_1/model_2/channel_1/MatMul/ReadVariableOp2t
8autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOp8autoencoder_1/model_2/enc_inner_0/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOp7autoencoder_1/model_2/enc_inner_0/MatMul/ReadVariableOp2t
8autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOp8autoencoder_1/model_2/enc_inner_1/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOp7autoencoder_1/model_2/enc_inner_1/MatMul/ReadVariableOp2v
9autoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOp9autoencoder_1/model_2/enc_middle_0/BiasAdd/ReadVariableOp2t
8autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOp8autoencoder_1/model_2/enc_middle_0/MatMul/ReadVariableOp2v
9autoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOp9autoencoder_1/model_2/enc_middle_1/BiasAdd/ReadVariableOp2t
8autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOp8autoencoder_1/model_2/enc_middle_1/MatMul/ReadVariableOp2t
8autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOp8autoencoder_1/model_2/enc_outer_0/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOp7autoencoder_1/model_2/enc_outer_0/MatMul/ReadVariableOp2t
8autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOp8autoencoder_1/model_2/enc_outer_1/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOp7autoencoder_1/model_2/enc_outer_1/MatMul/ReadVariableOp2t
8autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOp8autoencoder_1/model_3/dec_inner_0/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOp7autoencoder_1/model_3/dec_inner_0/MatMul/ReadVariableOp2t
8autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOp8autoencoder_1/model_3/dec_inner_1/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOp7autoencoder_1/model_3/dec_inner_1/MatMul/ReadVariableOp2v
9autoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOp9autoencoder_1/model_3/dec_middle_0/BiasAdd/ReadVariableOp2t
8autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOp8autoencoder_1/model_3/dec_middle_0/MatMul/ReadVariableOp2v
9autoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOp9autoencoder_1/model_3/dec_middle_1/BiasAdd/ReadVariableOp2t
8autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOp8autoencoder_1/model_3/dec_middle_1/MatMul/ReadVariableOp2t
8autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOp8autoencoder_1/model_3/dec_outer_0/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOp7autoencoder_1/model_3/dec_outer_0/MatMul/ReadVariableOp2t
8autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOp8autoencoder_1/model_3/dec_outer_1/BiasAdd/ReadVariableOp2r
7autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOp7autoencoder_1/model_3/dec_outer_1/MatMul/ReadVariableOp2r
7autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOp7autoencoder_1/model_3/dec_output/BiasAdd/ReadVariableOp2p
6autoencoder_1/model_3/dec_output/MatMul/ReadVariableOp6autoencoder_1/model_3/dec_output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?0
?
C__inference_model_2_layer_call_and_return_conditional_losses_166907
encoder_input
enc_outer_1_166711
enc_outer_1_166713
enc_outer_0_166738
enc_outer_0_166740
enc_middle_1_166765
enc_middle_1_166767
enc_middle_0_166792
enc_middle_0_166794
enc_inner_1_166819
enc_inner_1_166821
enc_inner_0_166846
enc_inner_0_166848
channel_1_166873
channel_1_166875
channel_0_166900
channel_0_166902
identity

identity_1??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_1_166711enc_outer_1_166713*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_1667002%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_166738enc_outer_0_166740*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_1667272%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_166765enc_middle_1_166767*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_1667542&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_166792enc_middle_0_166794*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_1667812&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_166819enc_inner_1_166821*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_1668082%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_166846enc_inner_0_166848*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_1668352%
#enc_inner_0/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_166873channel_1_166875*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_1668622#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_166900channel_0_166902*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_1668892#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?
?
,__inference_enc_outer_0_layer_call_fn_168886

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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_1667272
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
?
(__inference_model_3_layer_call_fn_167513
decoder_input_0
decoder_input_1
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0decoder_input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::22
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
_user_specified_namedecoder_input_1
?

*__inference_channel_0_layer_call_fn_169006

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
E__inference_channel_0_layer_call_and_return_conditional_losses_1668892
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
?+
?
C__inference_model_3_layer_call_and_return_conditional_losses_167318
decoder_input_0
decoder_input_1
dec_inner_1_167148
dec_inner_1_167150
dec_inner_0_167175
dec_inner_0_167177
dec_middle_1_167202
dec_middle_1_167204
dec_middle_0_167229
dec_middle_0_167231
dec_outer_0_167256
dec_outer_0_167258
dec_outer_1_167283
dec_outer_1_167285
dec_output_167312
dec_output_167314
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_1dec_inner_1_167148dec_inner_1_167150*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_1671372%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_167175dec_inner_0_167177*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_1671642%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_167202dec_middle_1_167204*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_1671912&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_167229dec_middle_0_167231*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_1672182&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_167256dec_outer_0_167258*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_1672452%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_167283dec_outer_1_167285*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_1672722%
#dec_outer_1/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
tf.concat/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0dec_output_167312dec_output_167314*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_1673012$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2H
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
_user_specified_namedecoder_input_1
?	
?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_167245

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
?
?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167864
x
model_2_167800
model_2_167802
model_2_167804
model_2_167806
model_2_167808
model_2_167810
model_2_167812
model_2_167814
model_2_167816
model_2_167818
model_2_167820
model_2_167822
model_2_167824
model_2_167826
model_2_167828
model_2_167830
model_3_167834
model_3_167836
model_3_167838
model_3_167840
model_3_167842
model_3_167844
model_3_167846
model_3_167848
model_3_167850
model_3_167852
model_3_167854
model_3_167856
model_3_167858
model_3_167860
identity??model_2/StatefulPartitionedCall?model_3/StatefulPartitionedCall?
model_2/StatefulPartitionedCallStatefulPartitionedCallxmodel_2_167800model_2_167802model_2_167804model_2_167806model_2_167808model_2_167810model_2_167812model_2_167814model_2_167816model_2_167818model_2_167820model_2_167822model_2_167824model_2_167826model_2_167828model_2_167830*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670002!
model_2/StatefulPartitionedCall?
model_3/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0(model_2/StatefulPartitionedCall:output:1model_3_167834model_3_167836model_3_167838model_3_167840model_3_167842model_3_167844model_3_167846model_3_167848model_3_167850model_3_167852model_3_167854model_3_167856model_3_167858model_3_167860*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674062!
model_3/StatefulPartitionedCall?
IdentityIdentity(model_3/StatefulPartitionedCall:output:0 ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_169097

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
?
?
.__inference_autoencoder_1_layer_call_fn_168486
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

unknown_28
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_1679962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_166727

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
,__inference_dec_outer_0_layer_call_fn_169126

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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_1672452
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
-__inference_dec_middle_0_layer_call_fn_169086

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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_1672182
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
??
?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_168245
x6
2model_2_enc_outer_1_matmul_readvariableop_resource7
3model_2_enc_outer_1_biasadd_readvariableop_resource6
2model_2_enc_outer_0_matmul_readvariableop_resource7
3model_2_enc_outer_0_biasadd_readvariableop_resource7
3model_2_enc_middle_1_matmul_readvariableop_resource8
4model_2_enc_middle_1_biasadd_readvariableop_resource7
3model_2_enc_middle_0_matmul_readvariableop_resource8
4model_2_enc_middle_0_biasadd_readvariableop_resource6
2model_2_enc_inner_1_matmul_readvariableop_resource7
3model_2_enc_inner_1_biasadd_readvariableop_resource6
2model_2_enc_inner_0_matmul_readvariableop_resource7
3model_2_enc_inner_0_biasadd_readvariableop_resource4
0model_2_channel_1_matmul_readvariableop_resource5
1model_2_channel_1_biasadd_readvariableop_resource4
0model_2_channel_0_matmul_readvariableop_resource5
1model_2_channel_0_biasadd_readvariableop_resource6
2model_3_dec_inner_1_matmul_readvariableop_resource7
3model_3_dec_inner_1_biasadd_readvariableop_resource6
2model_3_dec_inner_0_matmul_readvariableop_resource7
3model_3_dec_inner_0_biasadd_readvariableop_resource7
3model_3_dec_middle_1_matmul_readvariableop_resource8
4model_3_dec_middle_1_biasadd_readvariableop_resource7
3model_3_dec_middle_0_matmul_readvariableop_resource8
4model_3_dec_middle_0_biasadd_readvariableop_resource6
2model_3_dec_outer_0_matmul_readvariableop_resource7
3model_3_dec_outer_0_biasadd_readvariableop_resource6
2model_3_dec_outer_1_matmul_readvariableop_resource7
3model_3_dec_outer_1_biasadd_readvariableop_resource5
1model_3_dec_output_matmul_readvariableop_resource6
2model_3_dec_output_biasadd_readvariableop_resource
identity??(model_2/channel_0/BiasAdd/ReadVariableOp?'model_2/channel_0/MatMul/ReadVariableOp?(model_2/channel_1/BiasAdd/ReadVariableOp?'model_2/channel_1/MatMul/ReadVariableOp?*model_2/enc_inner_0/BiasAdd/ReadVariableOp?)model_2/enc_inner_0/MatMul/ReadVariableOp?*model_2/enc_inner_1/BiasAdd/ReadVariableOp?)model_2/enc_inner_1/MatMul/ReadVariableOp?+model_2/enc_middle_0/BiasAdd/ReadVariableOp?*model_2/enc_middle_0/MatMul/ReadVariableOp?+model_2/enc_middle_1/BiasAdd/ReadVariableOp?*model_2/enc_middle_1/MatMul/ReadVariableOp?*model_2/enc_outer_0/BiasAdd/ReadVariableOp?)model_2/enc_outer_0/MatMul/ReadVariableOp?*model_2/enc_outer_1/BiasAdd/ReadVariableOp?)model_2/enc_outer_1/MatMul/ReadVariableOp?*model_3/dec_inner_0/BiasAdd/ReadVariableOp?)model_3/dec_inner_0/MatMul/ReadVariableOp?*model_3/dec_inner_1/BiasAdd/ReadVariableOp?)model_3/dec_inner_1/MatMul/ReadVariableOp?+model_3/dec_middle_0/BiasAdd/ReadVariableOp?*model_3/dec_middle_0/MatMul/ReadVariableOp?+model_3/dec_middle_1/BiasAdd/ReadVariableOp?*model_3/dec_middle_1/MatMul/ReadVariableOp?*model_3/dec_outer_0/BiasAdd/ReadVariableOp?)model_3/dec_outer_0/MatMul/ReadVariableOp?*model_3/dec_outer_1/BiasAdd/ReadVariableOp?)model_3/dec_outer_1/MatMul/ReadVariableOp?)model_3/dec_output/BiasAdd/ReadVariableOp?(model_3/dec_output/MatMul/ReadVariableOp?
)model_2/enc_outer_1/MatMul/ReadVariableOpReadVariableOp2model_2_enc_outer_1_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_2/enc_outer_1/MatMul/ReadVariableOp?
model_2/enc_outer_1/MatMulMatMulx1model_2/enc_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_1/MatMul?
*model_2/enc_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_2/enc_outer_1/BiasAdd/ReadVariableOp?
model_2/enc_outer_1/BiasAddBiasAdd$model_2/enc_outer_1/MatMul:product:02model_2/enc_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_1/BiasAdd?
model_2/enc_outer_1/ReluRelu$model_2/enc_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_1/Relu?
)model_2/enc_outer_0/MatMul/ReadVariableOpReadVariableOp2model_2_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02+
)model_2/enc_outer_0/MatMul/ReadVariableOp?
model_2/enc_outer_0/MatMulMatMulx1model_2/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_0/MatMul?
*model_2/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_2/enc_outer_0/BiasAdd/ReadVariableOp?
model_2/enc_outer_0/BiasAddBiasAdd$model_2/enc_outer_0/MatMul:product:02model_2/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_0/BiasAdd?
model_2/enc_outer_0/ReluRelu$model_2/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_2/enc_outer_0/Relu?
*model_2/enc_middle_1/MatMul/ReadVariableOpReadVariableOp3model_2_enc_middle_1_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_2/enc_middle_1/MatMul/ReadVariableOp?
model_2/enc_middle_1/MatMulMatMul&model_2/enc_outer_1/Relu:activations:02model_2/enc_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_1/MatMul?
+model_2/enc_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_2_enc_middle_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_2/enc_middle_1/BiasAdd/ReadVariableOp?
model_2/enc_middle_1/BiasAddBiasAdd%model_2/enc_middle_1/MatMul:product:03model_2/enc_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_1/BiasAdd?
model_2/enc_middle_1/ReluRelu%model_2/enc_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_1/Relu?
*model_2/enc_middle_0/MatMul/ReadVariableOpReadVariableOp3model_2_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02,
*model_2/enc_middle_0/MatMul/ReadVariableOp?
model_2/enc_middle_0/MatMulMatMul&model_2/enc_outer_0/Relu:activations:02model_2/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_0/MatMul?
+model_2/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_2_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+model_2/enc_middle_0/BiasAdd/ReadVariableOp?
model_2/enc_middle_0/BiasAddBiasAdd%model_2/enc_middle_0/MatMul:product:03model_2/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_0/BiasAdd?
model_2/enc_middle_0/ReluRelu%model_2/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_2/enc_middle_0/Relu?
)model_2/enc_inner_1/MatMul/ReadVariableOpReadVariableOp2model_2_enc_inner_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_2/enc_inner_1/MatMul/ReadVariableOp?
model_2/enc_inner_1/MatMulMatMul'model_2/enc_middle_1/Relu:activations:01model_2/enc_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_1/MatMul?
*model_2/enc_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_2/enc_inner_1/BiasAdd/ReadVariableOp?
model_2/enc_inner_1/BiasAddBiasAdd$model_2/enc_inner_1/MatMul:product:02model_2/enc_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_1/BiasAdd?
model_2/enc_inner_1/ReluRelu$model_2/enc_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_1/Relu?
)model_2/enc_inner_0/MatMul/ReadVariableOpReadVariableOp2model_2_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02+
)model_2/enc_inner_0/MatMul/ReadVariableOp?
model_2/enc_inner_0/MatMulMatMul'model_2/enc_middle_0/Relu:activations:01model_2/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_0/MatMul?
*model_2/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_2_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_2/enc_inner_0/BiasAdd/ReadVariableOp?
model_2/enc_inner_0/BiasAddBiasAdd$model_2/enc_inner_0/MatMul:product:02model_2/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_0/BiasAdd?
model_2/enc_inner_0/ReluRelu$model_2/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_2/enc_inner_0/Relu?
'model_2/channel_1/MatMul/ReadVariableOpReadVariableOp0model_2_channel_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_2/channel_1/MatMul/ReadVariableOp?
model_2/channel_1/MatMulMatMul&model_2/enc_inner_1/Relu:activations:0/model_2/channel_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_1/MatMul?
(model_2/channel_1/BiasAdd/ReadVariableOpReadVariableOp1model_2_channel_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/channel_1/BiasAdd/ReadVariableOp?
model_2/channel_1/BiasAddBiasAdd"model_2/channel_1/MatMul:product:00model_2/channel_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_1/BiasAdd?
model_2/channel_1/SoftsignSoftsign"model_2/channel_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/channel_1/Softsign?
'model_2/channel_0/MatMul/ReadVariableOpReadVariableOp0model_2_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02)
'model_2/channel_0/MatMul/ReadVariableOp?
model_2/channel_0/MatMulMatMul&model_2/enc_inner_0/Relu:activations:0/model_2/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_0/MatMul?
(model_2/channel_0/BiasAdd/ReadVariableOpReadVariableOp1model_2_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/channel_0/BiasAdd/ReadVariableOp?
model_2/channel_0/BiasAddBiasAdd"model_2/channel_0/MatMul:product:00model_2/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/channel_0/BiasAdd?
model_2/channel_0/SoftsignSoftsign"model_2/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/channel_0/Softsign?
)model_3/dec_inner_1/MatMul/ReadVariableOpReadVariableOp2model_3_dec_inner_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_3/dec_inner_1/MatMul/ReadVariableOp?
model_3/dec_inner_1/MatMulMatMul(model_2/channel_1/Softsign:activations:01model_3/dec_inner_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_1/MatMul?
*model_3/dec_inner_1/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_inner_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_3/dec_inner_1/BiasAdd/ReadVariableOp?
model_3/dec_inner_1/BiasAddBiasAdd$model_3/dec_inner_1/MatMul:product:02model_3/dec_inner_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_1/BiasAdd?
model_3/dec_inner_1/ReluRelu$model_3/dec_inner_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_1/Relu?
)model_3/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_3_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_3/dec_inner_0/MatMul/ReadVariableOp?
model_3/dec_inner_0/MatMulMatMul(model_2/channel_0/Softsign:activations:01model_3/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_0/MatMul?
*model_3/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_3/dec_inner_0/BiasAdd/ReadVariableOp?
model_3/dec_inner_0/BiasAddBiasAdd$model_3/dec_inner_0/MatMul:product:02model_3/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_0/BiasAdd?
model_3/dec_inner_0/ReluRelu$model_3/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_3/dec_inner_0/Relu?
*model_3/dec_middle_1/MatMul/ReadVariableOpReadVariableOp3model_3_dec_middle_1_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_3/dec_middle_1/MatMul/ReadVariableOp?
model_3/dec_middle_1/MatMulMatMul&model_3/dec_inner_1/Relu:activations:02model_3/dec_middle_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_1/MatMul?
+model_3/dec_middle_1/BiasAdd/ReadVariableOpReadVariableOp4model_3_dec_middle_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_3/dec_middle_1/BiasAdd/ReadVariableOp?
model_3/dec_middle_1/BiasAddBiasAdd%model_3/dec_middle_1/MatMul:product:03model_3/dec_middle_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_1/BiasAdd?
model_3/dec_middle_1/ReluRelu%model_3/dec_middle_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_1/Relu?
*model_3/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_3_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_3/dec_middle_0/MatMul/ReadVariableOp?
model_3/dec_middle_0/MatMulMatMul&model_3/dec_inner_0/Relu:activations:02model_3/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_0/MatMul?
+model_3/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_3_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_3/dec_middle_0/BiasAdd/ReadVariableOp?
model_3/dec_middle_0/BiasAddBiasAdd%model_3/dec_middle_0/MatMul:product:03model_3/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_0/BiasAdd?
model_3/dec_middle_0/ReluRelu%model_3/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_middle_0/Relu?
)model_3/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_3_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_3/dec_outer_0/MatMul/ReadVariableOp?
model_3/dec_outer_0/MatMulMatMul'model_3/dec_middle_0/Relu:activations:01model_3/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_0/MatMul?
*model_3/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_3/dec_outer_0/BiasAdd/ReadVariableOp?
model_3/dec_outer_0/BiasAddBiasAdd$model_3/dec_outer_0/MatMul:product:02model_3/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_0/BiasAdd?
model_3/dec_outer_0/ReluRelu$model_3/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_0/Relu?
)model_3/dec_outer_1/MatMul/ReadVariableOpReadVariableOp2model_3_dec_outer_1_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_3/dec_outer_1/MatMul/ReadVariableOp?
model_3/dec_outer_1/MatMulMatMul'model_3/dec_middle_1/Relu:activations:01model_3/dec_outer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_1/MatMul?
*model_3/dec_outer_1/BiasAdd/ReadVariableOpReadVariableOp3model_3_dec_outer_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_3/dec_outer_1/BiasAdd/ReadVariableOp?
model_3/dec_outer_1/BiasAddBiasAdd$model_3/dec_outer_1/MatMul:product:02model_3/dec_outer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_1/BiasAdd?
model_3/dec_outer_1/ReluRelu$model_3/dec_outer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_3/dec_outer_1/Relu?
model_3/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model_3/tf.concat/concat/axis?
model_3/tf.concat/concatConcatV2&model_3/dec_outer_0/Relu:activations:0&model_3/dec_outer_1/Relu:activations:0&model_3/tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
model_3/tf.concat/concat?
(model_3/dec_output/MatMul/ReadVariableOpReadVariableOp1model_3_dec_output_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02*
(model_3/dec_output/MatMul/ReadVariableOp?
model_3/dec_output/MatMulMatMul!model_3/tf.concat/concat:output:00model_3/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dec_output/MatMul?
)model_3/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_3_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_3/dec_output/BiasAdd/ReadVariableOp?
model_3/dec_output/BiasAddBiasAdd#model_3/dec_output/MatMul:product:01model_3/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dec_output/BiasAdd?
model_3/dec_output/SigmoidSigmoid#model_3/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_3/dec_output/Sigmoid?
IdentityIdentitymodel_3/dec_output/Sigmoid:y:0)^model_2/channel_0/BiasAdd/ReadVariableOp(^model_2/channel_0/MatMul/ReadVariableOp)^model_2/channel_1/BiasAdd/ReadVariableOp(^model_2/channel_1/MatMul/ReadVariableOp+^model_2/enc_inner_0/BiasAdd/ReadVariableOp*^model_2/enc_inner_0/MatMul/ReadVariableOp+^model_2/enc_inner_1/BiasAdd/ReadVariableOp*^model_2/enc_inner_1/MatMul/ReadVariableOp,^model_2/enc_middle_0/BiasAdd/ReadVariableOp+^model_2/enc_middle_0/MatMul/ReadVariableOp,^model_2/enc_middle_1/BiasAdd/ReadVariableOp+^model_2/enc_middle_1/MatMul/ReadVariableOp+^model_2/enc_outer_0/BiasAdd/ReadVariableOp*^model_2/enc_outer_0/MatMul/ReadVariableOp+^model_2/enc_outer_1/BiasAdd/ReadVariableOp*^model_2/enc_outer_1/MatMul/ReadVariableOp+^model_3/dec_inner_0/BiasAdd/ReadVariableOp*^model_3/dec_inner_0/MatMul/ReadVariableOp+^model_3/dec_inner_1/BiasAdd/ReadVariableOp*^model_3/dec_inner_1/MatMul/ReadVariableOp,^model_3/dec_middle_0/BiasAdd/ReadVariableOp+^model_3/dec_middle_0/MatMul/ReadVariableOp,^model_3/dec_middle_1/BiasAdd/ReadVariableOp+^model_3/dec_middle_1/MatMul/ReadVariableOp+^model_3/dec_outer_0/BiasAdd/ReadVariableOp*^model_3/dec_outer_0/MatMul/ReadVariableOp+^model_3/dec_outer_1/BiasAdd/ReadVariableOp*^model_3/dec_outer_1/MatMul/ReadVariableOp*^model_3/dec_output/BiasAdd/ReadVariableOp)^model_3/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2T
(model_2/channel_0/BiasAdd/ReadVariableOp(model_2/channel_0/BiasAdd/ReadVariableOp2R
'model_2/channel_0/MatMul/ReadVariableOp'model_2/channel_0/MatMul/ReadVariableOp2T
(model_2/channel_1/BiasAdd/ReadVariableOp(model_2/channel_1/BiasAdd/ReadVariableOp2R
'model_2/channel_1/MatMul/ReadVariableOp'model_2/channel_1/MatMul/ReadVariableOp2X
*model_2/enc_inner_0/BiasAdd/ReadVariableOp*model_2/enc_inner_0/BiasAdd/ReadVariableOp2V
)model_2/enc_inner_0/MatMul/ReadVariableOp)model_2/enc_inner_0/MatMul/ReadVariableOp2X
*model_2/enc_inner_1/BiasAdd/ReadVariableOp*model_2/enc_inner_1/BiasAdd/ReadVariableOp2V
)model_2/enc_inner_1/MatMul/ReadVariableOp)model_2/enc_inner_1/MatMul/ReadVariableOp2Z
+model_2/enc_middle_0/BiasAdd/ReadVariableOp+model_2/enc_middle_0/BiasAdd/ReadVariableOp2X
*model_2/enc_middle_0/MatMul/ReadVariableOp*model_2/enc_middle_0/MatMul/ReadVariableOp2Z
+model_2/enc_middle_1/BiasAdd/ReadVariableOp+model_2/enc_middle_1/BiasAdd/ReadVariableOp2X
*model_2/enc_middle_1/MatMul/ReadVariableOp*model_2/enc_middle_1/MatMul/ReadVariableOp2X
*model_2/enc_outer_0/BiasAdd/ReadVariableOp*model_2/enc_outer_0/BiasAdd/ReadVariableOp2V
)model_2/enc_outer_0/MatMul/ReadVariableOp)model_2/enc_outer_0/MatMul/ReadVariableOp2X
*model_2/enc_outer_1/BiasAdd/ReadVariableOp*model_2/enc_outer_1/BiasAdd/ReadVariableOp2V
)model_2/enc_outer_1/MatMul/ReadVariableOp)model_2/enc_outer_1/MatMul/ReadVariableOp2X
*model_3/dec_inner_0/BiasAdd/ReadVariableOp*model_3/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_3/dec_inner_0/MatMul/ReadVariableOp)model_3/dec_inner_0/MatMul/ReadVariableOp2X
*model_3/dec_inner_1/BiasAdd/ReadVariableOp*model_3/dec_inner_1/BiasAdd/ReadVariableOp2V
)model_3/dec_inner_1/MatMul/ReadVariableOp)model_3/dec_inner_1/MatMul/ReadVariableOp2Z
+model_3/dec_middle_0/BiasAdd/ReadVariableOp+model_3/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_3/dec_middle_0/MatMul/ReadVariableOp*model_3/dec_middle_0/MatMul/ReadVariableOp2Z
+model_3/dec_middle_1/BiasAdd/ReadVariableOp+model_3/dec_middle_1/BiasAdd/ReadVariableOp2X
*model_3/dec_middle_1/MatMul/ReadVariableOp*model_3/dec_middle_1/MatMul/ReadVariableOp2X
*model_3/dec_outer_0/BiasAdd/ReadVariableOp*model_3/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_3/dec_outer_0/MatMul/ReadVariableOp)model_3/dec_outer_0/MatMul/ReadVariableOp2X
*model_3/dec_outer_1/BiasAdd/ReadVariableOp*model_3/dec_outer_1/BiasAdd/ReadVariableOp2V
)model_3/dec_outer_1/MatMul/ReadVariableOp)model_3/dec_outer_1/MatMul/ReadVariableOp2V
)model_3/dec_output/BiasAdd/ReadVariableOp)model_3/dec_output/BiasAdd/ReadVariableOp2T
(model_3/dec_output/MatMul/ReadVariableOp(model_3/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
,__inference_dec_inner_0_layer_call_fn_169046

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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_1671642
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_169137

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
?
$__inference_signature_wrapper_168134
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

unknown_28
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_1666852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_168937

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
??
?4
"__inference__traced_restore_169781
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate)
%assignvariableop_5_enc_outer_0_kernel'
#assignvariableop_6_enc_outer_0_bias)
%assignvariableop_7_enc_outer_1_kernel'
#assignvariableop_8_enc_outer_1_bias*
&assignvariableop_9_enc_middle_0_kernel)
%assignvariableop_10_enc_middle_0_bias+
'assignvariableop_11_enc_middle_1_kernel)
%assignvariableop_12_enc_middle_1_bias*
&assignvariableop_13_enc_inner_0_kernel(
$assignvariableop_14_enc_inner_0_bias*
&assignvariableop_15_enc_inner_1_kernel(
$assignvariableop_16_enc_inner_1_bias(
$assignvariableop_17_channel_0_kernel&
"assignvariableop_18_channel_0_bias(
$assignvariableop_19_channel_1_kernel&
"assignvariableop_20_channel_1_bias*
&assignvariableop_21_dec_inner_0_kernel(
$assignvariableop_22_dec_inner_0_bias*
&assignvariableop_23_dec_inner_1_kernel(
$assignvariableop_24_dec_inner_1_bias+
'assignvariableop_25_dec_middle_0_kernel)
%assignvariableop_26_dec_middle_0_bias+
'assignvariableop_27_dec_middle_1_kernel)
%assignvariableop_28_dec_middle_1_bias*
&assignvariableop_29_dec_outer_0_kernel(
$assignvariableop_30_dec_outer_0_bias*
&assignvariableop_31_dec_outer_1_kernel(
$assignvariableop_32_dec_outer_1_bias)
%assignvariableop_33_dec_output_kernel'
#assignvariableop_34_dec_output_bias
assignvariableop_35_total
assignvariableop_36_count1
-assignvariableop_37_adam_enc_outer_0_kernel_m/
+assignvariableop_38_adam_enc_outer_0_bias_m1
-assignvariableop_39_adam_enc_outer_1_kernel_m/
+assignvariableop_40_adam_enc_outer_1_bias_m2
.assignvariableop_41_adam_enc_middle_0_kernel_m0
,assignvariableop_42_adam_enc_middle_0_bias_m2
.assignvariableop_43_adam_enc_middle_1_kernel_m0
,assignvariableop_44_adam_enc_middle_1_bias_m1
-assignvariableop_45_adam_enc_inner_0_kernel_m/
+assignvariableop_46_adam_enc_inner_0_bias_m1
-assignvariableop_47_adam_enc_inner_1_kernel_m/
+assignvariableop_48_adam_enc_inner_1_bias_m/
+assignvariableop_49_adam_channel_0_kernel_m-
)assignvariableop_50_adam_channel_0_bias_m/
+assignvariableop_51_adam_channel_1_kernel_m-
)assignvariableop_52_adam_channel_1_bias_m1
-assignvariableop_53_adam_dec_inner_0_kernel_m/
+assignvariableop_54_adam_dec_inner_0_bias_m1
-assignvariableop_55_adam_dec_inner_1_kernel_m/
+assignvariableop_56_adam_dec_inner_1_bias_m2
.assignvariableop_57_adam_dec_middle_0_kernel_m0
,assignvariableop_58_adam_dec_middle_0_bias_m2
.assignvariableop_59_adam_dec_middle_1_kernel_m0
,assignvariableop_60_adam_dec_middle_1_bias_m1
-assignvariableop_61_adam_dec_outer_0_kernel_m/
+assignvariableop_62_adam_dec_outer_0_bias_m1
-assignvariableop_63_adam_dec_outer_1_kernel_m/
+assignvariableop_64_adam_dec_outer_1_bias_m0
,assignvariableop_65_adam_dec_output_kernel_m.
*assignvariableop_66_adam_dec_output_bias_m1
-assignvariableop_67_adam_enc_outer_0_kernel_v/
+assignvariableop_68_adam_enc_outer_0_bias_v1
-assignvariableop_69_adam_enc_outer_1_kernel_v/
+assignvariableop_70_adam_enc_outer_1_bias_v2
.assignvariableop_71_adam_enc_middle_0_kernel_v0
,assignvariableop_72_adam_enc_middle_0_bias_v2
.assignvariableop_73_adam_enc_middle_1_kernel_v0
,assignvariableop_74_adam_enc_middle_1_bias_v1
-assignvariableop_75_adam_enc_inner_0_kernel_v/
+assignvariableop_76_adam_enc_inner_0_bias_v1
-assignvariableop_77_adam_enc_inner_1_kernel_v/
+assignvariableop_78_adam_enc_inner_1_bias_v/
+assignvariableop_79_adam_channel_0_kernel_v-
)assignvariableop_80_adam_channel_0_bias_v/
+assignvariableop_81_adam_channel_1_kernel_v-
)assignvariableop_82_adam_channel_1_bias_v1
-assignvariableop_83_adam_dec_inner_0_kernel_v/
+assignvariableop_84_adam_dec_inner_0_bias_v1
-assignvariableop_85_adam_dec_inner_1_kernel_v/
+assignvariableop_86_adam_dec_inner_1_bias_v2
.assignvariableop_87_adam_dec_middle_0_kernel_v0
,assignvariableop_88_adam_dec_middle_0_bias_v2
.assignvariableop_89_adam_dec_middle_1_kernel_v0
,assignvariableop_90_adam_dec_middle_1_bias_v1
-assignvariableop_91_adam_dec_outer_0_kernel_v/
+assignvariableop_92_adam_dec_outer_0_bias_v1
-assignvariableop_93_adam_dec_outer_1_kernel_v/
+assignvariableop_94_adam_dec_outer_1_bias_v0
,assignvariableop_95_adam_dec_output_kernel_v.
*assignvariableop_96_adam_dec_output_bias_v
identity_98??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?,
value?,B?,bB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*p
dtypesf
d2b	2
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
AssignVariableOp_9AssignVariableOp&assignvariableop_9_enc_middle_0_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_enc_middle_0_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_enc_middle_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_enc_middle_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp&assignvariableop_13_enc_inner_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_enc_inner_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp&assignvariableop_15_enc_inner_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_enc_inner_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_channel_0_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_channel_0_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_channel_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_channel_1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_dec_inner_0_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_dec_inner_0_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp&assignvariableop_23_dec_inner_1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_dec_inner_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_dec_middle_0_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_dec_middle_0_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_dec_middle_1_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_dec_middle_1_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_dec_outer_0_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_dec_outer_0_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_dec_outer_1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_dec_outer_1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_dec_output_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dec_output_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp-assignvariableop_37_adam_enc_outer_0_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_enc_outer_0_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_enc_outer_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_enc_outer_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_enc_middle_0_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_enc_middle_0_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_enc_middle_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_enc_middle_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp-assignvariableop_45_adam_enc_inner_0_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_enc_inner_0_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_enc_inner_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_enc_inner_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_channel_0_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_channel_0_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_channel_1_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_channel_1_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_dec_inner_0_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_dec_inner_0_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_dec_inner_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_dec_inner_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_dec_middle_0_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp,assignvariableop_58_adam_dec_middle_0_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp.assignvariableop_59_adam_dec_middle_1_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_dec_middle_1_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp-assignvariableop_61_adam_dec_outer_0_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_dec_outer_0_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp-assignvariableop_63_adam_dec_outer_1_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp+assignvariableop_64_adam_dec_outer_1_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_dec_output_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dec_output_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp-assignvariableop_67_adam_enc_outer_0_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_enc_outer_0_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_enc_outer_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_enc_outer_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_enc_middle_0_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp,assignvariableop_72_adam_enc_middle_0_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp.assignvariableop_73_adam_enc_middle_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp,assignvariableop_74_adam_enc_middle_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp-assignvariableop_75_adam_enc_inner_0_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp+assignvariableop_76_adam_enc_inner_0_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp-assignvariableop_77_adam_enc_inner_1_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_enc_inner_1_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_channel_0_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_channel_0_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_channel_1_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_channel_1_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp-assignvariableop_83_adam_dec_inner_0_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp+assignvariableop_84_adam_dec_inner_0_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp-assignvariableop_85_adam_dec_inner_1_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adam_dec_inner_1_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp.assignvariableop_87_adam_dec_middle_0_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp,assignvariableop_88_adam_dec_middle_0_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp.assignvariableop_89_adam_dec_middle_1_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp,assignvariableop_90_adam_dec_middle_1_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp-assignvariableop_91_adam_dec_outer_0_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp+assignvariableop_92_adam_dec_outer_0_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp-assignvariableop_93_adam_dec_outer_1_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp+assignvariableop_94_adam_dec_outer_1_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp,assignvariableop_95_adam_dec_output_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_dec_output_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_969
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_97Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_97?
Identity_98IdentityIdentity_97:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96*
T0*
_output_shapes
: 2
Identity_98"#
identity_98Identity_98:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
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
AssignVariableOp_96AssignVariableOp_96:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_enc_inner_0_layer_call_fn_168966

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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_1668352
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_169037

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
?J
?

C__inference_model_3_layer_call_and_return_conditional_losses_168798
inputs_0
inputs_1.
*dec_inner_1_matmul_readvariableop_resource/
+dec_inner_1_biasadd_readvariableop_resource.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_1_matmul_readvariableop_resource0
,dec_middle_1_biasadd_readvariableop_resource/
+dec_middle_0_matmul_readvariableop_resource0
,dec_middle_0_biasadd_readvariableop_resource.
*dec_outer_0_matmul_readvariableop_resource/
+dec_outer_0_biasadd_readvariableop_resource.
*dec_outer_1_matmul_readvariableop_resource/
+dec_outer_1_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?"dec_inner_1/BiasAdd/ReadVariableOp?!dec_inner_1/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?#dec_middle_1/BiasAdd/ReadVariableOp?"dec_middle_1/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?"dec_outer_1/BiasAdd/ReadVariableOp?!dec_outer_1/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
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
dec_outer_1/Relup
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2dec_outer_0/Relu:activations:0dec_outer_1/Relu:activations:0tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
tf.concat/concat?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.concat/concat:output:0(dec_output/MatMul/ReadVariableOp:value:0*
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
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp#^dec_inner_1/BiasAdd/ReadVariableOp"^dec_inner_1/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp$^dec_middle_1/BiasAdd/ReadVariableOp#^dec_middle_1/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp#^dec_outer_1/BiasAdd/ReadVariableOp"^dec_outer_1/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2H
"dec_inner_1/BiasAdd/ReadVariableOp"dec_inner_1/BiasAdd/ReadVariableOp2F
!dec_inner_1/MatMul/ReadVariableOp!dec_inner_1/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2J
#dec_middle_1/BiasAdd/ReadVariableOp#dec_middle_1/BiasAdd/ReadVariableOp2H
"dec_middle_1/MatMul/ReadVariableOp"dec_middle_1/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2H
"dec_outer_1/BiasAdd/ReadVariableOp"dec_outer_1/BiasAdd/ReadVariableOp2F
!dec_outer_1/MatMul/ReadVariableOp!dec_outer_1/MatMul/ReadVariableOp2F
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
inputs/1
?*
?
C__inference_model_3_layer_call_and_return_conditional_losses_167406

inputs
inputs_1
dec_inner_1_167368
dec_inner_1_167370
dec_inner_0_167373
dec_inner_0_167375
dec_middle_1_167378
dec_middle_1_167380
dec_middle_0_167383
dec_middle_0_167385
dec_outer_0_167388
dec_outer_0_167390
dec_outer_1_167393
dec_outer_1_167395
dec_output_167400
dec_output_167402
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1dec_inner_1_167368dec_inner_1_167370*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_1671372%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_167373dec_inner_0_167375*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_1671642%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_167378dec_middle_1_167380*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_1671912&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_167383dec_middle_0_167385*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_1672182&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_167388dec_outer_0_167390*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_1672452%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_167393dec_outer_1_167395*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_1672722%
#dec_outer_1/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
tf.concat/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0dec_output_167400dec_output_167402*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_1673012$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_channel_0_layer_call_and_return_conditional_losses_168997

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
+__inference_dec_output_layer_call_fn_169166

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
F__inference_dec_output_layer_call_and_return_conditional_losses_1673012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
E__inference_channel_0_layer_call_and_return_conditional_losses_166889

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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_166700

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
?
(__inference_model_2_layer_call_fn_168647

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

unknown_14
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670002
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

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
װ
?)
__inference__traced_save_169480
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop1
-savev2_enc_outer_0_kernel_read_readvariableop/
+savev2_enc_outer_0_bias_read_readvariableop1
-savev2_enc_outer_1_kernel_read_readvariableop/
+savev2_enc_outer_1_bias_read_readvariableop2
.savev2_enc_middle_0_kernel_read_readvariableop0
,savev2_enc_middle_0_bias_read_readvariableop2
.savev2_enc_middle_1_kernel_read_readvariableop0
,savev2_enc_middle_1_bias_read_readvariableop1
-savev2_enc_inner_0_kernel_read_readvariableop/
+savev2_enc_inner_0_bias_read_readvariableop1
-savev2_enc_inner_1_kernel_read_readvariableop/
+savev2_enc_inner_1_bias_read_readvariableop/
+savev2_channel_0_kernel_read_readvariableop-
)savev2_channel_0_bias_read_readvariableop/
+savev2_channel_1_kernel_read_readvariableop-
)savev2_channel_1_bias_read_readvariableop1
-savev2_dec_inner_0_kernel_read_readvariableop/
+savev2_dec_inner_0_bias_read_readvariableop1
-savev2_dec_inner_1_kernel_read_readvariableop/
+savev2_dec_inner_1_bias_read_readvariableop2
.savev2_dec_middle_0_kernel_read_readvariableop0
,savev2_dec_middle_0_bias_read_readvariableop2
.savev2_dec_middle_1_kernel_read_readvariableop0
,savev2_dec_middle_1_bias_read_readvariableop1
-savev2_dec_outer_0_kernel_read_readvariableop/
+savev2_dec_outer_0_bias_read_readvariableop1
-savev2_dec_outer_1_kernel_read_readvariableop/
+savev2_dec_outer_1_bias_read_readvariableop0
,savev2_dec_output_kernel_read_readvariableop.
*savev2_dec_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_0_bias_m_read_readvariableop8
4savev2_adam_enc_outer_1_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_1_bias_m_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_0_bias_m_read_readvariableop9
5savev2_adam_enc_middle_1_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_1_bias_m_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_0_bias_m_read_readvariableop8
4savev2_adam_enc_inner_1_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_1_bias_m_read_readvariableop6
2savev2_adam_channel_0_kernel_m_read_readvariableop4
0savev2_adam_channel_0_bias_m_read_readvariableop6
2savev2_adam_channel_1_kernel_m_read_readvariableop4
0savev2_adam_channel_1_bias_m_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_0_bias_m_read_readvariableop8
4savev2_adam_dec_inner_1_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_1_bias_m_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_0_bias_m_read_readvariableop9
5savev2_adam_dec_middle_1_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_1_bias_m_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_0_bias_m_read_readvariableop8
4savev2_adam_dec_outer_1_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_1_bias_m_read_readvariableop7
3savev2_adam_dec_output_kernel_m_read_readvariableop5
1savev2_adam_dec_output_bias_m_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_0_bias_v_read_readvariableop8
4savev2_adam_enc_outer_1_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_1_bias_v_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_0_bias_v_read_readvariableop9
5savev2_adam_enc_middle_1_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_1_bias_v_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_0_bias_v_read_readvariableop8
4savev2_adam_enc_inner_1_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_1_bias_v_read_readvariableop6
2savev2_adam_channel_0_kernel_v_read_readvariableop4
0savev2_adam_channel_0_bias_v_read_readvariableop6
2savev2_adam_channel_1_kernel_v_read_readvariableop4
0savev2_adam_channel_1_bias_v_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_0_bias_v_read_readvariableop8
4savev2_adam_dec_inner_1_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_1_bias_v_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_0_bias_v_read_readvariableop9
5savev2_adam_dec_middle_1_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_1_bias_v_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_0_bias_v_read_readvariableop8
4savev2_adam_dec_outer_1_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_1_bias_v_read_readvariableop7
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
ShardedFilename?-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?,
value?,B?,bB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop-savev2_enc_outer_0_kernel_read_readvariableop+savev2_enc_outer_0_bias_read_readvariableop-savev2_enc_outer_1_kernel_read_readvariableop+savev2_enc_outer_1_bias_read_readvariableop.savev2_enc_middle_0_kernel_read_readvariableop,savev2_enc_middle_0_bias_read_readvariableop.savev2_enc_middle_1_kernel_read_readvariableop,savev2_enc_middle_1_bias_read_readvariableop-savev2_enc_inner_0_kernel_read_readvariableop+savev2_enc_inner_0_bias_read_readvariableop-savev2_enc_inner_1_kernel_read_readvariableop+savev2_enc_inner_1_bias_read_readvariableop+savev2_channel_0_kernel_read_readvariableop)savev2_channel_0_bias_read_readvariableop+savev2_channel_1_kernel_read_readvariableop)savev2_channel_1_bias_read_readvariableop-savev2_dec_inner_0_kernel_read_readvariableop+savev2_dec_inner_0_bias_read_readvariableop-savev2_dec_inner_1_kernel_read_readvariableop+savev2_dec_inner_1_bias_read_readvariableop.savev2_dec_middle_0_kernel_read_readvariableop,savev2_dec_middle_0_bias_read_readvariableop.savev2_dec_middle_1_kernel_read_readvariableop,savev2_dec_middle_1_bias_read_readvariableop-savev2_dec_outer_0_kernel_read_readvariableop+savev2_dec_outer_0_bias_read_readvariableop-savev2_dec_outer_1_kernel_read_readvariableop+savev2_dec_outer_1_bias_read_readvariableop,savev2_dec_output_kernel_read_readvariableop*savev2_dec_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_enc_outer_0_kernel_m_read_readvariableop2savev2_adam_enc_outer_0_bias_m_read_readvariableop4savev2_adam_enc_outer_1_kernel_m_read_readvariableop2savev2_adam_enc_outer_1_bias_m_read_readvariableop5savev2_adam_enc_middle_0_kernel_m_read_readvariableop3savev2_adam_enc_middle_0_bias_m_read_readvariableop5savev2_adam_enc_middle_1_kernel_m_read_readvariableop3savev2_adam_enc_middle_1_bias_m_read_readvariableop4savev2_adam_enc_inner_0_kernel_m_read_readvariableop2savev2_adam_enc_inner_0_bias_m_read_readvariableop4savev2_adam_enc_inner_1_kernel_m_read_readvariableop2savev2_adam_enc_inner_1_bias_m_read_readvariableop2savev2_adam_channel_0_kernel_m_read_readvariableop0savev2_adam_channel_0_bias_m_read_readvariableop2savev2_adam_channel_1_kernel_m_read_readvariableop0savev2_adam_channel_1_bias_m_read_readvariableop4savev2_adam_dec_inner_0_kernel_m_read_readvariableop2savev2_adam_dec_inner_0_bias_m_read_readvariableop4savev2_adam_dec_inner_1_kernel_m_read_readvariableop2savev2_adam_dec_inner_1_bias_m_read_readvariableop5savev2_adam_dec_middle_0_kernel_m_read_readvariableop3savev2_adam_dec_middle_0_bias_m_read_readvariableop5savev2_adam_dec_middle_1_kernel_m_read_readvariableop3savev2_adam_dec_middle_1_bias_m_read_readvariableop4savev2_adam_dec_outer_0_kernel_m_read_readvariableop2savev2_adam_dec_outer_0_bias_m_read_readvariableop4savev2_adam_dec_outer_1_kernel_m_read_readvariableop2savev2_adam_dec_outer_1_bias_m_read_readvariableop3savev2_adam_dec_output_kernel_m_read_readvariableop1savev2_adam_dec_output_bias_m_read_readvariableop4savev2_adam_enc_outer_0_kernel_v_read_readvariableop2savev2_adam_enc_outer_0_bias_v_read_readvariableop4savev2_adam_enc_outer_1_kernel_v_read_readvariableop2savev2_adam_enc_outer_1_bias_v_read_readvariableop5savev2_adam_enc_middle_0_kernel_v_read_readvariableop3savev2_adam_enc_middle_0_bias_v_read_readvariableop5savev2_adam_enc_middle_1_kernel_v_read_readvariableop3savev2_adam_enc_middle_1_bias_v_read_readvariableop4savev2_adam_enc_inner_0_kernel_v_read_readvariableop2savev2_adam_enc_inner_0_bias_v_read_readvariableop4savev2_adam_enc_inner_1_kernel_v_read_readvariableop2savev2_adam_enc_inner_1_bias_v_read_readvariableop2savev2_adam_channel_0_kernel_v_read_readvariableop0savev2_adam_channel_0_bias_v_read_readvariableop2savev2_adam_channel_1_kernel_v_read_readvariableop0savev2_adam_channel_1_bias_v_read_readvariableop4savev2_adam_dec_inner_0_kernel_v_read_readvariableop2savev2_adam_dec_inner_0_bias_v_read_readvariableop4savev2_adam_dec_inner_1_kernel_v_read_readvariableop2savev2_adam_dec_inner_1_bias_v_read_readvariableop5savev2_adam_dec_middle_0_kernel_v_read_readvariableop3savev2_adam_dec_middle_0_bias_v_read_readvariableop5savev2_adam_dec_middle_1_kernel_v_read_readvariableop3savev2_adam_dec_middle_1_bias_v_read_readvariableop4savev2_adam_dec_outer_0_kernel_v_read_readvariableop2savev2_adam_dec_outer_0_bias_v_read_readvariableop4savev2_adam_dec_outer_1_kernel_v_read_readvariableop2savev2_adam_dec_outer_1_bias_v_read_readvariableop3savev2_adam_dec_output_kernel_v_read_readvariableop1savev2_adam_dec_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *p
dtypesf
d2b	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	?<:<:	?<:<:<2:2:<2:2:2(:(:2(:(:(::(::(:(:(:(:(<:<:(<:<:<<:<:<<:<:	x?:?: : :	?<:<:	?<:<:<2:2:<2:2:2(:(:2(:(:(::(::(:(:(:(:(<:<:(<:<:<<:<:<<:<:	x?:?:	?<:<:	?<:<:<2:2:<2:2:2(:(:2(:(:(::(::(:(:(:(:(<:<:(<:<:<<:<:<<:<:	x?:?: 2(
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
:<:$
 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:(<: 

_output_shapes
:<:$ 

_output_shapes

:(<: 

_output_shapes
:<:$ 

_output_shapes

:<<: 

_output_shapes
:<:$  

_output_shapes

:<<: !

_output_shapes
:<:%"!

_output_shapes
:	x?:!#

_output_shapes	
:?:$

_output_shapes
: :%

_output_shapes
: :%&!

_output_shapes
:	?<: '

_output_shapes
:<:%(!

_output_shapes
:	?<: )

_output_shapes
:<:$* 

_output_shapes

:<2: +

_output_shapes
:2:$, 

_output_shapes

:<2: -

_output_shapes
:2:$. 

_output_shapes

:2(: /

_output_shapes
:(:$0 

_output_shapes

:2(: 1

_output_shapes
:(:$2 

_output_shapes

:(: 3

_output_shapes
::$4 

_output_shapes

:(: 5

_output_shapes
::$6 

_output_shapes

:(: 7

_output_shapes
:(:$8 

_output_shapes

:(: 9

_output_shapes
:(:$: 

_output_shapes

:(<: ;

_output_shapes
:<:$< 

_output_shapes

:(<: =

_output_shapes
:<:$> 

_output_shapes

:<<: ?

_output_shapes
:<:$@ 

_output_shapes

:<<: A

_output_shapes
:<:%B!

_output_shapes
:	x?:!C

_output_shapes	
:?:%D!

_output_shapes
:	?<: E

_output_shapes
:<:%F!

_output_shapes
:	?<: G

_output_shapes
:<:$H 

_output_shapes

:<2: I

_output_shapes
:2:$J 

_output_shapes

:<2: K

_output_shapes
:2:$L 

_output_shapes

:2(: M

_output_shapes
:(:$N 

_output_shapes

:2(: O

_output_shapes
:(:$P 

_output_shapes

:(: Q

_output_shapes
::$R 

_output_shapes

:(: S

_output_shapes
::$T 

_output_shapes

:(: U

_output_shapes
:(:$V 

_output_shapes

:(: W

_output_shapes
:(:$X 

_output_shapes

:(<: Y

_output_shapes
:<:$Z 

_output_shapes

:(<: [

_output_shapes
:<:$\ 

_output_shapes

:<<: ]

_output_shapes
:<:$^ 

_output_shapes

:<<: _

_output_shapes
:<:%`!

_output_shapes
:	x?:!a

_output_shapes	
:?:b

_output_shapes
: 
?0
?
C__inference_model_2_layer_call_and_return_conditional_losses_167000

inputs
enc_outer_1_166958
enc_outer_1_166960
enc_outer_0_166963
enc_outer_0_166965
enc_middle_1_166968
enc_middle_1_166970
enc_middle_0_166973
enc_middle_0_166975
enc_inner_1_166978
enc_inner_1_166980
enc_inner_0_166983
enc_inner_0_166985
channel_1_166988
channel_1_166990
channel_0_166993
channel_0_166995
identity

identity_1??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_1_166958enc_outer_1_166960*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_1667002%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_166963enc_outer_0_166965*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_1667272%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_166968enc_middle_1_166970*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_1667542&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_166973enc_middle_0_166975*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_1667812&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_166978enc_inner_1_166980*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_1668082%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_166983enc_inner_0_166985*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_1668352%
#enc_inner_0/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_166988channel_1_166990*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_1668622#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_166993channel_0_166995*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_1668892#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dec_output_layer_call_and_return_conditional_losses_169157

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x?*
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
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_167191

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
?
(__inference_model_3_layer_call_fn_167437
decoder_input_0
decoder_input_1
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0decoder_input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::22
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
_user_specified_namedecoder_input_1
?
?
,__inference_dec_inner_1_layer_call_fn_169066

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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_1671372
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
F__inference_dec_output_layer_call_and_return_conditional_losses_167301

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x?*
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
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
(__inference_model_2_layer_call_fn_167037
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

unknown_14
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670002
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

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?
?
.__inference_autoencoder_1_layer_call_fn_167927
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

unknown_28
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_1678642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_166835

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
E__inference_channel_1_layer_call_and_return_conditional_losses_169017

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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_169117

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
?U
?
C__inference_model_2_layer_call_and_return_conditional_losses_168608

inputs.
*enc_outer_1_matmul_readvariableop_resource/
+enc_outer_1_biasadd_readvariableop_resource.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_1_matmul_readvariableop_resource0
,enc_middle_1_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_1_matmul_readvariableop_resource/
+enc_inner_1_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_1_matmul_readvariableop_resource-
)channel_1_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity

identity_1?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp? channel_1/BiasAdd/ReadVariableOp?channel_1/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?"enc_inner_1/BiasAdd/ReadVariableOp?!enc_inner_1/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?#enc_middle_1/BiasAdd/ReadVariableOp?"enc_middle_1/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?"enc_outer_1/BiasAdd/ReadVariableOp?!enc_outer_1/MatMul/ReadVariableOp?
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
channel_0/Softsign?
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity channel_1/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2D
 channel_1/BiasAdd/ReadVariableOp channel_1/BiasAdd/ReadVariableOp2B
channel_1/MatMul/ReadVariableOpchannel_1/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2H
"enc_inner_1/BiasAdd/ReadVariableOp"enc_inner_1/BiasAdd/ReadVariableOp2F
!enc_inner_1/MatMul/ReadVariableOp!enc_inner_1/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2J
#enc_middle_1/BiasAdd/ReadVariableOp#enc_middle_1/BiasAdd/ReadVariableOp2H
"enc_middle_1/MatMul/ReadVariableOp"enc_middle_1/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp2H
"enc_outer_1/BiasAdd/ReadVariableOp"enc_outer_1/BiasAdd/ReadVariableOp2F
!enc_outer_1/MatMul/ReadVariableOp!enc_outer_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_168917

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
?
?
.__inference_autoencoder_1_layer_call_fn_168421
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

unknown_28
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_1678642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_169057

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
?+
?
C__inference_model_3_layer_call_and_return_conditional_losses_167360
decoder_input_0
decoder_input_1
dec_inner_1_167322
dec_inner_1_167324
dec_inner_0_167327
dec_inner_0_167329
dec_middle_1_167332
dec_middle_1_167334
dec_middle_0_167337
dec_middle_0_167339
dec_outer_0_167342
dec_outer_0_167344
dec_outer_1_167347
dec_outer_1_167349
dec_output_167354
dec_output_167356
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_1dec_inner_1_167322dec_inner_1_167324*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_1671372%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_167327dec_inner_0_167329*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_1671642%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_167332dec_middle_1_167334*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_1671912&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_167337dec_middle_0_167339*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_1672182&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_167342dec_outer_0_167344*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_1672452%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_167347dec_outer_1_167349*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_1672722%
#dec_outer_1/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
tf.concat/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0dec_output_167354dec_output_167356*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_1673012$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2H
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
_user_specified_namedecoder_input_1
?

?
(__inference_model_3_layer_call_fn_168832
inputs_0
inputs_1
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::22
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
inputs/1
?	
?
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_167164

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
,__inference_enc_outer_1_layer_call_fn_168906

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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_1667002
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_167137

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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_167218

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
?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167727
input_1
model_2_167595
model_2_167597
model_2_167599
model_2_167601
model_2_167603
model_2_167605
model_2_167607
model_2_167609
model_2_167611
model_2_167613
model_2_167615
model_2_167617
model_2_167619
model_2_167621
model_2_167623
model_2_167625
model_3_167697
model_3_167699
model_3_167701
model_3_167703
model_3_167705
model_3_167707
model_3_167709
model_3_167711
model_3_167713
model_3_167715
model_3_167717
model_3_167719
model_3_167721
model_3_167723
identity??model_2/StatefulPartitionedCall?model_3/StatefulPartitionedCall?
model_2/StatefulPartitionedCallStatefulPartitionedCallinput_1model_2_167595model_2_167597model_2_167599model_2_167601model_2_167603model_2_167605model_2_167607model_2_167609model_2_167611model_2_167613model_2_167615model_2_167617model_2_167619model_2_167621model_2_167623model_2_167625*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670002!
model_2/StatefulPartitionedCall?
model_3/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0(model_2/StatefulPartitionedCall:output:1model_3_167697model_3_167699model_3_167701model_3_167703model_3_167705model_3_167707model_3_167709model_3_167711model_3_167713model_3_167715model_3_167717model_3_167719model_3_167721model_3_167723*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674062!
model_3/StatefulPartitionedCall?
IdentityIdentity(model_3/StatefulPartitionedCall:output:0 ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
.__inference_autoencoder_1_layer_call_fn_168059
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

unknown_28
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_1679962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_166808

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
?0
?
C__inference_model_2_layer_call_and_return_conditional_losses_167084

inputs
enc_outer_1_167042
enc_outer_1_167044
enc_outer_0_167047
enc_outer_0_167049
enc_middle_1_167052
enc_middle_1_167054
enc_middle_0_167057
enc_middle_0_167059
enc_inner_1_167062
enc_inner_1_167064
enc_inner_0_167067
enc_inner_0_167069
channel_1_167072
channel_1_167074
channel_0_167077
channel_0_167079
identity

identity_1??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_1_167042enc_outer_1_167044*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_1667002%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_167047enc_outer_0_167049*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_1667272%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_167052enc_middle_1_167054*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_1667542&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_167057enc_middle_0_167059*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_1667812&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_167062enc_inner_1_167064*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_1668082%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_167067enc_inner_0_167069*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_1668352%
#enc_inner_0/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_167072channel_1_167074*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_1668622#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_167077channel_0_167079*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_1668892#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_enc_inner_1_layer_call_fn_168986

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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_1668082
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
?*
?
C__inference_model_3_layer_call_and_return_conditional_losses_167482

inputs
inputs_1
dec_inner_1_167444
dec_inner_1_167446
dec_inner_0_167449
dec_inner_0_167451
dec_middle_1_167454
dec_middle_1_167456
dec_middle_0_167459
dec_middle_0_167461
dec_outer_0_167464
dec_outer_0_167466
dec_outer_1_167469
dec_outer_1_167471
dec_output_167476
dec_output_167478
identity??#dec_inner_0/StatefulPartitionedCall?#dec_inner_1/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?$dec_middle_1/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?#dec_outer_1/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1dec_inner_1_167444dec_inner_1_167446*
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_1671372%
#dec_inner_1/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_167449dec_inner_0_167451*
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_1671642%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_1/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_1/StatefulPartitionedCall:output:0dec_middle_1_167454dec_middle_1_167456*
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_1671912&
$dec_middle_1/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_167459dec_middle_0_167461*
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_1672182&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_167464dec_outer_0_167466*
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_1672452%
#dec_outer_0/StatefulPartitionedCall?
#dec_outer_1/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_1/StatefulPartitionedCall:output:0dec_outer_1_167469dec_outer_1_167471*
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_1672722%
#dec_outer_1/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2,dec_outer_0/StatefulPartitionedCall:output:0,dec_outer_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
tf.concat/concat?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0dec_output_167476dec_output_167478*
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
F__inference_dec_output_layer_call_and_return_conditional_losses_1673012$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall$^dec_inner_1/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall%^dec_middle_1/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall$^dec_outer_1/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2J
#dec_inner_1/StatefulPartitionedCall#dec_inner_1/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2L
$dec_middle_1/StatefulPartitionedCall$dec_middle_1/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2J
#dec_outer_1/StatefulPartitionedCall#dec_outer_1/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_168977

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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_166781

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
?
(__inference_model_2_layer_call_fn_168686

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

unknown_14
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670842
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

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_166754

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
?J
?

C__inference_model_3_layer_call_and_return_conditional_losses_168742
inputs_0
inputs_1.
*dec_inner_1_matmul_readvariableop_resource/
+dec_inner_1_biasadd_readvariableop_resource.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_1_matmul_readvariableop_resource0
,dec_middle_1_biasadd_readvariableop_resource/
+dec_middle_0_matmul_readvariableop_resource0
,dec_middle_0_biasadd_readvariableop_resource.
*dec_outer_0_matmul_readvariableop_resource/
+dec_outer_0_biasadd_readvariableop_resource.
*dec_outer_1_matmul_readvariableop_resource/
+dec_outer_1_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?"dec_inner_1/BiasAdd/ReadVariableOp?!dec_inner_1/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?#dec_middle_1/BiasAdd/ReadVariableOp?"dec_middle_1/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?"dec_outer_1/BiasAdd/ReadVariableOp?!dec_outer_1/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
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
dec_outer_1/Relup
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2dec_outer_0/Relu:activations:0dec_outer_1/Relu:activations:0tf.concat/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????x2
tf.concat/concat?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.concat/concat:output:0(dec_output/MatMul/ReadVariableOp:value:0*
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
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp#^dec_inner_1/BiasAdd/ReadVariableOp"^dec_inner_1/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp$^dec_middle_1/BiasAdd/ReadVariableOp#^dec_middle_1/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp#^dec_outer_1/BiasAdd/ReadVariableOp"^dec_outer_1/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2H
"dec_inner_1/BiasAdd/ReadVariableOp"dec_inner_1/BiasAdd/ReadVariableOp2F
!dec_inner_1/MatMul/ReadVariableOp!dec_inner_1/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2J
#dec_middle_1/BiasAdd/ReadVariableOp#dec_middle_1/BiasAdd/ReadVariableOp2H
"dec_middle_1/MatMul/ReadVariableOp"dec_middle_1/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2H
"dec_outer_1/BiasAdd/ReadVariableOp"dec_outer_1/BiasAdd/ReadVariableOp2F
!dec_outer_1/MatMul/ReadVariableOp!dec_outer_1/MatMul/ReadVariableOp2F
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
inputs/1
?

*__inference_channel_1_layer_call_fn_169026

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
E__inference_channel_1_layer_call_and_return_conditional_losses_1668622
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
-__inference_dec_middle_1_layer_call_fn_169106

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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_1671912
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
E__inference_channel_1_layer_call_and_return_conditional_losses_166862

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
?
(__inference_model_2_layer_call_fn_167121
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

unknown_14
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1670842
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

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?	
?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_168897

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
?U
?
C__inference_model_2_layer_call_and_return_conditional_losses_168547

inputs.
*enc_outer_1_matmul_readvariableop_resource/
+enc_outer_1_biasadd_readvariableop_resource.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_1_matmul_readvariableop_resource0
,enc_middle_1_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_1_matmul_readvariableop_resource/
+enc_inner_1_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_1_matmul_readvariableop_resource-
)channel_1_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity

identity_1?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp? channel_1/BiasAdd/ReadVariableOp?channel_1/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?"enc_inner_1/BiasAdd/ReadVariableOp?!enc_inner_1/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?#enc_middle_1/BiasAdd/ReadVariableOp?"enc_middle_1/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?"enc_outer_1/BiasAdd/ReadVariableOp?!enc_outer_1/MatMul/ReadVariableOp?
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
channel_0/Softsign?
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity channel_1/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp!^channel_1/BiasAdd/ReadVariableOp ^channel_1/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp#^enc_inner_1/BiasAdd/ReadVariableOp"^enc_inner_1/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp$^enc_middle_1/BiasAdd/ReadVariableOp#^enc_middle_1/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp#^enc_outer_1/BiasAdd/ReadVariableOp"^enc_outer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2D
 channel_1/BiasAdd/ReadVariableOp channel_1/BiasAdd/ReadVariableOp2B
channel_1/MatMul/ReadVariableOpchannel_1/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2H
"enc_inner_1/BiasAdd/ReadVariableOp"enc_inner_1/BiasAdd/ReadVariableOp2F
!enc_inner_1/MatMul/ReadVariableOp!enc_inner_1/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2J
#enc_middle_1/BiasAdd/ReadVariableOp#enc_middle_1/BiasAdd/ReadVariableOp2H
"enc_middle_1/MatMul/ReadVariableOp"enc_middle_1/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp2H
"enc_outer_1/BiasAdd/ReadVariableOp"enc_outer_1/BiasAdd/ReadVariableOp2F
!enc_outer_1/MatMul/ReadVariableOp!enc_outer_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_model_3_layer_call_fn_168866
inputs_0
inputs_1
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_1674822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????:?????????::::::::::::::22
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
inputs/1
?
?
-__inference_enc_middle_1_layer_call_fn_168946

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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_1667542
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_167272

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
-__inference_enc_middle_0_layer_call_fn_168926

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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_1667812
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_169077

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
,__inference_dec_outer_1_layer_call_fn_169146

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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_1672722
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
?0
?
C__inference_model_2_layer_call_and_return_conditional_losses_166952
encoder_input
enc_outer_1_166910
enc_outer_1_166912
enc_outer_0_166915
enc_outer_0_166917
enc_middle_1_166920
enc_middle_1_166922
enc_middle_0_166925
enc_middle_0_166927
enc_inner_1_166930
enc_inner_1_166932
enc_inner_0_166935
enc_inner_0_166937
channel_1_166940
channel_1_166942
channel_0_166945
channel_0_166947
identity

identity_1??!channel_0/StatefulPartitionedCall?!channel_1/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?#enc_inner_1/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?$enc_middle_1/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?#enc_outer_1/StatefulPartitionedCall?
#enc_outer_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_1_166910enc_outer_1_166912*
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_1667002%
#enc_outer_1/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_166915enc_outer_0_166917*
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_1667272%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_1/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_1/StatefulPartitionedCall:output:0enc_middle_1_166920enc_middle_1_166922*
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_1667542&
$enc_middle_1/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_166925enc_middle_0_166927*
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_1667812&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_1/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_1/StatefulPartitionedCall:output:0enc_inner_1_166930enc_inner_1_166932*
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_1668082%
#enc_inner_1/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_166935enc_inner_0_166937*
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_1668352%
#enc_inner_0/StatefulPartitionedCall?
!channel_1/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_1/StatefulPartitionedCall:output:0channel_1_166940channel_1_166942*
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
E__inference_channel_1_layer_call_and_return_conditional_losses_1668622#
!channel_1/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_166945channel_0_166947*
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
E__inference_channel_0_layer_call_and_return_conditional_losses_1668892#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*channel_1/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall"^channel_1/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall$^enc_inner_1/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall%^enc_middle_1/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall$^enc_outer_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*g
_input_shapesV
T:??????????::::::::::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2F
!channel_1/StatefulPartitionedCall!channel_1/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2J
#enc_inner_1/StatefulPartitionedCall#enc_inner_1/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2L
$enc_middle_1/StatefulPartitionedCall$enc_middle_1/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall2J
#enc_outer_1/StatefulPartitionedCall#enc_outer_1/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input"?L
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
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
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
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?L
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
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?I
_tf_keras_network?H{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_1", "inbound_nodes": [[["enc_outer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_1", "inbound_nodes": [[["enc_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_1", "inbound_nodes": [[["enc_inner_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0], ["channel_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 784]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_1", "inbound_nodes": [[["enc_outer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_1", "inbound_nodes": [[["enc_middle_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_1", "inbound_nodes": [[["enc_inner_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0], ["channel_1", 0, 0]]}}}
?N
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
layer_with_weights-6
layer-9
 	variables
!trainable_variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?J
_tf_keras_network?J{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}, "name": "decoder_input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_1", "inbound_nodes": [[["decoder_input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_1", "inbound_nodes": [[["dec_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_1", "inbound_nodes": [[["dec_middle_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["dec_outer_0", 0, 0, {"axis": 1}], ["dec_outer_1", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.concat", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0], ["decoder_input_1", 0, 0]], "output_layers": [["dec_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}, "name": "decoder_input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_1", "inbound_nodes": [[["decoder_input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_1", "inbound_nodes": [[["dec_inner_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_1", "inbound_nodes": [[["dec_middle_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["dec_outer_0", 0, 0, {"axis": 1}], ["dec_outer_1", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.concat", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0], ["decoder_input_1", 0, 0]], "output_layers": [["dec_output", 0, 0]]}}}
?
$iter

%beta_1

&beta_2
	'decay
(learning_rate)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?"
	optimizer
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29"
trackable_list_wrapper
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Gmetrics
Hnon_trainable_variables
Ilayer_regularization_losses
trainable_variables
Jlayer_metrics
regularization_losses

Klayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
?

)kernel
*bias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

+kernel
,bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

-kernel
.bias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

/kernel
0bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

1kernel
2bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

5kernel
6bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

7kernel
8bias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815"
trackable_list_wrapper
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
lmetrics
mnon_trainable_variables
nlayer_regularization_losses
trainable_variables
olayer_metrics
regularization_losses

players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_1"}}
?

9kernel
:bias
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

;kernel
<bias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

=kernel
>bias
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

?kernel
@bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Ckernel
Dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_1", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
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
F13"
trackable_list_wrapper
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
F13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 	variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
!trainable_variables
?layer_metrics
"regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
%:#<22enc_middle_0/kernel
:22enc_middle_0/bias
%:#<22enc_middle_1/kernel
:22enc_middle_1/bias
$:"2(2enc_inner_0/kernel
:(2enc_inner_0/bias
$:"2(2enc_inner_1/kernel
:(2enc_inner_1/bias
": (2channel_0/kernel
:2channel_0/bias
": (2channel_1/kernel
:2channel_1/bias
$:"(2dec_inner_0/kernel
:(2dec_inner_0/bias
$:"(2dec_inner_1/kernel
:(2dec_inner_1/bias
%:#(<2dec_middle_0/kernel
:<2dec_middle_0/bias
%:#(<2dec_middle_1/kernel
:<2dec_middle_1/bias
$:"<<2dec_outer_0/kernel
:<2dec_outer_0/bias
$:"<<2dec_outer_1/kernel
:<2dec_outer_1/bias
$:"	x?2dec_output/kernel
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
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ltrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
M	variables
?layer_metrics
Nregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ptrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
Q	variables
?layer_metrics
Rregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ttrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
U	variables
?layer_metrics
Vregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
Y	variables
?layer_metrics
Zregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
\trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
]	variables
?layer_metrics
^regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
`trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
a	variables
?layer_metrics
bregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
dtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
e	variables
?layer_metrics
fregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
htrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
i	variables
?layer_metrics
jregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
	0

1
2
3
4
5
6
7
8"
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
qtrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
r	variables
?layer_metrics
sregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
utrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
v	variables
?layer_metrics
wregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
ytrainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
z	variables
?layer_metrics
{regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
}trainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
~	variables
?layer_metrics
regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(	?<2Adam/enc_outer_0/kernel/m
#:!<2Adam/enc_outer_0/bias/m
*:(	?<2Adam/enc_outer_1/kernel/m
#:!<2Adam/enc_outer_1/bias/m
*:(<22Adam/enc_middle_0/kernel/m
$:"22Adam/enc_middle_0/bias/m
*:(<22Adam/enc_middle_1/kernel/m
$:"22Adam/enc_middle_1/bias/m
):'2(2Adam/enc_inner_0/kernel/m
#:!(2Adam/enc_inner_0/bias/m
):'2(2Adam/enc_inner_1/kernel/m
#:!(2Adam/enc_inner_1/bias/m
':%(2Adam/channel_0/kernel/m
!:2Adam/channel_0/bias/m
':%(2Adam/channel_1/kernel/m
!:2Adam/channel_1/bias/m
):'(2Adam/dec_inner_0/kernel/m
#:!(2Adam/dec_inner_0/bias/m
):'(2Adam/dec_inner_1/kernel/m
#:!(2Adam/dec_inner_1/bias/m
*:((<2Adam/dec_middle_0/kernel/m
$:"<2Adam/dec_middle_0/bias/m
*:((<2Adam/dec_middle_1/kernel/m
$:"<2Adam/dec_middle_1/bias/m
):'<<2Adam/dec_outer_0/kernel/m
#:!<2Adam/dec_outer_0/bias/m
):'<<2Adam/dec_outer_1/kernel/m
#:!<2Adam/dec_outer_1/bias/m
):'	x?2Adam/dec_output/kernel/m
#:!?2Adam/dec_output/bias/m
*:(	?<2Adam/enc_outer_0/kernel/v
#:!<2Adam/enc_outer_0/bias/v
*:(	?<2Adam/enc_outer_1/kernel/v
#:!<2Adam/enc_outer_1/bias/v
*:(<22Adam/enc_middle_0/kernel/v
$:"22Adam/enc_middle_0/bias/v
*:(<22Adam/enc_middle_1/kernel/v
$:"22Adam/enc_middle_1/bias/v
):'2(2Adam/enc_inner_0/kernel/v
#:!(2Adam/enc_inner_0/bias/v
):'2(2Adam/enc_inner_1/kernel/v
#:!(2Adam/enc_inner_1/bias/v
':%(2Adam/channel_0/kernel/v
!:2Adam/channel_0/bias/v
':%(2Adam/channel_1/kernel/v
!:2Adam/channel_1/bias/v
):'(2Adam/dec_inner_0/kernel/v
#:!(2Adam/dec_inner_0/bias/v
):'(2Adam/dec_inner_1/kernel/v
#:!(2Adam/dec_inner_1/bias/v
*:((<2Adam/dec_middle_0/kernel/v
$:"<2Adam/dec_middle_0/bias/v
*:((<2Adam/dec_middle_1/kernel/v
$:"<2Adam/dec_middle_1/bias/v
):'<<2Adam/dec_outer_0/kernel/v
#:!<2Adam/dec_outer_0/bias/v
):'<<2Adam/dec_outer_1/kernel/v
#:!<2Adam/dec_outer_1/bias/v
):'	x?2Adam/dec_output/kernel/v
#:!?2Adam/dec_output/bias/v
?2?
!__inference__wrapped_model_166685?
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
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167794
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_168356
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_168245
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167727?
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
.__inference_autoencoder_1_layer_call_fn_167927
.__inference_autoencoder_1_layer_call_fn_168421
.__inference_autoencoder_1_layer_call_fn_168486
.__inference_autoencoder_1_layer_call_fn_168059?
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
C__inference_model_2_layer_call_and_return_conditional_losses_168547
C__inference_model_2_layer_call_and_return_conditional_losses_166907
C__inference_model_2_layer_call_and_return_conditional_losses_168608
C__inference_model_2_layer_call_and_return_conditional_losses_166952?
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
(__inference_model_2_layer_call_fn_168686
(__inference_model_2_layer_call_fn_167037
(__inference_model_2_layer_call_fn_168647
(__inference_model_2_layer_call_fn_167121?
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
C__inference_model_3_layer_call_and_return_conditional_losses_167360
C__inference_model_3_layer_call_and_return_conditional_losses_168798
C__inference_model_3_layer_call_and_return_conditional_losses_168742
C__inference_model_3_layer_call_and_return_conditional_losses_167318?
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
(__inference_model_3_layer_call_fn_167513
(__inference_model_3_layer_call_fn_168832
(__inference_model_3_layer_call_fn_168866
(__inference_model_3_layer_call_fn_167437?
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
$__inference_signature_wrapper_168134input_1"?
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
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_168877?
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
,__inference_enc_outer_0_layer_call_fn_168886?
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
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_168897?
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
,__inference_enc_outer_1_layer_call_fn_168906?
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
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_168917?
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
-__inference_enc_middle_0_layer_call_fn_168926?
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
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_168937?
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
-__inference_enc_middle_1_layer_call_fn_168946?
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
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_168957?
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
,__inference_enc_inner_0_layer_call_fn_168966?
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
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_168977?
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
,__inference_enc_inner_1_layer_call_fn_168986?
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
E__inference_channel_0_layer_call_and_return_conditional_losses_168997?
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
*__inference_channel_0_layer_call_fn_169006?
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
E__inference_channel_1_layer_call_and_return_conditional_losses_169017?
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
*__inference_channel_1_layer_call_fn_169026?
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
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_169037?
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
,__inference_dec_inner_0_layer_call_fn_169046?
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
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_169057?
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
,__inference_dec_inner_1_layer_call_fn_169066?
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
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_169077?
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
-__inference_dec_middle_0_layer_call_fn_169086?
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
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_169097?
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
-__inference_dec_middle_1_layer_call_fn_169106?
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
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_169117?
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
,__inference_dec_outer_0_layer_call_fn_169126?
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
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_169137?
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
,__inference_dec_outer_1_layer_call_fn_169146?
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
F__inference_dec_output_layer_call_and_return_conditional_losses_169157?
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
+__inference_dec_output_layer_call_fn_169166?
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
!__inference__wrapped_model_166685?+,)*/0-.34127856;<9:?@=>ABCDEF1?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167727?+,)*/0-.34127856;<9:?@=>ABCDEFA?>
'?$
"?
input_1??????????
?

trainingp"&?#
?
0??????????
? ?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_167794?+,)*/0-.34127856;<9:?@=>ABCDEFA?>
'?$
"?
input_1??????????
?

trainingp "&?#
?
0??????????
? ?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_168245?+,)*/0-.34127856;<9:?@=>ABCDEF;?8
!?
?
x??????????
?

trainingp"&?#
?
0??????????
? ?
I__inference_autoencoder_1_layer_call_and_return_conditional_losses_168356?+,)*/0-.34127856;<9:?@=>ABCDEF;?8
!?
?
x??????????
?

trainingp "&?#
?
0??????????
? ?
.__inference_autoencoder_1_layer_call_fn_167927~+,)*/0-.34127856;<9:?@=>ABCDEFA?>
'?$
"?
input_1??????????
?

trainingp"????????????
.__inference_autoencoder_1_layer_call_fn_168059~+,)*/0-.34127856;<9:?@=>ABCDEFA?>
'?$
"?
input_1??????????
?

trainingp "????????????
.__inference_autoencoder_1_layer_call_fn_168421x+,)*/0-.34127856;<9:?@=>ABCDEF;?8
!?
?
x??????????
?

trainingp"????????????
.__inference_autoencoder_1_layer_call_fn_168486x+,)*/0-.34127856;<9:?@=>ABCDEF;?8
!?
?
x??????????
?

trainingp "????????????
E__inference_channel_0_layer_call_and_return_conditional_losses_168997\56/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_0_layer_call_fn_169006O56/?,
%?"
 ?
inputs?????????(
? "???????????
E__inference_channel_1_layer_call_and_return_conditional_losses_169017\78/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_channel_1_layer_call_fn_169026O78/?,
%?"
 ?
inputs?????????(
? "???????????
G__inference_dec_inner_0_layer_call_and_return_conditional_losses_169037\9:/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_0_layer_call_fn_169046O9:/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_inner_1_layer_call_and_return_conditional_losses_169057\;</?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? 
,__inference_dec_inner_1_layer_call_fn_169066O;</?,
%?"
 ?
inputs?????????
? "??????????(?
H__inference_dec_middle_0_layer_call_and_return_conditional_losses_169077\=>/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_0_layer_call_fn_169086O=>/?,
%?"
 ?
inputs?????????(
? "??????????<?
H__inference_dec_middle_1_layer_call_and_return_conditional_losses_169097\?@/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? ?
-__inference_dec_middle_1_layer_call_fn_169106O?@/?,
%?"
 ?
inputs?????????(
? "??????????<?
G__inference_dec_outer_0_layer_call_and_return_conditional_losses_169117\AB/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_0_layer_call_fn_169126OAB/?,
%?"
 ?
inputs?????????<
? "??????????<?
G__inference_dec_outer_1_layer_call_and_return_conditional_losses_169137\CD/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? 
,__inference_dec_outer_1_layer_call_fn_169146OCD/?,
%?"
 ?
inputs?????????<
? "??????????<?
F__inference_dec_output_layer_call_and_return_conditional_losses_169157]EF/?,
%?"
 ?
inputs?????????x
? "&?#
?
0??????????
? 
+__inference_dec_output_layer_call_fn_169166PEF/?,
%?"
 ?
inputs?????????x
? "????????????
G__inference_enc_inner_0_layer_call_and_return_conditional_losses_168957\12/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_0_layer_call_fn_168966O12/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_inner_1_layer_call_and_return_conditional_losses_168977\34/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? 
,__inference_enc_inner_1_layer_call_fn_168986O34/?,
%?"
 ?
inputs?????????2
? "??????????(?
H__inference_enc_middle_0_layer_call_and_return_conditional_losses_168917\-./?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_0_layer_call_fn_168926O-./?,
%?"
 ?
inputs?????????<
? "??????????2?
H__inference_enc_middle_1_layer_call_and_return_conditional_losses_168937\/0/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? ?
-__inference_enc_middle_1_layer_call_fn_168946O/0/?,
%?"
 ?
inputs?????????<
? "??????????2?
G__inference_enc_outer_0_layer_call_and_return_conditional_losses_168877])*0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_0_layer_call_fn_168886P)*0?-
&?#
!?
inputs??????????
? "??????????<?
G__inference_enc_outer_1_layer_call_and_return_conditional_losses_168897]+,0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? ?
,__inference_enc_outer_1_layer_call_fn_168906P+,0?-
&?#
!?
inputs??????????
? "??????????<?
C__inference_model_2_layer_call_and_return_conditional_losses_166907?+,)*/0-.34127856??<
5?2
(?%
encoder_input??????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_166952?+,)*/0-.34127856??<
5?2
(?%
encoder_input??????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_168547?+,)*/0-.341278568?5
.?+
!?
inputs??????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_168608?+,)*/0-.341278568?5
.?+
!?
inputs??????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
(__inference_model_2_layer_call_fn_167037?+,)*/0-.34127856??<
5?2
(?%
encoder_input??????????
p

 
? "=?:
?
0?????????
?
1??????????
(__inference_model_2_layer_call_fn_167121?+,)*/0-.34127856??<
5?2
(?%
encoder_input??????????
p 

 
? "=?:
?
0?????????
?
1??????????
(__inference_model_2_layer_call_fn_168647?+,)*/0-.341278568?5
.?+
!?
inputs??????????
p

 
? "=?:
?
0?????????
?
1??????????
(__inference_model_2_layer_call_fn_168686?+,)*/0-.341278568?5
.?+
!?
inputs??????????
p 

 
? "=?:
?
0?????????
?
1??????????
C__inference_model_3_layer_call_and_return_conditional_losses_167318?;<9:?@=>ABCDEFp?m
f?c
Y?V
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_3_layer_call_and_return_conditional_losses_167360?;<9:?@=>ABCDEFp?m
f?c
Y?V
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_3_layer_call_and_return_conditional_losses_168742?;<9:?@=>ABCDEFb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_3_layer_call_and_return_conditional_losses_168798?;<9:?@=>ABCDEFb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "&?#
?
0??????????
? ?
(__inference_model_3_layer_call_fn_167437?;<9:?@=>ABCDEFp?m
f?c
Y?V
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
p

 
? "????????????
(__inference_model_3_layer_call_fn_167513?;<9:?@=>ABCDEFp?m
f?c
Y?V
)?&
decoder_input_0?????????
)?&
decoder_input_1?????????
p 

 
? "????????????
(__inference_model_3_layer_call_fn_168832?;<9:?@=>ABCDEFb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "????????????
(__inference_model_3_layer_call_fn_168866?;<9:?@=>ABCDEFb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "????????????
$__inference_signature_wrapper_168134?+,)*/0-.34127856;<9:?@=>ABCDEF<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????