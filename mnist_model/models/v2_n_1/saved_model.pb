??
??
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

dec_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*"
shared_namedec_output/kernel
x
%dec_output/kernel/Read/ReadVariableOpReadVariableOpdec_output/kernel*
_output_shapes
:	<?*
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
Adam/dec_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*)
shared_nameAdam/dec_output/kernel/m
?
,Adam/dec_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_output/kernel/m*
_output_shapes
:	<?*
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
Adam/dec_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*)
shared_nameAdam/dec_output/kernel/v
?
,Adam/dec_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_output/kernel/v*
_output_shapes
:	<?*
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
?Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Q
value?QB?Q B?Q
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
?
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
	variables
trainable_variables
regularization_losses
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
 learning_rate!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
 
?
	variables
1metrics
2non_trainable_variables
3layer_regularization_losses
trainable_variables
4layer_metrics
regularization_losses

5layers
 
 
h

!kernel
"bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

#kernel
$bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
h

%kernel
&bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

'kernel
(bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
8
!0
"1
#2
$3
%4
&5
'6
(7
8
!0
"1
#2
$3
%4
&5
'6
(7
 
?
	variables
Fmetrics
Gnon_trainable_variables
Hlayer_regularization_losses
trainable_variables
Ilayer_metrics
regularization_losses

Jlayers
 
h

)kernel
*bias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
h

+kernel
,bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h

-kernel
.bias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api

W	keras_api
h

/kernel
0bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
8
)0
*1
+2
,3
-4
.5
/6
07
8
)0
*1
+2
,3
-4
.5
/6
07
 
?
	variables
\metrics
]non_trainable_variables
^layer_regularization_losses
trainable_variables
_layer_metrics
regularization_losses

`layers
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
OM
VARIABLE_VALUEenc_middle_0/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEenc_middle_0/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEenc_inner_0/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEenc_inner_0/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEchannel_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEchannel_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_inner_0/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdec_inner_0/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdec_middle_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_middle_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdec_outer_0/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdec_outer_0/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdec_output/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdec_output/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE

a0
 
 
 

0
1

!0
"1

!0
"1
 
?
6trainable_variables
bmetrics
cnon_trainable_variables
dlayer_regularization_losses
7	variables
elayer_metrics
8regularization_losses

flayers

#0
$1

#0
$1
 
?
:trainable_variables
gmetrics
hnon_trainable_variables
ilayer_regularization_losses
;	variables
jlayer_metrics
<regularization_losses

klayers

%0
&1

%0
&1
 
?
>trainable_variables
lmetrics
mnon_trainable_variables
nlayer_regularization_losses
?	variables
olayer_metrics
@regularization_losses

players

'0
(1

'0
(1
 
?
Btrainable_variables
qmetrics
rnon_trainable_variables
slayer_regularization_losses
C	variables
tlayer_metrics
Dregularization_losses

ulayers
 
 
 
 
#
	0

1
2
3
4

)0
*1

)0
*1
 
?
Ktrainable_variables
vmetrics
wnon_trainable_variables
xlayer_regularization_losses
L	variables
ylayer_metrics
Mregularization_losses

zlayers

+0
,1

+0
,1
 
?
Otrainable_variables
{metrics
|non_trainable_variables
}layer_regularization_losses
P	variables
~layer_metrics
Qregularization_losses

layers

-0
.1

-0
.1
 
?
Strainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
T	variables
?layer_metrics
Uregularization_losses
?layers
 
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
 
 
 
 
*
0
1
2
3
4
5
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
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_inner_0/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_inner_0/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/channel_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/channel_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_inner_0/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_inner_0/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_outer_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_outer_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/enc_middle_0/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/enc_middle_0/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/enc_inner_0/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/enc_inner_0/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/channel_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/channel_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_inner_0/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_inner_0/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dec_middle_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_middle_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/dec_outer_0/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dec_outer_0/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dec_output/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dec_output/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1enc_outer_0/kernelenc_outer_0/biasenc_middle_0/kernelenc_middle_0/biasenc_inner_0/kernelenc_inner_0/biaschannel_0/kernelchannel_0/biasdec_inner_0/kerneldec_inner_0/biasdec_middle_0/kerneldec_middle_0/biasdec_outer_0/kerneldec_outer_0/biasdec_output/kerneldec_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *,
f'R%
#__inference_signature_wrapper_96872
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp&enc_outer_0/kernel/Read/ReadVariableOp$enc_outer_0/bias/Read/ReadVariableOp'enc_middle_0/kernel/Read/ReadVariableOp%enc_middle_0/bias/Read/ReadVariableOp&enc_inner_0/kernel/Read/ReadVariableOp$enc_inner_0/bias/Read/ReadVariableOp$channel_0/kernel/Read/ReadVariableOp"channel_0/bias/Read/ReadVariableOp&dec_inner_0/kernel/Read/ReadVariableOp$dec_inner_0/bias/Read/ReadVariableOp'dec_middle_0/kernel/Read/ReadVariableOp%dec_middle_0/bias/Read/ReadVariableOp&dec_outer_0/kernel/Read/ReadVariableOp$dec_outer_0/bias/Read/ReadVariableOp%dec_output/kernel/Read/ReadVariableOp#dec_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/enc_outer_0/kernel/m/Read/ReadVariableOp+Adam/enc_outer_0/bias/m/Read/ReadVariableOp.Adam/enc_middle_0/kernel/m/Read/ReadVariableOp,Adam/enc_middle_0/bias/m/Read/ReadVariableOp-Adam/enc_inner_0/kernel/m/Read/ReadVariableOp+Adam/enc_inner_0/bias/m/Read/ReadVariableOp+Adam/channel_0/kernel/m/Read/ReadVariableOp)Adam/channel_0/bias/m/Read/ReadVariableOp-Adam/dec_inner_0/kernel/m/Read/ReadVariableOp+Adam/dec_inner_0/bias/m/Read/ReadVariableOp.Adam/dec_middle_0/kernel/m/Read/ReadVariableOp,Adam/dec_middle_0/bias/m/Read/ReadVariableOp-Adam/dec_outer_0/kernel/m/Read/ReadVariableOp+Adam/dec_outer_0/bias/m/Read/ReadVariableOp,Adam/dec_output/kernel/m/Read/ReadVariableOp*Adam/dec_output/bias/m/Read/ReadVariableOp-Adam/enc_outer_0/kernel/v/Read/ReadVariableOp+Adam/enc_outer_0/bias/v/Read/ReadVariableOp.Adam/enc_middle_0/kernel/v/Read/ReadVariableOp,Adam/enc_middle_0/bias/v/Read/ReadVariableOp-Adam/enc_inner_0/kernel/v/Read/ReadVariableOp+Adam/enc_inner_0/bias/v/Read/ReadVariableOp+Adam/channel_0/kernel/v/Read/ReadVariableOp)Adam/channel_0/bias/v/Read/ReadVariableOp-Adam/dec_inner_0/kernel/v/Read/ReadVariableOp+Adam/dec_inner_0/bias/v/Read/ReadVariableOp.Adam/dec_middle_0/kernel/v/Read/ReadVariableOp,Adam/dec_middle_0/bias/v/Read/ReadVariableOp-Adam/dec_outer_0/kernel/v/Read/ReadVariableOp+Adam/dec_outer_0/bias/v/Read/ReadVariableOp,Adam/dec_output/kernel/v/Read/ReadVariableOp*Adam/dec_output/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
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
GPU2*0,1J 8? *'
f"R 
__inference__traced_save_97630
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateenc_outer_0/kernelenc_outer_0/biasenc_middle_0/kernelenc_middle_0/biasenc_inner_0/kernelenc_inner_0/biaschannel_0/kernelchannel_0/biasdec_inner_0/kerneldec_inner_0/biasdec_middle_0/kerneldec_middle_0/biasdec_outer_0/kerneldec_outer_0/biasdec_output/kerneldec_output/biastotalcountAdam/enc_outer_0/kernel/mAdam/enc_outer_0/bias/mAdam/enc_middle_0/kernel/mAdam/enc_middle_0/bias/mAdam/enc_inner_0/kernel/mAdam/enc_inner_0/bias/mAdam/channel_0/kernel/mAdam/channel_0/bias/mAdam/dec_inner_0/kernel/mAdam/dec_inner_0/bias/mAdam/dec_middle_0/kernel/mAdam/dec_middle_0/bias/mAdam/dec_outer_0/kernel/mAdam/dec_outer_0/bias/mAdam/dec_output/kernel/mAdam/dec_output/bias/mAdam/enc_outer_0/kernel/vAdam/enc_outer_0/bias/vAdam/enc_middle_0/kernel/vAdam/enc_middle_0/bias/vAdam/enc_inner_0/kernel/vAdam/enc_inner_0/bias/vAdam/channel_0/kernel/vAdam/channel_0/bias/vAdam/dec_inner_0/kernel/vAdam/dec_inner_0/bias/vAdam/dec_middle_0/kernel/vAdam/dec_middle_0/bias/vAdam/dec_outer_0/kernel/vAdam/dec_outer_0/bias/vAdam/dec_output/kernel/vAdam/dec_output/bias/v*C
Tin<
:28*
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
GPU2*0,1J 8? **
f%R#
!__inference__traced_restore_97805??

?
?
+__inference_dec_outer_0_layer_call_fn_97422

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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_963502
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
,__inference_dec_middle_0_layer_call_fn_97402

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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_963232
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
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_96350

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
?
?
@__inference_model_layer_call_and_return_conditional_losses_96166
encoder_input
enc_outer_0_96079
enc_outer_0_96081
enc_middle_0_96106
enc_middle_0_96108
enc_inner_0_96133
enc_inner_0_96135
channel_0_96160
channel_0_96162
identity??!channel_0/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_96079enc_outer_0_96081*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_960682%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_96106enc_middle_0_96108*
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
GPU2*0,1J 8? *P
fKRI
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_960952&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_96133enc_inner_0_96135*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_961222%
#enc_inner_0/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_96160channel_0_96162*
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
GPU2*0,1J 8? *M
fHRF
D__inference_channel_0_layer_call_and_return_conditional_losses_961492#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?

?
+__inference_autoencoder_layer_call_fn_96825
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_967902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
+__inference_autoencoder_layer_call_fn_97068
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_967902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?k
?
__inference__traced_save_97630
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop1
-savev2_enc_outer_0_kernel_read_readvariableop/
+savev2_enc_outer_0_bias_read_readvariableop2
.savev2_enc_middle_0_kernel_read_readvariableop0
,savev2_enc_middle_0_bias_read_readvariableop1
-savev2_enc_inner_0_kernel_read_readvariableop/
+savev2_enc_inner_0_bias_read_readvariableop/
+savev2_channel_0_kernel_read_readvariableop-
)savev2_channel_0_bias_read_readvariableop1
-savev2_dec_inner_0_kernel_read_readvariableop/
+savev2_dec_inner_0_bias_read_readvariableop2
.savev2_dec_middle_0_kernel_read_readvariableop0
,savev2_dec_middle_0_bias_read_readvariableop1
-savev2_dec_outer_0_kernel_read_readvariableop/
+savev2_dec_outer_0_bias_read_readvariableop0
,savev2_dec_output_kernel_read_readvariableop.
*savev2_dec_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_m_read_readvariableop6
2savev2_adam_enc_outer_0_bias_m_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_m_read_readvariableop7
3savev2_adam_enc_middle_0_bias_m_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_m_read_readvariableop6
2savev2_adam_enc_inner_0_bias_m_read_readvariableop6
2savev2_adam_channel_0_kernel_m_read_readvariableop4
0savev2_adam_channel_0_bias_m_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_m_read_readvariableop6
2savev2_adam_dec_inner_0_bias_m_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_m_read_readvariableop7
3savev2_adam_dec_middle_0_bias_m_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_m_read_readvariableop6
2savev2_adam_dec_outer_0_bias_m_read_readvariableop7
3savev2_adam_dec_output_kernel_m_read_readvariableop5
1savev2_adam_dec_output_bias_m_read_readvariableop8
4savev2_adam_enc_outer_0_kernel_v_read_readvariableop6
2savev2_adam_enc_outer_0_bias_v_read_readvariableop9
5savev2_adam_enc_middle_0_kernel_v_read_readvariableop7
3savev2_adam_enc_middle_0_bias_v_read_readvariableop8
4savev2_adam_enc_inner_0_kernel_v_read_readvariableop6
2savev2_adam_enc_inner_0_bias_v_read_readvariableop6
2savev2_adam_channel_0_kernel_v_read_readvariableop4
0savev2_adam_channel_0_bias_v_read_readvariableop8
4savev2_adam_dec_inner_0_kernel_v_read_readvariableop6
2savev2_adam_dec_inner_0_bias_v_read_readvariableop9
5savev2_adam_dec_middle_0_kernel_v_read_readvariableop7
3savev2_adam_dec_middle_0_bias_v_read_readvariableop8
4savev2_adam_dec_outer_0_kernel_v_read_readvariableop6
2savev2_adam_dec_outer_0_bias_v_read_readvariableop7
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop-savev2_enc_outer_0_kernel_read_readvariableop+savev2_enc_outer_0_bias_read_readvariableop.savev2_enc_middle_0_kernel_read_readvariableop,savev2_enc_middle_0_bias_read_readvariableop-savev2_enc_inner_0_kernel_read_readvariableop+savev2_enc_inner_0_bias_read_readvariableop+savev2_channel_0_kernel_read_readvariableop)savev2_channel_0_bias_read_readvariableop-savev2_dec_inner_0_kernel_read_readvariableop+savev2_dec_inner_0_bias_read_readvariableop.savev2_dec_middle_0_kernel_read_readvariableop,savev2_dec_middle_0_bias_read_readvariableop-savev2_dec_outer_0_kernel_read_readvariableop+savev2_dec_outer_0_bias_read_readvariableop,savev2_dec_output_kernel_read_readvariableop*savev2_dec_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_enc_outer_0_kernel_m_read_readvariableop2savev2_adam_enc_outer_0_bias_m_read_readvariableop5savev2_adam_enc_middle_0_kernel_m_read_readvariableop3savev2_adam_enc_middle_0_bias_m_read_readvariableop4savev2_adam_enc_inner_0_kernel_m_read_readvariableop2savev2_adam_enc_inner_0_bias_m_read_readvariableop2savev2_adam_channel_0_kernel_m_read_readvariableop0savev2_adam_channel_0_bias_m_read_readvariableop4savev2_adam_dec_inner_0_kernel_m_read_readvariableop2savev2_adam_dec_inner_0_bias_m_read_readvariableop5savev2_adam_dec_middle_0_kernel_m_read_readvariableop3savev2_adam_dec_middle_0_bias_m_read_readvariableop4savev2_adam_dec_outer_0_kernel_m_read_readvariableop2savev2_adam_dec_outer_0_bias_m_read_readvariableop3savev2_adam_dec_output_kernel_m_read_readvariableop1savev2_adam_dec_output_bias_m_read_readvariableop4savev2_adam_enc_outer_0_kernel_v_read_readvariableop2savev2_adam_enc_outer_0_bias_v_read_readvariableop5savev2_adam_enc_middle_0_kernel_v_read_readvariableop3savev2_adam_enc_middle_0_bias_v_read_readvariableop4savev2_adam_enc_inner_0_kernel_v_read_readvariableop2savev2_adam_enc_inner_0_bias_v_read_readvariableop2savev2_adam_channel_0_kernel_v_read_readvariableop0savev2_adam_channel_0_bias_v_read_readvariableop4savev2_adam_dec_inner_0_kernel_v_read_readvariableop2savev2_adam_dec_inner_0_bias_v_read_readvariableop5savev2_adam_dec_middle_0_kernel_v_read_readvariableop3savev2_adam_dec_middle_0_bias_v_read_readvariableop4savev2_adam_dec_outer_0_kernel_v_read_readvariableop2savev2_adam_dec_outer_0_bias_v_read_readvariableop3savev2_adam_dec_output_kernel_v_read_readvariableop1savev2_adam_dec_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	?<:<:<2:2:2(:(:(::(:(:(<:<:<<:<:	<?:?: : :	?<:<:<2:2:2(:(:(::(:(:(<:<:<<:<:	<?:?:	?<:<:<2:2:2(:(:(::(:(:(<:<:<<:<:	<?:?: 2(
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
:<:$ 

_output_shapes

:<2: 	

_output_shapes
:2:$
 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:(<: 

_output_shapes
:<:$ 

_output_shapes

:<<: 

_output_shapes
:<:%!

_output_shapes
:	<?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?<: 

_output_shapes
:<:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

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

:(: !

_output_shapes
:(:$" 

_output_shapes

:(<: #

_output_shapes
:<:$$ 

_output_shapes

:<<: %

_output_shapes
:<:%&!

_output_shapes
:	<?:!'

_output_shapes	
:?:%(!

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

:2(: -

_output_shapes
:(:$. 

_output_shapes

:(: /

_output_shapes
::$0 

_output_shapes

:(: 1

_output_shapes
:(:$2 

_output_shapes

:(<: 3

_output_shapes
:<:$4 

_output_shapes

:<<: 5

_output_shapes
:<:%6!

_output_shapes
:	<?:!7

_output_shapes	
:?:8

_output_shapes
: 
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_96420
decoder_input_0
dec_inner_0_96398
dec_inner_0_96400
dec_middle_0_96403
dec_middle_0_96405
dec_outer_0_96408
dec_outer_0_96410
dec_output_96414
dec_output_96416
identity??#dec_inner_0/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_96398dec_inner_0_96400*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_962962%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_96403dec_middle_0_96405*
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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_963232&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_96408dec_outer_0_96410*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_963502%
#dec_outer_0/StatefulPartitionedCall?
tf.identity/IdentityIdentity,dec_outer_0/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????<2
tf.identity/Identity?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.identity/Identity:output:0dec_output_96414dec_output_96416*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dec_output_layer_call_and_return_conditional_losses_963782$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0
?	
?
E__inference_dec_output_layer_call_and_return_conditional_losses_96378

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<?*
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
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?]
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96933
x4
0model_enc_outer_0_matmul_readvariableop_resource5
1model_enc_outer_0_biasadd_readvariableop_resource5
1model_enc_middle_0_matmul_readvariableop_resource6
2model_enc_middle_0_biasadd_readvariableop_resource4
0model_enc_inner_0_matmul_readvariableop_resource5
1model_enc_inner_0_biasadd_readvariableop_resource2
.model_channel_0_matmul_readvariableop_resource3
/model_channel_0_biasadd_readvariableop_resource6
2model_1_dec_inner_0_matmul_readvariableop_resource7
3model_1_dec_inner_0_biasadd_readvariableop_resource7
3model_1_dec_middle_0_matmul_readvariableop_resource8
4model_1_dec_middle_0_biasadd_readvariableop_resource6
2model_1_dec_outer_0_matmul_readvariableop_resource7
3model_1_dec_outer_0_biasadd_readvariableop_resource5
1model_1_dec_output_matmul_readvariableop_resource6
2model_1_dec_output_biasadd_readvariableop_resource
identity??&model/channel_0/BiasAdd/ReadVariableOp?%model/channel_0/MatMul/ReadVariableOp?(model/enc_inner_0/BiasAdd/ReadVariableOp?'model/enc_inner_0/MatMul/ReadVariableOp?)model/enc_middle_0/BiasAdd/ReadVariableOp?(model/enc_middle_0/MatMul/ReadVariableOp?(model/enc_outer_0/BiasAdd/ReadVariableOp?'model/enc_outer_0/MatMul/ReadVariableOp?*model_1/dec_inner_0/BiasAdd/ReadVariableOp?)model_1/dec_inner_0/MatMul/ReadVariableOp?+model_1/dec_middle_0/BiasAdd/ReadVariableOp?*model_1/dec_middle_0/MatMul/ReadVariableOp?*model_1/dec_outer_0/BiasAdd/ReadVariableOp?)model_1/dec_outer_0/MatMul/ReadVariableOp?)model_1/dec_output/BiasAdd/ReadVariableOp?(model_1/dec_output/MatMul/ReadVariableOp?
'model/enc_outer_0/MatMul/ReadVariableOpReadVariableOp0model_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02)
'model/enc_outer_0/MatMul/ReadVariableOp?
model/enc_outer_0/MatMulMatMulx/model/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/enc_outer_0/MatMul?
(model/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp1model_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02*
(model/enc_outer_0/BiasAdd/ReadVariableOp?
model/enc_outer_0/BiasAddBiasAdd"model/enc_outer_0/MatMul:product:00model/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/enc_outer_0/BiasAdd?
model/enc_outer_0/ReluRelu"model/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model/enc_outer_0/Relu?
(model/enc_middle_0/MatMul/ReadVariableOpReadVariableOp1model_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02*
(model/enc_middle_0/MatMul/ReadVariableOp?
model/enc_middle_0/MatMulMatMul$model/enc_outer_0/Relu:activations:00model/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model/enc_middle_0/MatMul?
)model/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp2model_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)model/enc_middle_0/BiasAdd/ReadVariableOp?
model/enc_middle_0/BiasAddBiasAdd#model/enc_middle_0/MatMul:product:01model/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model/enc_middle_0/BiasAdd?
model/enc_middle_0/ReluRelu#model/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model/enc_middle_0/Relu?
'model/enc_inner_0/MatMul/ReadVariableOpReadVariableOp0model_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02)
'model/enc_inner_0/MatMul/ReadVariableOp?
model/enc_inner_0/MatMulMatMul%model/enc_middle_0/Relu:activations:0/model/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/enc_inner_0/MatMul?
(model/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp1model_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02*
(model/enc_inner_0/BiasAdd/ReadVariableOp?
model/enc_inner_0/BiasAddBiasAdd"model/enc_inner_0/MatMul:product:00model/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/enc_inner_0/BiasAdd?
model/enc_inner_0/ReluRelu"model/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model/enc_inner_0/Relu?
%model/channel_0/MatMul/ReadVariableOpReadVariableOp.model_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02'
%model/channel_0/MatMul/ReadVariableOp?
model/channel_0/MatMulMatMul$model/enc_inner_0/Relu:activations:0-model/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/channel_0/MatMul?
&model/channel_0/BiasAdd/ReadVariableOpReadVariableOp/model_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/channel_0/BiasAdd/ReadVariableOp?
model/channel_0/BiasAddBiasAdd model/channel_0/MatMul:product:0.model/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/channel_0/BiasAdd?
model/channel_0/SoftsignSoftsign model/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/channel_0/Softsign?
)model_1/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_1_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_1/dec_inner_0/MatMul/ReadVariableOp?
model_1/dec_inner_0/MatMulMatMul&model/channel_0/Softsign:activations:01model_1/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_1/dec_inner_0/MatMul?
*model_1/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_1_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_1/dec_inner_0/BiasAdd/ReadVariableOp?
model_1/dec_inner_0/BiasAddBiasAdd$model_1/dec_inner_0/MatMul:product:02model_1/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_1/dec_inner_0/BiasAdd?
model_1/dec_inner_0/ReluRelu$model_1/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_1/dec_inner_0/Relu?
*model_1/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_1_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_1/dec_middle_0/MatMul/ReadVariableOp?
model_1/dec_middle_0/MatMulMatMul&model_1/dec_inner_0/Relu:activations:02model_1/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_middle_0/MatMul?
+model_1/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_1_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_1/dec_middle_0/BiasAdd/ReadVariableOp?
model_1/dec_middle_0/BiasAddBiasAdd%model_1/dec_middle_0/MatMul:product:03model_1/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_middle_0/BiasAdd?
model_1/dec_middle_0/ReluRelu%model_1/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_middle_0/Relu?
)model_1/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_1_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_1/dec_outer_0/MatMul/ReadVariableOp?
model_1/dec_outer_0/MatMulMatMul'model_1/dec_middle_0/Relu:activations:01model_1/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_outer_0/MatMul?
*model_1/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_1_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_1/dec_outer_0/BiasAdd/ReadVariableOp?
model_1/dec_outer_0/BiasAddBiasAdd$model_1/dec_outer_0/MatMul:product:02model_1/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_outer_0/BiasAdd?
model_1/dec_outer_0/ReluRelu$model_1/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_outer_0/Relu?
model_1/tf.identity/IdentityIdentity&model_1/dec_outer_0/Relu:activations:0*
T0*'
_output_shapes
:?????????<2
model_1/tf.identity/Identity?
(model_1/dec_output/MatMul/ReadVariableOpReadVariableOp1model_1_dec_output_matmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02*
(model_1/dec_output/MatMul/ReadVariableOp?
model_1/dec_output/MatMulMatMul%model_1/tf.identity/Identity:output:00model_1/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dec_output/MatMul?
)model_1/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_1_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_1/dec_output/BiasAdd/ReadVariableOp?
model_1/dec_output/BiasAddBiasAdd#model_1/dec_output/MatMul:product:01model_1/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dec_output/BiasAdd?
model_1/dec_output/SigmoidSigmoid#model_1/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dec_output/Sigmoid?
IdentityIdentitymodel_1/dec_output/Sigmoid:y:0'^model/channel_0/BiasAdd/ReadVariableOp&^model/channel_0/MatMul/ReadVariableOp)^model/enc_inner_0/BiasAdd/ReadVariableOp(^model/enc_inner_0/MatMul/ReadVariableOp*^model/enc_middle_0/BiasAdd/ReadVariableOp)^model/enc_middle_0/MatMul/ReadVariableOp)^model/enc_outer_0/BiasAdd/ReadVariableOp(^model/enc_outer_0/MatMul/ReadVariableOp+^model_1/dec_inner_0/BiasAdd/ReadVariableOp*^model_1/dec_inner_0/MatMul/ReadVariableOp,^model_1/dec_middle_0/BiasAdd/ReadVariableOp+^model_1/dec_middle_0/MatMul/ReadVariableOp+^model_1/dec_outer_0/BiasAdd/ReadVariableOp*^model_1/dec_outer_0/MatMul/ReadVariableOp*^model_1/dec_output/BiasAdd/ReadVariableOp)^model_1/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2P
&model/channel_0/BiasAdd/ReadVariableOp&model/channel_0/BiasAdd/ReadVariableOp2N
%model/channel_0/MatMul/ReadVariableOp%model/channel_0/MatMul/ReadVariableOp2T
(model/enc_inner_0/BiasAdd/ReadVariableOp(model/enc_inner_0/BiasAdd/ReadVariableOp2R
'model/enc_inner_0/MatMul/ReadVariableOp'model/enc_inner_0/MatMul/ReadVariableOp2V
)model/enc_middle_0/BiasAdd/ReadVariableOp)model/enc_middle_0/BiasAdd/ReadVariableOp2T
(model/enc_middle_0/MatMul/ReadVariableOp(model/enc_middle_0/MatMul/ReadVariableOp2T
(model/enc_outer_0/BiasAdd/ReadVariableOp(model/enc_outer_0/BiasAdd/ReadVariableOp2R
'model/enc_outer_0/MatMul/ReadVariableOp'model/enc_outer_0/MatMul/ReadVariableOp2X
*model_1/dec_inner_0/BiasAdd/ReadVariableOp*model_1/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_1/dec_inner_0/MatMul/ReadVariableOp)model_1/dec_inner_0/MatMul/ReadVariableOp2Z
+model_1/dec_middle_0/BiasAdd/ReadVariableOp+model_1/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_1/dec_middle_0/MatMul/ReadVariableOp*model_1/dec_middle_0/MatMul/ReadVariableOp2X
*model_1/dec_outer_0/BiasAdd/ReadVariableOp*model_1/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_1/dec_outer_0/MatMul/ReadVariableOp)model_1/dec_outer_0/MatMul/ReadVariableOp2V
)model_1/dec_output/BiasAdd/ReadVariableOp)model_1/dec_output/BiasAdd/ReadVariableOp2T
(model_1/dec_output/MatMul/ReadVariableOp(model_1/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
%__inference_model_layer_call_fn_96236
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?	
?
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_97333

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
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_96395
decoder_input_0
dec_inner_0_96307
dec_inner_0_96309
dec_middle_0_96334
dec_middle_0_96336
dec_outer_0_96361
dec_outer_0_96363
dec_output_96389
dec_output_96391
identity??#dec_inner_0/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0dec_inner_0_96307dec_inner_0_96309*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_962962%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_96334dec_middle_0_96336*
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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_963232&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_96361dec_outer_0_96363*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_963502%
#dec_outer_0/StatefulPartitionedCall?
tf.identity/IdentityIdentity,dec_outer_0/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????<2
tf.identity/Identity?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.identity/Identity:output:0dec_output_96389dec_output_96391*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dec_output_layer_call_and_return_conditional_losses_963782$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_96448

inputs
dec_inner_0_96426
dec_inner_0_96428
dec_middle_0_96431
dec_middle_0_96433
dec_outer_0_96436
dec_outer_0_96438
dec_output_96442
dec_output_96444
identity??#dec_inner_0/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_96426dec_inner_0_96428*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_962962%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_96431dec_middle_0_96433*
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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_963232&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_96436dec_outer_0_96438*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_963502%
#dec_outer_0/StatefulPartitionedCall?
tf.identity/IdentityIdentity,dec_outer_0/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????<2
tf.identity/Identity?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.identity/Identity:output:0dec_output_96442dec_output_96444*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dec_output_layer_call_and_return_conditional_losses_963782$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_97805
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate)
%assignvariableop_5_enc_outer_0_kernel'
#assignvariableop_6_enc_outer_0_bias*
&assignvariableop_7_enc_middle_0_kernel(
$assignvariableop_8_enc_middle_0_bias)
%assignvariableop_9_enc_inner_0_kernel(
$assignvariableop_10_enc_inner_0_bias(
$assignvariableop_11_channel_0_kernel&
"assignvariableop_12_channel_0_bias*
&assignvariableop_13_dec_inner_0_kernel(
$assignvariableop_14_dec_inner_0_bias+
'assignvariableop_15_dec_middle_0_kernel)
%assignvariableop_16_dec_middle_0_bias*
&assignvariableop_17_dec_outer_0_kernel(
$assignvariableop_18_dec_outer_0_bias)
%assignvariableop_19_dec_output_kernel'
#assignvariableop_20_dec_output_bias
assignvariableop_21_total
assignvariableop_22_count1
-assignvariableop_23_adam_enc_outer_0_kernel_m/
+assignvariableop_24_adam_enc_outer_0_bias_m2
.assignvariableop_25_adam_enc_middle_0_kernel_m0
,assignvariableop_26_adam_enc_middle_0_bias_m1
-assignvariableop_27_adam_enc_inner_0_kernel_m/
+assignvariableop_28_adam_enc_inner_0_bias_m/
+assignvariableop_29_adam_channel_0_kernel_m-
)assignvariableop_30_adam_channel_0_bias_m1
-assignvariableop_31_adam_dec_inner_0_kernel_m/
+assignvariableop_32_adam_dec_inner_0_bias_m2
.assignvariableop_33_adam_dec_middle_0_kernel_m0
,assignvariableop_34_adam_dec_middle_0_bias_m1
-assignvariableop_35_adam_dec_outer_0_kernel_m/
+assignvariableop_36_adam_dec_outer_0_bias_m0
,assignvariableop_37_adam_dec_output_kernel_m.
*assignvariableop_38_adam_dec_output_bias_m1
-assignvariableop_39_adam_enc_outer_0_kernel_v/
+assignvariableop_40_adam_enc_outer_0_bias_v2
.assignvariableop_41_adam_enc_middle_0_kernel_v0
,assignvariableop_42_adam_enc_middle_0_bias_v1
-assignvariableop_43_adam_enc_inner_0_kernel_v/
+assignvariableop_44_adam_enc_inner_0_bias_v/
+assignvariableop_45_adam_channel_0_kernel_v-
)assignvariableop_46_adam_channel_0_bias_v1
-assignvariableop_47_adam_dec_inner_0_kernel_v/
+assignvariableop_48_adam_dec_inner_0_bias_v2
.assignvariableop_49_adam_dec_middle_0_kernel_v0
,assignvariableop_50_adam_dec_middle_0_bias_v1
-assignvariableop_51_adam_dec_outer_0_kernel_v/
+assignvariableop_52_adam_dec_outer_0_bias_v0
,assignvariableop_53_adam_dec_output_kernel_v.
*assignvariableop_54_adam_dec_output_bias_v
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
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
AssignVariableOp_7AssignVariableOp&assignvariableop_7_enc_middle_0_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_enc_middle_0_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_enc_inner_0_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_enc_inner_0_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_channel_0_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_channel_0_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp&assignvariableop_13_dec_inner_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dec_inner_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_dec_middle_0_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_dec_middle_0_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_dec_outer_0_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dec_outer_0_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_dec_output_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dec_output_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_enc_outer_0_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_enc_outer_0_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_enc_middle_0_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_enc_middle_0_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_enc_inner_0_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_enc_inner_0_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_channel_0_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_channel_0_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dec_inner_0_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dec_inner_0_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_dec_middle_0_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_dec_middle_0_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_dec_outer_0_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_dec_outer_0_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_dec_output_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dec_output_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_enc_outer_0_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_enc_outer_0_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_enc_middle_0_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_enc_middle_0_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp-assignvariableop_43_adam_enc_inner_0_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_enc_inner_0_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_channel_0_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_channel_0_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_dec_inner_0_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dec_inner_0_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp.assignvariableop_49_adam_dec_middle_0_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp,assignvariableop_50_adam_dec_middle_0_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_dec_outer_0_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_dec_outer_0_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_dec_output_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_dec_output_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55?

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
%__inference_model_layer_call_fn_96281
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?	
?
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_96323

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
~
)__inference_channel_0_layer_call_fn_97362

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
GPU2*0,1J 8? *M
fHRF
D__inference_channel_0_layer_call_and_return_conditional_losses_961492
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
+__inference_enc_inner_0_layer_call_fn_97342

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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_961222
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
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_97413

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
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_97373

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
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96715
x
model_96680
model_96682
model_96684
model_96686
model_96688
model_96690
model_96692
model_96694
model_1_96697
model_1_96699
model_1_96701
model_1_96703
model_1_96705
model_1_96707
model_1_96709
model_1_96711
identity??model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallxmodel_96680model_96682model_96684model_96686model_96688model_96690model_96692model_96694*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962172
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_96697model_1_96699model_1_96701model_1_96703model_1_96705model_1_96707model_1_96709model_1_96711*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964482!
model_1/StatefulPartitionedCall?
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?

?
+__inference_autoencoder_layer_call_fn_97031
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_967152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?]
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96994
x4
0model_enc_outer_0_matmul_readvariableop_resource5
1model_enc_outer_0_biasadd_readvariableop_resource5
1model_enc_middle_0_matmul_readvariableop_resource6
2model_enc_middle_0_biasadd_readvariableop_resource4
0model_enc_inner_0_matmul_readvariableop_resource5
1model_enc_inner_0_biasadd_readvariableop_resource2
.model_channel_0_matmul_readvariableop_resource3
/model_channel_0_biasadd_readvariableop_resource6
2model_1_dec_inner_0_matmul_readvariableop_resource7
3model_1_dec_inner_0_biasadd_readvariableop_resource7
3model_1_dec_middle_0_matmul_readvariableop_resource8
4model_1_dec_middle_0_biasadd_readvariableop_resource6
2model_1_dec_outer_0_matmul_readvariableop_resource7
3model_1_dec_outer_0_biasadd_readvariableop_resource5
1model_1_dec_output_matmul_readvariableop_resource6
2model_1_dec_output_biasadd_readvariableop_resource
identity??&model/channel_0/BiasAdd/ReadVariableOp?%model/channel_0/MatMul/ReadVariableOp?(model/enc_inner_0/BiasAdd/ReadVariableOp?'model/enc_inner_0/MatMul/ReadVariableOp?)model/enc_middle_0/BiasAdd/ReadVariableOp?(model/enc_middle_0/MatMul/ReadVariableOp?(model/enc_outer_0/BiasAdd/ReadVariableOp?'model/enc_outer_0/MatMul/ReadVariableOp?*model_1/dec_inner_0/BiasAdd/ReadVariableOp?)model_1/dec_inner_0/MatMul/ReadVariableOp?+model_1/dec_middle_0/BiasAdd/ReadVariableOp?*model_1/dec_middle_0/MatMul/ReadVariableOp?*model_1/dec_outer_0/BiasAdd/ReadVariableOp?)model_1/dec_outer_0/MatMul/ReadVariableOp?)model_1/dec_output/BiasAdd/ReadVariableOp?(model_1/dec_output/MatMul/ReadVariableOp?
'model/enc_outer_0/MatMul/ReadVariableOpReadVariableOp0model_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype02)
'model/enc_outer_0/MatMul/ReadVariableOp?
model/enc_outer_0/MatMulMatMulx/model/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/enc_outer_0/MatMul?
(model/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp1model_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02*
(model/enc_outer_0/BiasAdd/ReadVariableOp?
model/enc_outer_0/BiasAddBiasAdd"model/enc_outer_0/MatMul:product:00model/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/enc_outer_0/BiasAdd?
model/enc_outer_0/ReluRelu"model/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model/enc_outer_0/Relu?
(model/enc_middle_0/MatMul/ReadVariableOpReadVariableOp1model_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype02*
(model/enc_middle_0/MatMul/ReadVariableOp?
model/enc_middle_0/MatMulMatMul$model/enc_outer_0/Relu:activations:00model/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model/enc_middle_0/MatMul?
)model/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp2model_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)model/enc_middle_0/BiasAdd/ReadVariableOp?
model/enc_middle_0/BiasAddBiasAdd#model/enc_middle_0/MatMul:product:01model/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model/enc_middle_0/BiasAdd?
model/enc_middle_0/ReluRelu#model/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model/enc_middle_0/Relu?
'model/enc_inner_0/MatMul/ReadVariableOpReadVariableOp0model_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02)
'model/enc_inner_0/MatMul/ReadVariableOp?
model/enc_inner_0/MatMulMatMul%model/enc_middle_0/Relu:activations:0/model/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/enc_inner_0/MatMul?
(model/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp1model_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02*
(model/enc_inner_0/BiasAdd/ReadVariableOp?
model/enc_inner_0/BiasAddBiasAdd"model/enc_inner_0/MatMul:product:00model/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/enc_inner_0/BiasAdd?
model/enc_inner_0/ReluRelu"model/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model/enc_inner_0/Relu?
%model/channel_0/MatMul/ReadVariableOpReadVariableOp.model_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02'
%model/channel_0/MatMul/ReadVariableOp?
model/channel_0/MatMulMatMul$model/enc_inner_0/Relu:activations:0-model/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/channel_0/MatMul?
&model/channel_0/BiasAdd/ReadVariableOpReadVariableOp/model_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/channel_0/BiasAdd/ReadVariableOp?
model/channel_0/BiasAddBiasAdd model/channel_0/MatMul:product:0.model/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/channel_0/BiasAdd?
model/channel_0/SoftsignSoftsign model/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/channel_0/Softsign?
)model_1/dec_inner_0/MatMul/ReadVariableOpReadVariableOp2model_1_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02+
)model_1/dec_inner_0/MatMul/ReadVariableOp?
model_1/dec_inner_0/MatMulMatMul&model/channel_0/Softsign:activations:01model_1/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_1/dec_inner_0/MatMul?
*model_1/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp3model_1_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*model_1/dec_inner_0/BiasAdd/ReadVariableOp?
model_1/dec_inner_0/BiasAddBiasAdd$model_1/dec_inner_0/MatMul:product:02model_1/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model_1/dec_inner_0/BiasAdd?
model_1/dec_inner_0/ReluRelu$model_1/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model_1/dec_inner_0/Relu?
*model_1/dec_middle_0/MatMul/ReadVariableOpReadVariableOp3model_1_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02,
*model_1/dec_middle_0/MatMul/ReadVariableOp?
model_1/dec_middle_0/MatMulMatMul&model_1/dec_inner_0/Relu:activations:02model_1/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_middle_0/MatMul?
+model_1/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp4model_1_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02-
+model_1/dec_middle_0/BiasAdd/ReadVariableOp?
model_1/dec_middle_0/BiasAddBiasAdd%model_1/dec_middle_0/MatMul:product:03model_1/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_middle_0/BiasAdd?
model_1/dec_middle_0/ReluRelu%model_1/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_middle_0/Relu?
)model_1/dec_outer_0/MatMul/ReadVariableOpReadVariableOp2model_1_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02+
)model_1/dec_outer_0/MatMul/ReadVariableOp?
model_1/dec_outer_0/MatMulMatMul'model_1/dec_middle_0/Relu:activations:01model_1/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_outer_0/MatMul?
*model_1/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp3model_1_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02,
*model_1/dec_outer_0/BiasAdd/ReadVariableOp?
model_1/dec_outer_0/BiasAddBiasAdd$model_1/dec_outer_0/MatMul:product:02model_1/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_outer_0/BiasAdd?
model_1/dec_outer_0/ReluRelu$model_1/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model_1/dec_outer_0/Relu?
model_1/tf.identity/IdentityIdentity&model_1/dec_outer_0/Relu:activations:0*
T0*'
_output_shapes
:?????????<2
model_1/tf.identity/Identity?
(model_1/dec_output/MatMul/ReadVariableOpReadVariableOp1model_1_dec_output_matmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02*
(model_1/dec_output/MatMul/ReadVariableOp?
model_1/dec_output/MatMulMatMul%model_1/tf.identity/Identity:output:00model_1/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dec_output/MatMul?
)model_1/dec_output/BiasAdd/ReadVariableOpReadVariableOp2model_1_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_1/dec_output/BiasAdd/ReadVariableOp?
model_1/dec_output/BiasAddBiasAdd#model_1/dec_output/MatMul:product:01model_1/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dec_output/BiasAdd?
model_1/dec_output/SigmoidSigmoid#model_1/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dec_output/Sigmoid?
IdentityIdentitymodel_1/dec_output/Sigmoid:y:0'^model/channel_0/BiasAdd/ReadVariableOp&^model/channel_0/MatMul/ReadVariableOp)^model/enc_inner_0/BiasAdd/ReadVariableOp(^model/enc_inner_0/MatMul/ReadVariableOp*^model/enc_middle_0/BiasAdd/ReadVariableOp)^model/enc_middle_0/MatMul/ReadVariableOp)^model/enc_outer_0/BiasAdd/ReadVariableOp(^model/enc_outer_0/MatMul/ReadVariableOp+^model_1/dec_inner_0/BiasAdd/ReadVariableOp*^model_1/dec_inner_0/MatMul/ReadVariableOp,^model_1/dec_middle_0/BiasAdd/ReadVariableOp+^model_1/dec_middle_0/MatMul/ReadVariableOp+^model_1/dec_outer_0/BiasAdd/ReadVariableOp*^model_1/dec_outer_0/MatMul/ReadVariableOp*^model_1/dec_output/BiasAdd/ReadVariableOp)^model_1/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2P
&model/channel_0/BiasAdd/ReadVariableOp&model/channel_0/BiasAdd/ReadVariableOp2N
%model/channel_0/MatMul/ReadVariableOp%model/channel_0/MatMul/ReadVariableOp2T
(model/enc_inner_0/BiasAdd/ReadVariableOp(model/enc_inner_0/BiasAdd/ReadVariableOp2R
'model/enc_inner_0/MatMul/ReadVariableOp'model/enc_inner_0/MatMul/ReadVariableOp2V
)model/enc_middle_0/BiasAdd/ReadVariableOp)model/enc_middle_0/BiasAdd/ReadVariableOp2T
(model/enc_middle_0/MatMul/ReadVariableOp(model/enc_middle_0/MatMul/ReadVariableOp2T
(model/enc_outer_0/BiasAdd/ReadVariableOp(model/enc_outer_0/BiasAdd/ReadVariableOp2R
'model/enc_outer_0/MatMul/ReadVariableOp'model/enc_outer_0/MatMul/ReadVariableOp2X
*model_1/dec_inner_0/BiasAdd/ReadVariableOp*model_1/dec_inner_0/BiasAdd/ReadVariableOp2V
)model_1/dec_inner_0/MatMul/ReadVariableOp)model_1/dec_inner_0/MatMul/ReadVariableOp2Z
+model_1/dec_middle_0/BiasAdd/ReadVariableOp+model_1/dec_middle_0/BiasAdd/ReadVariableOp2X
*model_1/dec_middle_0/MatMul/ReadVariableOp*model_1/dec_middle_0/MatMul/ReadVariableOp2X
*model_1/dec_outer_0/BiasAdd/ReadVariableOp*model_1/dec_outer_0/BiasAdd/ReadVariableOp2V
)model_1/dec_outer_0/MatMul/ReadVariableOp)model_1/dec_outer_0/MatMul/ReadVariableOp2V
)model_1/dec_output/BiasAdd/ReadVariableOp)model_1/dec_output/BiasAdd/ReadVariableOp2T
(model_1/dec_output/MatMul/ReadVariableOp(model_1/dec_output/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
@__inference_model_layer_call_and_return_conditional_losses_96190
encoder_input
enc_outer_0_96169
enc_outer_0_96171
enc_middle_0_96174
enc_middle_0_96176
enc_inner_0_96179
enc_inner_0_96181
channel_0_96184
channel_0_96186
identity??!channel_0/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_outer_0_96169enc_outer_0_96171*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_960682%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_96174enc_middle_0_96176*
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
GPU2*0,1J 8? *P
fKRI
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_960952&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_96179enc_inner_0_96181*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_961222%
#enc_inner_0/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_96184channel_0_96186*
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
GPU2*0,1J 8? *M
fHRF
D__inference_channel_0_layer_call_and_return_conditional_losses_961492#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoder_input
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96674
input_1
model_96639
model_96641
model_96643
model_96645
model_96647
model_96649
model_96651
model_96653
model_1_96656
model_1_96658
model_1_96660
model_1_96662
model_1_96664
model_1_96666
model_1_96668
model_1_96670
identity??model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_96639model_96641model_96643model_96645model_96647model_96649model_96651model_96653*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962622
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_96656model_1_96658model_1_96660model_1_96662model_1_96664model_1_96666model_1_96668model_1_96670*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964942!
model_1/StatefulPartitionedCall?
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_97293

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
?
?
@__inference_model_layer_call_and_return_conditional_losses_96217

inputs
enc_outer_0_96196
enc_outer_0_96198
enc_middle_0_96201
enc_middle_0_96203
enc_inner_0_96206
enc_inner_0_96208
channel_0_96211
channel_0_96213
identity??!channel_0/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_96196enc_outer_0_96198*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_960682%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_96201enc_middle_0_96203*
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
GPU2*0,1J 8? *P
fKRI
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_960952&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_96206enc_inner_0_96208*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_961222%
#enc_inner_0/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_96211channel_0_96213*
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
GPU2*0,1J 8? *M
fHRF
D__inference_channel_0_layer_call_and_return_conditional_losses_961492#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
B__inference_model_1_layer_call_and_return_conditional_losses_97207

inputs.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_0_matmul_readvariableop_resource0
,dec_middle_0_biasadd_readvariableop_resource.
*dec_outer_0_matmul_readvariableop_resource/
+dec_outer_0_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
!dec_inner_0/MatMul/ReadVariableOpReadVariableOp*dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_0/MatMul/ReadVariableOp?
dec_inner_0/MatMulMatMulinputs)dec_inner_0/MatMul/ReadVariableOp:value:0*
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
tf.identity/IdentityIdentitydec_outer_0/Relu:activations:0*
T0*'
_output_shapes
:?????????<2
tf.identity/Identity?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.identity/Identity:output:0(dec_output/MatMul/ReadVariableOp:value:0*
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
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2F
!dec_output/BiasAdd/ReadVariableOp!dec_output/BiasAdd/ReadVariableOp2D
 dec_output/MatMul/ReadVariableOp dec_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
B__inference_model_1_layer_call_and_return_conditional_losses_97240

inputs.
*dec_inner_0_matmul_readvariableop_resource/
+dec_inner_0_biasadd_readvariableop_resource/
+dec_middle_0_matmul_readvariableop_resource0
,dec_middle_0_biasadd_readvariableop_resource.
*dec_outer_0_matmul_readvariableop_resource/
+dec_outer_0_biasadd_readvariableop_resource-
)dec_output_matmul_readvariableop_resource.
*dec_output_biasadd_readvariableop_resource
identity??"dec_inner_0/BiasAdd/ReadVariableOp?!dec_inner_0/MatMul/ReadVariableOp?#dec_middle_0/BiasAdd/ReadVariableOp?"dec_middle_0/MatMul/ReadVariableOp?"dec_outer_0/BiasAdd/ReadVariableOp?!dec_outer_0/MatMul/ReadVariableOp?!dec_output/BiasAdd/ReadVariableOp? dec_output/MatMul/ReadVariableOp?
!dec_inner_0/MatMul/ReadVariableOpReadVariableOp*dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02#
!dec_inner_0/MatMul/ReadVariableOp?
dec_inner_0/MatMulMatMulinputs)dec_inner_0/MatMul/ReadVariableOp:value:0*
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
tf.identity/IdentityIdentitydec_outer_0/Relu:activations:0*
T0*'
_output_shapes
:?????????<2
tf.identity/Identity?
 dec_output/MatMul/ReadVariableOpReadVariableOp)dec_output_matmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02"
 dec_output/MatMul/ReadVariableOp?
dec_output/MatMulMatMultf.identity/Identity:output:0(dec_output/MatMul/ReadVariableOp:value:0*
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
dec_output/Sigmoid?
IdentityIdentitydec_output/Sigmoid:y:0#^dec_inner_0/BiasAdd/ReadVariableOp"^dec_inner_0/MatMul/ReadVariableOp$^dec_middle_0/BiasAdd/ReadVariableOp#^dec_middle_0/MatMul/ReadVariableOp#^dec_outer_0/BiasAdd/ReadVariableOp"^dec_outer_0/MatMul/ReadVariableOp"^dec_output/BiasAdd/ReadVariableOp!^dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2H
"dec_inner_0/BiasAdd/ReadVariableOp"dec_inner_0/BiasAdd/ReadVariableOp2F
!dec_inner_0/MatMul/ReadVariableOp!dec_inner_0/MatMul/ReadVariableOp2J
#dec_middle_0/BiasAdd/ReadVariableOp#dec_middle_0/BiasAdd/ReadVariableOp2H
"dec_middle_0/MatMul/ReadVariableOp"dec_middle_0/MatMul/ReadVariableOp2H
"dec_outer_0/BiasAdd/ReadVariableOp"dec_outer_0/BiasAdd/ReadVariableOp2F
!dec_outer_0/MatMul/ReadVariableOp!dec_outer_0/MatMul/ReadVariableOp2F
!dec_output/BiasAdd/ReadVariableOp!dec_output/BiasAdd/ReadVariableOp2D
 dec_output/MatMul/ReadVariableOp dec_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_channel_0_layer_call_and_return_conditional_losses_96149

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
?
?
'__inference_model_1_layer_call_fn_96513
decoder_input_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0
?)
?
@__inference_model_layer_call_and_return_conditional_losses_97100

inputs.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?
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
channel_0/Softsign?
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_96296

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
+__inference_enc_outer_0_layer_call_fn_97302

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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_960682
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
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_96494

inputs
dec_inner_0_96472
dec_inner_0_96474
dec_middle_0_96477
dec_middle_0_96479
dec_outer_0_96482
dec_outer_0_96484
dec_output_96488
dec_output_96490
identity??#dec_inner_0/StatefulPartitionedCall?$dec_middle_0/StatefulPartitionedCall?#dec_outer_0/StatefulPartitionedCall?"dec_output/StatefulPartitionedCall?
#dec_inner_0/StatefulPartitionedCallStatefulPartitionedCallinputsdec_inner_0_96472dec_inner_0_96474*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_962962%
#dec_inner_0/StatefulPartitionedCall?
$dec_middle_0/StatefulPartitionedCallStatefulPartitionedCall,dec_inner_0/StatefulPartitionedCall:output:0dec_middle_0_96477dec_middle_0_96479*
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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_963232&
$dec_middle_0/StatefulPartitionedCall?
#dec_outer_0/StatefulPartitionedCallStatefulPartitionedCall-dec_middle_0/StatefulPartitionedCall:output:0dec_outer_0_96482dec_outer_0_96484*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_963502%
#dec_outer_0/StatefulPartitionedCall?
tf.identity/IdentityIdentity,dec_outer_0/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????<2
tf.identity/Identity?
"dec_output/StatefulPartitionedCallStatefulPartitionedCalltf.identity/Identity:output:0dec_output_96488dec_output_96490*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dec_output_layer_call_and_return_conditional_losses_963782$
"dec_output/StatefulPartitionedCall?
IdentityIdentity+dec_output/StatefulPartitionedCall:output:0$^dec_inner_0/StatefulPartitionedCall%^dec_middle_0/StatefulPartitionedCall$^dec_outer_0/StatefulPartitionedCall#^dec_output/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#dec_inner_0/StatefulPartitionedCall#dec_inner_0/StatefulPartitionedCall2L
$dec_middle_0/StatefulPartitionedCall$dec_middle_0/StatefulPartitionedCall2J
#dec_outer_0/StatefulPartitionedCall#dec_outer_0/StatefulPartitionedCall2H
"dec_output/StatefulPartitionedCall"dec_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dec_output_layer_call_and_return_conditional_losses_97433

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<?*
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
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?q
?
 __inference__wrapped_model_96053
input_1@
<autoencoder_model_enc_outer_0_matmul_readvariableop_resourceA
=autoencoder_model_enc_outer_0_biasadd_readvariableop_resourceA
=autoencoder_model_enc_middle_0_matmul_readvariableop_resourceB
>autoencoder_model_enc_middle_0_biasadd_readvariableop_resource@
<autoencoder_model_enc_inner_0_matmul_readvariableop_resourceA
=autoencoder_model_enc_inner_0_biasadd_readvariableop_resource>
:autoencoder_model_channel_0_matmul_readvariableop_resource?
;autoencoder_model_channel_0_biasadd_readvariableop_resourceB
>autoencoder_model_1_dec_inner_0_matmul_readvariableop_resourceC
?autoencoder_model_1_dec_inner_0_biasadd_readvariableop_resourceC
?autoencoder_model_1_dec_middle_0_matmul_readvariableop_resourceD
@autoencoder_model_1_dec_middle_0_biasadd_readvariableop_resourceB
>autoencoder_model_1_dec_outer_0_matmul_readvariableop_resourceC
?autoencoder_model_1_dec_outer_0_biasadd_readvariableop_resourceA
=autoencoder_model_1_dec_output_matmul_readvariableop_resourceB
>autoencoder_model_1_dec_output_biasadd_readvariableop_resource
identity??2autoencoder/model/channel_0/BiasAdd/ReadVariableOp?1autoencoder/model/channel_0/MatMul/ReadVariableOp?4autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOp?3autoencoder/model/enc_inner_0/MatMul/ReadVariableOp?5autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOp?4autoencoder/model/enc_middle_0/MatMul/ReadVariableOp?4autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOp?3autoencoder/model/enc_outer_0/MatMul/ReadVariableOp?6autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOp?5autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOp?7autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOp?6autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOp?6autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOp?5autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOp?5autoencoder/model_1/dec_output/BiasAdd/ReadVariableOp?4autoencoder/model_1/dec_output/MatMul/ReadVariableOp?
3autoencoder/model/enc_outer_0/MatMul/ReadVariableOpReadVariableOp<autoencoder_model_enc_outer_0_matmul_readvariableop_resource*
_output_shapes
:	?<*
dtype025
3autoencoder/model/enc_outer_0/MatMul/ReadVariableOp?
$autoencoder/model/enc_outer_0/MatMulMatMulinput_1;autoencoder/model/enc_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2&
$autoencoder/model/enc_outer_0/MatMul?
4autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_enc_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype026
4autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOp?
%autoencoder/model/enc_outer_0/BiasAddBiasAdd.autoencoder/model/enc_outer_0/MatMul:product:0<autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2'
%autoencoder/model/enc_outer_0/BiasAdd?
"autoencoder/model/enc_outer_0/ReluRelu.autoencoder/model/enc_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2$
"autoencoder/model/enc_outer_0/Relu?
4autoencoder/model/enc_middle_0/MatMul/ReadVariableOpReadVariableOp=autoencoder_model_enc_middle_0_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype026
4autoencoder/model/enc_middle_0/MatMul/ReadVariableOp?
%autoencoder/model/enc_middle_0/MatMulMatMul0autoencoder/model/enc_outer_0/Relu:activations:0<autoencoder/model/enc_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22'
%autoencoder/model/enc_middle_0/MatMul?
5autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOpReadVariableOp>autoencoder_model_enc_middle_0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOp?
&autoencoder/model/enc_middle_0/BiasAddBiasAdd/autoencoder/model/enc_middle_0/MatMul:product:0=autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&autoencoder/model/enc_middle_0/BiasAdd?
#autoencoder/model/enc_middle_0/ReluRelu/autoencoder/model/enc_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22%
#autoencoder/model/enc_middle_0/Relu?
3autoencoder/model/enc_inner_0/MatMul/ReadVariableOpReadVariableOp<autoencoder_model_enc_inner_0_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype025
3autoencoder/model/enc_inner_0/MatMul/ReadVariableOp?
$autoencoder/model/enc_inner_0/MatMulMatMul1autoencoder/model/enc_middle_0/Relu:activations:0;autoencoder/model/enc_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2&
$autoencoder/model/enc_inner_0/MatMul?
4autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_model_enc_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype026
4autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOp?
%autoencoder/model/enc_inner_0/BiasAddBiasAdd.autoencoder/model/enc_inner_0/MatMul:product:0<autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2'
%autoencoder/model/enc_inner_0/BiasAdd?
"autoencoder/model/enc_inner_0/ReluRelu.autoencoder/model/enc_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2$
"autoencoder/model/enc_inner_0/Relu?
1autoencoder/model/channel_0/MatMul/ReadVariableOpReadVariableOp:autoencoder_model_channel_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype023
1autoencoder/model/channel_0/MatMul/ReadVariableOp?
"autoencoder/model/channel_0/MatMulMatMul0autoencoder/model/enc_inner_0/Relu:activations:09autoencoder/model/channel_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/model/channel_0/MatMul?
2autoencoder/model/channel_0/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_model_channel_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/model/channel_0/BiasAdd/ReadVariableOp?
#autoencoder/model/channel_0/BiasAddBiasAdd,autoencoder/model/channel_0/MatMul:product:0:autoencoder/model/channel_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/model/channel_0/BiasAdd?
$autoencoder/model/channel_0/SoftsignSoftsign,autoencoder/model/channel_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2&
$autoencoder/model/channel_0/Softsign?
5autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOpReadVariableOp>autoencoder_model_1_dec_inner_0_matmul_readvariableop_resource*
_output_shapes

:(*
dtype027
5autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOp?
&autoencoder/model_1/dec_inner_0/MatMulMatMul2autoencoder/model/channel_0/Softsign:activations:0=autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2(
&autoencoder/model_1/dec_inner_0/MatMul?
6autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_model_1_dec_inner_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype028
6autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOp?
'autoencoder/model_1/dec_inner_0/BiasAddBiasAdd0autoencoder/model_1/dec_inner_0/MatMul:product:0>autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'autoencoder/model_1/dec_inner_0/BiasAdd?
$autoencoder/model_1/dec_inner_0/ReluRelu0autoencoder/model_1/dec_inner_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2&
$autoencoder/model_1/dec_inner_0/Relu?
6autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOpReadVariableOp?autoencoder_model_1_dec_middle_0_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype028
6autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOp?
'autoencoder/model_1/dec_middle_0/MatMulMatMul2autoencoder/model_1/dec_inner_0/Relu:activations:0>autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder/model_1/dec_middle_0/MatMul?
7autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_model_1_dec_middle_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype029
7autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOp?
(autoencoder/model_1/dec_middle_0/BiasAddBiasAdd1autoencoder/model_1/dec_middle_0/MatMul:product:0?autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder/model_1/dec_middle_0/BiasAdd?
%autoencoder/model_1/dec_middle_0/ReluRelu1autoencoder/model_1/dec_middle_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2'
%autoencoder/model_1/dec_middle_0/Relu?
5autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOpReadVariableOp>autoencoder_model_1_dec_outer_0_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype027
5autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOp?
&autoencoder/model_1/dec_outer_0/MatMulMatMul3autoencoder/model_1/dec_middle_0/Relu:activations:0=autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2(
&autoencoder/model_1/dec_outer_0/MatMul?
6autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_model_1_dec_outer_0_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype028
6autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOp?
'autoencoder/model_1/dec_outer_0/BiasAddBiasAdd0autoencoder/model_1/dec_outer_0/MatMul:product:0>autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2)
'autoencoder/model_1/dec_outer_0/BiasAdd?
$autoencoder/model_1/dec_outer_0/ReluRelu0autoencoder/model_1/dec_outer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2&
$autoencoder/model_1/dec_outer_0/Relu?
(autoencoder/model_1/tf.identity/IdentityIdentity2autoencoder/model_1/dec_outer_0/Relu:activations:0*
T0*'
_output_shapes
:?????????<2*
(autoencoder/model_1/tf.identity/Identity?
4autoencoder/model_1/dec_output/MatMul/ReadVariableOpReadVariableOp=autoencoder_model_1_dec_output_matmul_readvariableop_resource*
_output_shapes
:	<?*
dtype026
4autoencoder/model_1/dec_output/MatMul/ReadVariableOp?
%autoencoder/model_1/dec_output/MatMulMatMul1autoencoder/model_1/tf.identity/Identity:output:0<autoencoder/model_1/dec_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%autoencoder/model_1/dec_output/MatMul?
5autoencoder/model_1/dec_output/BiasAdd/ReadVariableOpReadVariableOp>autoencoder_model_1_dec_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5autoencoder/model_1/dec_output/BiasAdd/ReadVariableOp?
&autoencoder/model_1/dec_output/BiasAddBiasAdd/autoencoder/model_1/dec_output/MatMul:product:0=autoencoder/model_1/dec_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/model_1/dec_output/BiasAdd?
&autoencoder/model_1/dec_output/SigmoidSigmoid/autoencoder/model_1/dec_output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/model_1/dec_output/Sigmoid?
IdentityIdentity*autoencoder/model_1/dec_output/Sigmoid:y:03^autoencoder/model/channel_0/BiasAdd/ReadVariableOp2^autoencoder/model/channel_0/MatMul/ReadVariableOp5^autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOp4^autoencoder/model/enc_inner_0/MatMul/ReadVariableOp6^autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOp5^autoencoder/model/enc_middle_0/MatMul/ReadVariableOp5^autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOp4^autoencoder/model/enc_outer_0/MatMul/ReadVariableOp7^autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOp6^autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOp8^autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOp7^autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOp7^autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOp6^autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOp6^autoencoder/model_1/dec_output/BiasAdd/ReadVariableOp5^autoencoder/model_1/dec_output/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2h
2autoencoder/model/channel_0/BiasAdd/ReadVariableOp2autoencoder/model/channel_0/BiasAdd/ReadVariableOp2f
1autoencoder/model/channel_0/MatMul/ReadVariableOp1autoencoder/model/channel_0/MatMul/ReadVariableOp2l
4autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOp4autoencoder/model/enc_inner_0/BiasAdd/ReadVariableOp2j
3autoencoder/model/enc_inner_0/MatMul/ReadVariableOp3autoencoder/model/enc_inner_0/MatMul/ReadVariableOp2n
5autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOp5autoencoder/model/enc_middle_0/BiasAdd/ReadVariableOp2l
4autoencoder/model/enc_middle_0/MatMul/ReadVariableOp4autoencoder/model/enc_middle_0/MatMul/ReadVariableOp2l
4autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOp4autoencoder/model/enc_outer_0/BiasAdd/ReadVariableOp2j
3autoencoder/model/enc_outer_0/MatMul/ReadVariableOp3autoencoder/model/enc_outer_0/MatMul/ReadVariableOp2p
6autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOp6autoencoder/model_1/dec_inner_0/BiasAdd/ReadVariableOp2n
5autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOp5autoencoder/model_1/dec_inner_0/MatMul/ReadVariableOp2r
7autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOp7autoencoder/model_1/dec_middle_0/BiasAdd/ReadVariableOp2p
6autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOp6autoencoder/model_1/dec_middle_0/MatMul/ReadVariableOp2p
6autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOp6autoencoder/model_1/dec_outer_0/BiasAdd/ReadVariableOp2n
5autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOp5autoencoder/model_1/dec_outer_0/MatMul/ReadVariableOp2n
5autoencoder/model_1/dec_output/BiasAdd/ReadVariableOp5autoencoder/model_1/dec_output/BiasAdd/ReadVariableOp2l
4autoencoder/model_1/dec_output/MatMul/ReadVariableOp4autoencoder/model_1/dec_output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
+__inference_autoencoder_layer_call_fn_96750
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_967152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
#__inference_signature_wrapper_96872
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *)
f$R"
 __inference__wrapped_model_960532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
@__inference_model_layer_call_and_return_conditional_losses_96262

inputs
enc_outer_0_96241
enc_outer_0_96243
enc_middle_0_96246
enc_middle_0_96248
enc_inner_0_96251
enc_inner_0_96253
channel_0_96256
channel_0_96258
identity??!channel_0/StatefulPartitionedCall?#enc_inner_0/StatefulPartitionedCall?$enc_middle_0/StatefulPartitionedCall?#enc_outer_0/StatefulPartitionedCall?
#enc_outer_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_outer_0_96241enc_outer_0_96243*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_960682%
#enc_outer_0/StatefulPartitionedCall?
$enc_middle_0/StatefulPartitionedCallStatefulPartitionedCall,enc_outer_0/StatefulPartitionedCall:output:0enc_middle_0_96246enc_middle_0_96248*
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
GPU2*0,1J 8? *P
fKRI
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_960952&
$enc_middle_0/StatefulPartitionedCall?
#enc_inner_0/StatefulPartitionedCallStatefulPartitionedCall-enc_middle_0/StatefulPartitionedCall:output:0enc_inner_0_96251enc_inner_0_96253*
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
GPU2*0,1J 8? *O
fJRH
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_961222%
#enc_inner_0/StatefulPartitionedCall?
!channel_0/StatefulPartitionedCallStatefulPartitionedCall,enc_inner_0/StatefulPartitionedCall:output:0channel_0_96256channel_0_96258*
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
GPU2*0,1J 8? *M
fHRF
D__inference_channel_0_layer_call_and_return_conditional_losses_961492#
!channel_0/StatefulPartitionedCall?
IdentityIdentity*channel_0/StatefulPartitionedCall:output:0"^channel_0/StatefulPartitionedCall$^enc_inner_0/StatefulPartitionedCall%^enc_middle_0/StatefulPartitionedCall$^enc_outer_0/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2F
!channel_0/StatefulPartitionedCall!channel_0/StatefulPartitionedCall2J
#enc_inner_0/StatefulPartitionedCall#enc_inner_0/StatefulPartitionedCall2L
$enc_middle_0/StatefulPartitionedCall$enc_middle_0/StatefulPartitionedCall2J
#enc_outer_0/StatefulPartitionedCall#enc_outer_0/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96636
input_1
model_96559
model_96561
model_96563
model_96565
model_96567
model_96569
model_96571
model_96573
model_1_96618
model_1_96620
model_1_96622
model_1_96624
model_1_96626
model_1_96628
model_1_96630
model_1_96632
identity??model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_96559model_96561model_96563model_96565model_96567model_96569model_96571model_96573*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962172
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_96618model_1_96620model_1_96622model_1_96624model_1_96626model_1_96628model_1_96630model_1_96632*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964482!
model_1/StatefulPartitionedCall?
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
+__inference_dec_inner_0_layer_call_fn_97382

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
GPU2*0,1J 8? *O
fJRH
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_962962
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
?)
?
@__inference_model_layer_call_and_return_conditional_losses_97132

inputs.
*enc_outer_0_matmul_readvariableop_resource/
+enc_outer_0_biasadd_readvariableop_resource/
+enc_middle_0_matmul_readvariableop_resource0
,enc_middle_0_biasadd_readvariableop_resource.
*enc_inner_0_matmul_readvariableop_resource/
+enc_inner_0_biasadd_readvariableop_resource,
(channel_0_matmul_readvariableop_resource-
)channel_0_biasadd_readvariableop_resource
identity?? channel_0/BiasAdd/ReadVariableOp?channel_0/MatMul/ReadVariableOp?"enc_inner_0/BiasAdd/ReadVariableOp?!enc_inner_0/MatMul/ReadVariableOp?#enc_middle_0/BiasAdd/ReadVariableOp?"enc_middle_0/MatMul/ReadVariableOp?"enc_outer_0/BiasAdd/ReadVariableOp?!enc_outer_0/MatMul/ReadVariableOp?
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
channel_0/Softsign?
IdentityIdentity channel_0/Softsign:activations:0!^channel_0/BiasAdd/ReadVariableOp ^channel_0/MatMul/ReadVariableOp#^enc_inner_0/BiasAdd/ReadVariableOp"^enc_inner_0/MatMul/ReadVariableOp$^enc_middle_0/BiasAdd/ReadVariableOp#^enc_middle_0/MatMul/ReadVariableOp#^enc_outer_0/BiasAdd/ReadVariableOp"^enc_outer_0/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2D
 channel_0/BiasAdd/ReadVariableOp channel_0/BiasAdd/ReadVariableOp2B
channel_0/MatMul/ReadVariableOpchannel_0/MatMul/ReadVariableOp2H
"enc_inner_0/BiasAdd/ReadVariableOp"enc_inner_0/BiasAdd/ReadVariableOp2F
!enc_inner_0/MatMul/ReadVariableOp!enc_inner_0/MatMul/ReadVariableOp2J
#enc_middle_0/BiasAdd/ReadVariableOp#enc_middle_0/BiasAdd/ReadVariableOp2H
"enc_middle_0/MatMul/ReadVariableOp"enc_middle_0/MatMul/ReadVariableOp2H
"enc_outer_0/BiasAdd/ReadVariableOp"enc_outer_0/BiasAdd/ReadVariableOp2F
!enc_outer_0/MatMul/ReadVariableOp!enc_outer_0/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_96122

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
?
?
%__inference_model_layer_call_fn_97153

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_97261

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96790
x
model_96755
model_96757
model_96759
model_96761
model_96763
model_96765
model_96767
model_96769
model_1_96772
model_1_96774
model_1_96776
model_1_96778
model_1_96780
model_1_96782
model_1_96784
model_1_96786
identity??model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallxmodel_96755model_96757model_96759model_96761model_96763model_96765model_96767model_96769*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962622
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_96772model_1_96774model_1_96776model_1_96778model_1_96780model_1_96782model_1_96784model_1_96786*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964942!
model_1/StatefulPartitionedCall?
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
%__inference_model_layer_call_fn_97174

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_962622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dec_output_layer_call_fn_97442

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
GPU2*0,1J 8? *N
fIRG
E__inference_dec_output_layer_call_and_return_conditional_losses_963782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
,__inference_enc_middle_0_layer_call_fn_97322

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
GPU2*0,1J 8? *P
fKRI
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_960952
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
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_96068

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
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_96095

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
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_97313

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
D__inference_channel_0_layer_call_and_return_conditional_losses_97353

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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_97393

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
?
?
'__inference_model_1_layer_call_fn_97282

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_96467
decoder_input_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_input_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_964482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedecoder_input_0"?L
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
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
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
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?,
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
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?)
_tf_keras_network?){"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 784]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_outer_0", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_middle_0", "inbound_nodes": [[["enc_outer_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "enc_inner_0", "inbound_nodes": [[["enc_middle_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "channel_0", "inbound_nodes": [[["enc_inner_0", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["channel_0", 0, 0]]}}}
?/
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?-
_tf_keras_network?,{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.identity", "trainable": true, "dtype": "float32", "function": "identity"}, "name": "tf.identity", "inbound_nodes": [["dec_outer_0", 0, 0, {"name": "concat"}]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.identity", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0]], "output_layers": [["dec_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}, "name": "decoder_input_0", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_inner_0", "inbound_nodes": [[["decoder_input_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_middle_0", "inbound_nodes": [[["dec_inner_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_outer_0", "inbound_nodes": [[["dec_middle_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.identity", "trainable": true, "dtype": "float32", "function": "identity"}, "name": "tf.identity", "inbound_nodes": [["dec_outer_0", 0, 0, {"name": "concat"}]]}, {"class_name": "Dense", "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_output", "inbound_nodes": [[["tf.identity", 0, 0, {}]]]}], "input_layers": [["decoder_input_0", 0, 0]], "output_layers": [["dec_output", 0, 0]]}}}
?
iter

beta_1

beta_2
	decay
 learning_rate!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?"
	optimizer
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015"
trackable_list_wrapper
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
1metrics
2non_trainable_variables
3layer_regularization_losses
trainable_variables
4layer_metrics
regularization_losses

5layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
?

!kernel
"bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

#kernel
$bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_middle_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

%kernel
&bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

'kernel
(bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "channel_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "channel_0", "trainable": true, "dtype": "float32", "units": 1, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
X
!0
"1
#2
$3
%4
&5
'6
(7"
trackable_list_wrapper
X
!0
"1
#2
$3
%4
&5
'6
(7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Fmetrics
Gnon_trainable_variables
Hlayer_regularization_losses
trainable_variables
Ilayer_metrics
regularization_losses

Jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input_0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input_0"}}
?

)kernel
*bias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_inner_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_inner_0", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

+kernel
,bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_middle_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_middle_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?

-kernel
.bias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_outer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_outer_0", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?
W	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.identity", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.identity", "trainable": true, "dtype": "float32", "function": "identity"}}
?

/kernel
0bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
\metrics
]non_trainable_variables
^layer_regularization_losses
trainable_variables
_layer_metrics
regularization_losses

`layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
%:#	?<2enc_outer_0/kernel
:<2enc_outer_0/bias
%:#<22enc_middle_0/kernel
:22enc_middle_0/bias
$:"2(2enc_inner_0/kernel
:(2enc_inner_0/bias
": (2channel_0/kernel
:2channel_0/bias
$:"(2dec_inner_0/kernel
:(2dec_inner_0/bias
%:#(<2dec_middle_0/kernel
:<2dec_middle_0/bias
$:"<<2dec_outer_0/kernel
:<2dec_outer_0/bias
$:"	<?2dec_output/kernel
:?2dec_output/bias
'
a0"
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
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6trainable_variables
bmetrics
cnon_trainable_variables
dlayer_regularization_losses
7	variables
elayer_metrics
8regularization_losses

flayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:trainable_variables
gmetrics
hnon_trainable_variables
ilayer_regularization_losses
;	variables
jlayer_metrics
<regularization_losses

klayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>trainable_variables
lmetrics
mnon_trainable_variables
nlayer_regularization_losses
?	variables
olayer_metrics
@regularization_losses

players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Btrainable_variables
qmetrics
rnon_trainable_variables
slayer_regularization_losses
C	variables
tlayer_metrics
Dregularization_losses

ulayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
	0

1
2
3
4"
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
Ktrainable_variables
vmetrics
wnon_trainable_variables
xlayer_regularization_losses
L	variables
ylayer_metrics
Mregularization_losses

zlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
Otrainable_variables
{metrics
|non_trainable_variables
}layer_regularization_losses
P	variables
~layer_metrics
Qregularization_losses

layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
Strainable_variables
?metrics
?non_trainable_variables
 ?layer_regularization_losses
T	variables
?layer_metrics
Uregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
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
*:(<22Adam/enc_middle_0/kernel/m
$:"22Adam/enc_middle_0/bias/m
):'2(2Adam/enc_inner_0/kernel/m
#:!(2Adam/enc_inner_0/bias/m
':%(2Adam/channel_0/kernel/m
!:2Adam/channel_0/bias/m
):'(2Adam/dec_inner_0/kernel/m
#:!(2Adam/dec_inner_0/bias/m
*:((<2Adam/dec_middle_0/kernel/m
$:"<2Adam/dec_middle_0/bias/m
):'<<2Adam/dec_outer_0/kernel/m
#:!<2Adam/dec_outer_0/bias/m
):'	<?2Adam/dec_output/kernel/m
#:!?2Adam/dec_output/bias/m
*:(	?<2Adam/enc_outer_0/kernel/v
#:!<2Adam/enc_outer_0/bias/v
*:(<22Adam/enc_middle_0/kernel/v
$:"22Adam/enc_middle_0/bias/v
):'2(2Adam/enc_inner_0/kernel/v
#:!(2Adam/enc_inner_0/bias/v
':%(2Adam/channel_0/kernel/v
!:2Adam/channel_0/bias/v
):'(2Adam/dec_inner_0/kernel/v
#:!(2Adam/dec_inner_0/bias/v
*:((<2Adam/dec_middle_0/kernel/v
$:"<2Adam/dec_middle_0/bias/v
):'<<2Adam/dec_outer_0/kernel/v
#:!<2Adam/dec_outer_0/bias/v
):'	<?2Adam/dec_output/kernel/v
#:!?2Adam/dec_output/bias/v
?2?
 __inference__wrapped_model_96053?
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
F__inference_autoencoder_layer_call_and_return_conditional_losses_96933
F__inference_autoencoder_layer_call_and_return_conditional_losses_96994
F__inference_autoencoder_layer_call_and_return_conditional_losses_96674
F__inference_autoencoder_layer_call_and_return_conditional_losses_96636?
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
?2?
+__inference_autoencoder_layer_call_fn_97031
+__inference_autoencoder_layer_call_fn_96825
+__inference_autoencoder_layer_call_fn_97068
+__inference_autoencoder_layer_call_fn_96750?
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
@__inference_model_layer_call_and_return_conditional_losses_96166
@__inference_model_layer_call_and_return_conditional_losses_96190
@__inference_model_layer_call_and_return_conditional_losses_97100
@__inference_model_layer_call_and_return_conditional_losses_97132?
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
%__inference_model_layer_call_fn_96236
%__inference_model_layer_call_fn_97153
%__inference_model_layer_call_fn_96281
%__inference_model_layer_call_fn_97174?
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
B__inference_model_1_layer_call_and_return_conditional_losses_97240
B__inference_model_1_layer_call_and_return_conditional_losses_96395
B__inference_model_1_layer_call_and_return_conditional_losses_97207
B__inference_model_1_layer_call_and_return_conditional_losses_96420?
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
'__inference_model_1_layer_call_fn_97261
'__inference_model_1_layer_call_fn_97282
'__inference_model_1_layer_call_fn_96467
'__inference_model_1_layer_call_fn_96513?
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
#__inference_signature_wrapper_96872input_1"?
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
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_97293?
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
+__inference_enc_outer_0_layer_call_fn_97302?
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
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_97313?
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
,__inference_enc_middle_0_layer_call_fn_97322?
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
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_97333?
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
+__inference_enc_inner_0_layer_call_fn_97342?
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
D__inference_channel_0_layer_call_and_return_conditional_losses_97353?
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
)__inference_channel_0_layer_call_fn_97362?
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
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_97373?
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
+__inference_dec_inner_0_layer_call_fn_97382?
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
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_97393?
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
,__inference_dec_middle_0_layer_call_fn_97402?
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
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_97413?
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
+__inference_dec_outer_0_layer_call_fn_97422?
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
E__inference_dec_output_layer_call_and_return_conditional_losses_97433?
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
*__inference_dec_output_layer_call_fn_97442?
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
 __inference__wrapped_model_96053{!"#$%&'()*+,-./01?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
F__inference_autoencoder_layer_call_and_return_conditional_losses_96636}!"#$%&'()*+,-./0A?>
'?$
"?
input_1??????????
?

trainingp"&?#
?
0??????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96674}!"#$%&'()*+,-./0A?>
'?$
"?
input_1??????????
?

trainingp "&?#
?
0??????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96933w!"#$%&'()*+,-./0;?8
!?
?
x??????????
?

trainingp"&?#
?
0??????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_96994w!"#$%&'()*+,-./0;?8
!?
?
x??????????
?

trainingp "&?#
?
0??????????
? ?
+__inference_autoencoder_layer_call_fn_96750p!"#$%&'()*+,-./0A?>
'?$
"?
input_1??????????
?

trainingp"????????????
+__inference_autoencoder_layer_call_fn_96825p!"#$%&'()*+,-./0A?>
'?$
"?
input_1??????????
?

trainingp "????????????
+__inference_autoencoder_layer_call_fn_97031j!"#$%&'()*+,-./0;?8
!?
?
x??????????
?

trainingp"????????????
+__inference_autoencoder_layer_call_fn_97068j!"#$%&'()*+,-./0;?8
!?
?
x??????????
?

trainingp "????????????
D__inference_channel_0_layer_call_and_return_conditional_losses_97353\'(/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? |
)__inference_channel_0_layer_call_fn_97362O'(/?,
%?"
 ?
inputs?????????(
? "???????????
F__inference_dec_inner_0_layer_call_and_return_conditional_losses_97373\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? ~
+__inference_dec_inner_0_layer_call_fn_97382O)*/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_dec_middle_0_layer_call_and_return_conditional_losses_97393\+,/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????<
? 
,__inference_dec_middle_0_layer_call_fn_97402O+,/?,
%?"
 ?
inputs?????????(
? "??????????<?
F__inference_dec_outer_0_layer_call_and_return_conditional_losses_97413\-./?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? ~
+__inference_dec_outer_0_layer_call_fn_97422O-./?,
%?"
 ?
inputs?????????<
? "??????????<?
E__inference_dec_output_layer_call_and_return_conditional_losses_97433]/0/?,
%?"
 ?
inputs?????????<
? "&?#
?
0??????????
? ~
*__inference_dec_output_layer_call_fn_97442P/0/?,
%?"
 ?
inputs?????????<
? "????????????
F__inference_enc_inner_0_layer_call_and_return_conditional_losses_97333\%&/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? ~
+__inference_enc_inner_0_layer_call_fn_97342O%&/?,
%?"
 ?
inputs?????????2
? "??????????(?
G__inference_enc_middle_0_layer_call_and_return_conditional_losses_97313\#$/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????2
? 
,__inference_enc_middle_0_layer_call_fn_97322O#$/?,
%?"
 ?
inputs?????????<
? "??????????2?
F__inference_enc_outer_0_layer_call_and_return_conditional_losses_97293]!"0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????<
? 
+__inference_enc_outer_0_layer_call_fn_97302P!"0?-
&?#
!?
inputs??????????
? "??????????<?
B__inference_model_1_layer_call_and_return_conditional_losses_96395t)*+,-./0@?=
6?3
)?&
decoder_input_0?????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_96420t)*+,-./0@?=
6?3
)?&
decoder_input_0?????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_97207k)*+,-./07?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_97240k)*+,-./07?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
'__inference_model_1_layer_call_fn_96467g)*+,-./0@?=
6?3
)?&
decoder_input_0?????????
p

 
? "????????????
'__inference_model_1_layer_call_fn_96513g)*+,-./0@?=
6?3
)?&
decoder_input_0?????????
p 

 
? "????????????
'__inference_model_1_layer_call_fn_97261^)*+,-./07?4
-?*
 ?
inputs?????????
p

 
? "????????????
'__inference_model_1_layer_call_fn_97282^)*+,-./07?4
-?*
 ?
inputs?????????
p 

 
? "????????????
@__inference_model_layer_call_and_return_conditional_losses_96166r!"#$%&'(??<
5?2
(?%
encoder_input??????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_96190r!"#$%&'(??<
5?2
(?%
encoder_input??????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_97100k!"#$%&'(8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_97132k!"#$%&'(8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_96236e!"#$%&'(??<
5?2
(?%
encoder_input??????????
p

 
? "???????????
%__inference_model_layer_call_fn_96281e!"#$%&'(??<
5?2
(?%
encoder_input??????????
p 

 
? "???????????
%__inference_model_layer_call_fn_97153^!"#$%&'(8?5
.?+
!?
inputs??????????
p

 
? "???????????
%__inference_model_layer_call_fn_97174^!"#$%&'(8?5
.?+
!?
inputs??????????
p 

 
? "???????????
#__inference_signature_wrapper_96872?!"#$%&'()*+,-./0<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????