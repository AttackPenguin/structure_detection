??
??
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
|
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_56/kernel
u
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel* 
_output_shapes
:
??*
dtype0
s
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_56/bias
l
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes	
:?*
dtype0
{
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d* 
shared_namedense_57/kernel
t
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes
:	?d*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:d*
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

:dd*
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
:d*
dtype0
z
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d	* 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

:d	*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:	*
dtype0
z
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	d* 
shared_namedense_60/kernel
s
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes

:	d*
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
:d*
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

:dd*
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
:d*
dtype0
{
dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?* 
shared_namedense_62/kernel
t
#dense_62/kernel/Read/ReadVariableOpReadVariableOpdense_62/kernel*
_output_shapes
:	d?*
dtype0
s
dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_62/bias
l
!dense_62/bias/Read/ReadVariableOpReadVariableOpdense_62/bias*
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
Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_56/kernel/m
?
*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_56/bias/m
z
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameAdam/dense_57/kernel/m
?
*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_58/kernel/m
?
*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d	*'
shared_nameAdam/dense_59/kernel/m
?
*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

:d	*
dtype0
?
Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:	*
dtype0
?
Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	d*'
shared_nameAdam/dense_60/kernel/m
?
*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes

:	d*
dtype0
?
Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_61/kernel/m
?
*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*'
shared_nameAdam/dense_62/kernel/m
?
*Adam/dense_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/m*
_output_shapes
:	d?*
dtype0
?
Adam/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_62/bias/m
z
(Adam/dense_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_56/kernel/v
?
*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_56/bias/v
z
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameAdam/dense_57/kernel/v
?
*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_58/kernel/v
?
*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d	*'
shared_nameAdam/dense_59/kernel/v
?
*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

:d	*
dtype0
?
Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:	*
dtype0
?
Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	d*'
shared_nameAdam/dense_60/kernel/v
?
*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes

:	d*
dtype0
?
Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_61/kernel/v
?
*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*'
shared_nameAdam/dense_62/kernel/v
?
*Adam/dense_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/v*
_output_shapes
:	d?*
dtype0
?
Adam/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_62/bias/v
z
(Adam/dense_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?L
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?K
value?KB?K B?K
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
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratem? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?
f
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
f
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
 
?
-layer_metrics
	variables
.metrics

/layers
0layer_regularization_losses
1non_trainable_variables
trainable_variables
regularization_losses
 
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

kernel
 bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

!kernel
"bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

#kernel
$bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

%kernel
&bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
8
0
 1
!2
"3
#4
$5
%6
&7
8
0
 1
!2
"3
#4
$5
%6
&7
 
?
Flayer_metrics
	variables
Gmetrics

Hlayers
Ilayer_regularization_losses
Jnon_trainable_variables
trainable_variables
regularization_losses
h

'kernel
(bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

)kernel
*bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

+kernel
,bias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
R
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
*
'0
(1
)2
*3
+4
,5
*
'0
(1
)2
*3
+4
,5
 
?
[layer_metrics
	variables
\metrics

]layers
^layer_regularization_losses
_non_trainable_variables
trainable_variables
regularization_losses
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
KI
VARIABLE_VALUEdense_56/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_56/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_57/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_57/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_58/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_58/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_59/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_59/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_60/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_60/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_61/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_61/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_62/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_62/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 

`0

0
1
 
 
 
 
 
?
alayer_metrics
bmetrics
2	variables

clayers
dlayer_regularization_losses
enon_trainable_variables
3trainable_variables
4regularization_losses

0
 1

0
 1
 
?
flayer_metrics
gmetrics
6	variables

hlayers
ilayer_regularization_losses
jnon_trainable_variables
7trainable_variables
8regularization_losses

!0
"1

!0
"1
 
?
klayer_metrics
lmetrics
:	variables

mlayers
nlayer_regularization_losses
onon_trainable_variables
;trainable_variables
<regularization_losses

#0
$1

#0
$1
 
?
player_metrics
qmetrics
>	variables

rlayers
slayer_regularization_losses
tnon_trainable_variables
?trainable_variables
@regularization_losses

%0
&1

%0
&1
 
?
ulayer_metrics
vmetrics
B	variables

wlayers
xlayer_regularization_losses
ynon_trainable_variables
Ctrainable_variables
Dregularization_losses
 
 
#
	0

1
2
3
4
 
 

'0
(1

'0
(1
 
?
zlayer_metrics
{metrics
K	variables

|layers
}layer_regularization_losses
~non_trainable_variables
Ltrainable_variables
Mregularization_losses

)0
*1

)0
*1
 
?
layer_metrics
?metrics
O	variables
?layers
 ?layer_regularization_losses
?non_trainable_variables
Ptrainable_variables
Qregularization_losses

+0
,1

+0
,1
 
?
?layer_metrics
?metrics
S	variables
?layers
 ?layer_regularization_losses
?non_trainable_variables
Ttrainable_variables
Uregularization_losses
 
 
 
?
?layer_metrics
?metrics
W	variables
?layers
 ?layer_regularization_losses
?non_trainable_variables
Xtrainable_variables
Yregularization_losses
 
 

0
1
2
3
 
 
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
nl
VARIABLE_VALUEAdam/dense_56/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_56/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_57/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_57/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_58/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_58/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_59/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_59/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_60/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_60/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_61/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_61/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_62/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_62/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_56/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_56/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_57/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_57/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_58/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_58/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_59/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_59/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_60/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_60/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_61/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_61/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_62/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_62/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/biasdense_60/kerneldense_60/biasdense_61/kerneldense_61/biasdense_62/kerneldense_62/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_968946
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp#dense_62/kernel/Read/ReadVariableOp!dense_62/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp*Adam/dense_62/kernel/m/Read/ReadVariableOp(Adam/dense_62/bias/m/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOp*Adam/dense_62/kernel/v/Read/ReadVariableOp(Adam/dense_62/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
__inference__traced_save_969691
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/biasdense_60/kerneldense_60/biasdense_61/kerneldense_61/biasdense_62/kerneldense_62/biastotalcountAdam/dense_56/kernel/mAdam/dense_56/bias/mAdam/dense_57/kernel/mAdam/dense_57/bias/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/dense_59/kernel/mAdam/dense_59/bias/mAdam/dense_60/kernel/mAdam/dense_60/bias/mAdam/dense_61/kernel/mAdam/dense_61/bias/mAdam/dense_62/kernel/mAdam/dense_62/bias/mAdam/dense_56/kernel/vAdam/dense_56/bias/vAdam/dense_57/kernel/vAdam/dense_57/bias/vAdam/dense_58/kernel/vAdam/dense_58/bias/vAdam/dense_59/kernel/vAdam/dense_59/bias/vAdam/dense_60/kernel/vAdam/dense_60/bias/vAdam/dense_61/kernel/vAdam/dense_61/bias/vAdam/dense_62/kernel/vAdam/dense_62/bias/v*=
Tin6
422*
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
"__inference__traced_restore_969848??

?	
?
D__inference_dense_57_layer_call_and_return_conditional_losses_968266

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

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
?
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_968337
flatten_8_input
dense_56_968250
dense_56_968252
dense_57_968277
dense_57_968279
dense_58_968304
dense_58_968306
dense_59_968331
dense_59_968333
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallflatten_8_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_9682202
flatten_8/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_56_968250dense_56_968252*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_9682392"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_968277dense_57_968279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_9682662"
 dense_57/StatefulPartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_968304dense_58_968306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_9682932"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_968331dense_59_968333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_9683202"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?
?
.__inference_sequential_17_layer_call_fn_969352

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?	
?
D__inference_dense_57_layer_call_and_return_conditional_losses_969394

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

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
D__inference_dense_58_layer_call_and_return_conditional_losses_968293

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?h
?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_969010
x9
5sequential_16_dense_56_matmul_readvariableop_resource:
6sequential_16_dense_56_biasadd_readvariableop_resource9
5sequential_16_dense_57_matmul_readvariableop_resource:
6sequential_16_dense_57_biasadd_readvariableop_resource9
5sequential_16_dense_58_matmul_readvariableop_resource:
6sequential_16_dense_58_biasadd_readvariableop_resource9
5sequential_16_dense_59_matmul_readvariableop_resource:
6sequential_16_dense_59_biasadd_readvariableop_resource9
5sequential_17_dense_60_matmul_readvariableop_resource:
6sequential_17_dense_60_biasadd_readvariableop_resource9
5sequential_17_dense_61_matmul_readvariableop_resource:
6sequential_17_dense_61_biasadd_readvariableop_resource9
5sequential_17_dense_62_matmul_readvariableop_resource:
6sequential_17_dense_62_biasadd_readvariableop_resource
identity??-sequential_16/dense_56/BiasAdd/ReadVariableOp?,sequential_16/dense_56/MatMul/ReadVariableOp?-sequential_16/dense_57/BiasAdd/ReadVariableOp?,sequential_16/dense_57/MatMul/ReadVariableOp?-sequential_16/dense_58/BiasAdd/ReadVariableOp?,sequential_16/dense_58/MatMul/ReadVariableOp?-sequential_16/dense_59/BiasAdd/ReadVariableOp?,sequential_16/dense_59/MatMul/ReadVariableOp?-sequential_17/dense_60/BiasAdd/ReadVariableOp?,sequential_17/dense_60/MatMul/ReadVariableOp?-sequential_17/dense_61/BiasAdd/ReadVariableOp?,sequential_17/dense_61/MatMul/ReadVariableOp?-sequential_17/dense_62/BiasAdd/ReadVariableOp?,sequential_17/dense_62/MatMul/ReadVariableOp?
sequential_16/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
sequential_16/flatten_8/Const?
sequential_16/flatten_8/ReshapeReshapex&sequential_16/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2!
sequential_16/flatten_8/Reshape?
,sequential_16/dense_56/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_56_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_16/dense_56/MatMul/ReadVariableOp?
sequential_16/dense_56/MatMulMatMul(sequential_16/flatten_8/Reshape:output:04sequential_16/dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_16/dense_56/MatMul?
-sequential_16/dense_56/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_56_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_16/dense_56/BiasAdd/ReadVariableOp?
sequential_16/dense_56/BiasAddBiasAdd'sequential_16/dense_56/MatMul:product:05sequential_16/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_16/dense_56/BiasAdd?
sequential_16/dense_56/ReluRelu'sequential_16/dense_56/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_16/dense_56/Relu?
,sequential_16/dense_57/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_57_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02.
,sequential_16/dense_57/MatMul/ReadVariableOp?
sequential_16/dense_57/MatMulMatMul)sequential_16/dense_56/Relu:activations:04sequential_16/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_57/MatMul?
-sequential_16/dense_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_57_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_57/BiasAdd/ReadVariableOp?
sequential_16/dense_57/BiasAddBiasAdd'sequential_16/dense_57/MatMul:product:05sequential_16/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_16/dense_57/BiasAdd?
sequential_16/dense_57/ReluRelu'sequential_16/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_57/Relu?
,sequential_16/dense_58/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_58_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_16/dense_58/MatMul/ReadVariableOp?
sequential_16/dense_58/MatMulMatMul)sequential_16/dense_57/Relu:activations:04sequential_16/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_58/MatMul?
-sequential_16/dense_58/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_58_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_58/BiasAdd/ReadVariableOp?
sequential_16/dense_58/BiasAddBiasAdd'sequential_16/dense_58/MatMul:product:05sequential_16/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_16/dense_58/BiasAdd?
sequential_16/dense_58/ReluRelu'sequential_16/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_58/Relu?
,sequential_16/dense_59/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_59_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype02.
,sequential_16/dense_59/MatMul/ReadVariableOp?
sequential_16/dense_59/MatMulMatMul)sequential_16/dense_58/Relu:activations:04sequential_16/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential_16/dense_59/MatMul?
-sequential_16/dense_59/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_59_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02/
-sequential_16/dense_59/BiasAdd/ReadVariableOp?
sequential_16/dense_59/BiasAddBiasAdd'sequential_16/dense_59/MatMul:product:05sequential_16/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2 
sequential_16/dense_59/BiasAdd?
sequential_16/dense_59/SoftsignSoftsign'sequential_16/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2!
sequential_16/dense_59/Softsign?
,sequential_17/dense_60/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_60_matmul_readvariableop_resource*
_output_shapes

:	d*
dtype02.
,sequential_17/dense_60/MatMul/ReadVariableOp?
sequential_17/dense_60/MatMulMatMul-sequential_16/dense_59/Softsign:activations:04sequential_17/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_60/MatMul?
-sequential_17/dense_60/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_60_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_17/dense_60/BiasAdd/ReadVariableOp?
sequential_17/dense_60/BiasAddBiasAdd'sequential_17/dense_60/MatMul:product:05sequential_17/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_17/dense_60/BiasAdd?
sequential_17/dense_60/ReluRelu'sequential_17/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_60/Relu?
,sequential_17/dense_61/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_61_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_17/dense_61/MatMul/ReadVariableOp?
sequential_17/dense_61/MatMulMatMul)sequential_17/dense_60/Relu:activations:04sequential_17/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_61/MatMul?
-sequential_17/dense_61/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_61_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_17/dense_61/BiasAdd/ReadVariableOp?
sequential_17/dense_61/BiasAddBiasAdd'sequential_17/dense_61/MatMul:product:05sequential_17/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_17/dense_61/BiasAdd?
sequential_17/dense_61/ReluRelu'sequential_17/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_61/Relu?
,sequential_17/dense_62/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_62_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02.
,sequential_17/dense_62/MatMul/ReadVariableOp?
sequential_17/dense_62/MatMulMatMul)sequential_17/dense_61/Relu:activations:04sequential_17/dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_17/dense_62/MatMul?
-sequential_17/dense_62/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_17/dense_62/BiasAdd/ReadVariableOp?
sequential_17/dense_62/BiasAddBiasAdd'sequential_17/dense_62/MatMul:product:05sequential_17/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_62/BiasAdd?
sequential_17/dense_62/SigmoidSigmoid'sequential_17/dense_62/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_62/Sigmoid?
sequential_17/reshape_8/ShapeShape"sequential_17/dense_62/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_17/reshape_8/Shape?
+sequential_17/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_17/reshape_8/strided_slice/stack?
-sequential_17/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_17/reshape_8/strided_slice/stack_1?
-sequential_17/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_17/reshape_8/strided_slice/stack_2?
%sequential_17/reshape_8/strided_sliceStridedSlice&sequential_17/reshape_8/Shape:output:04sequential_17/reshape_8/strided_slice/stack:output:06sequential_17/reshape_8/strided_slice/stack_1:output:06sequential_17/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_17/reshape_8/strided_slice?
'sequential_17/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_17/reshape_8/Reshape/shape/1?
'sequential_17/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_17/reshape_8/Reshape/shape/2?
%sequential_17/reshape_8/Reshape/shapePack.sequential_17/reshape_8/strided_slice:output:00sequential_17/reshape_8/Reshape/shape/1:output:00sequential_17/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_17/reshape_8/Reshape/shape?
sequential_17/reshape_8/ReshapeReshape"sequential_17/dense_62/Sigmoid:y:0.sequential_17/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_17/reshape_8/Reshape?
IdentityIdentity(sequential_17/reshape_8/Reshape:output:0.^sequential_16/dense_56/BiasAdd/ReadVariableOp-^sequential_16/dense_56/MatMul/ReadVariableOp.^sequential_16/dense_57/BiasAdd/ReadVariableOp-^sequential_16/dense_57/MatMul/ReadVariableOp.^sequential_16/dense_58/BiasAdd/ReadVariableOp-^sequential_16/dense_58/MatMul/ReadVariableOp.^sequential_16/dense_59/BiasAdd/ReadVariableOp-^sequential_16/dense_59/MatMul/ReadVariableOp.^sequential_17/dense_60/BiasAdd/ReadVariableOp-^sequential_17/dense_60/MatMul/ReadVariableOp.^sequential_17/dense_61/BiasAdd/ReadVariableOp-^sequential_17/dense_61/MatMul/ReadVariableOp.^sequential_17/dense_62/BiasAdd/ReadVariableOp-^sequential_17/dense_62/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2^
-sequential_16/dense_56/BiasAdd/ReadVariableOp-sequential_16/dense_56/BiasAdd/ReadVariableOp2\
,sequential_16/dense_56/MatMul/ReadVariableOp,sequential_16/dense_56/MatMul/ReadVariableOp2^
-sequential_16/dense_57/BiasAdd/ReadVariableOp-sequential_16/dense_57/BiasAdd/ReadVariableOp2\
,sequential_16/dense_57/MatMul/ReadVariableOp,sequential_16/dense_57/MatMul/ReadVariableOp2^
-sequential_16/dense_58/BiasAdd/ReadVariableOp-sequential_16/dense_58/BiasAdd/ReadVariableOp2\
,sequential_16/dense_58/MatMul/ReadVariableOp,sequential_16/dense_58/MatMul/ReadVariableOp2^
-sequential_16/dense_59/BiasAdd/ReadVariableOp-sequential_16/dense_59/BiasAdd/ReadVariableOp2\
,sequential_16/dense_59/MatMul/ReadVariableOp,sequential_16/dense_59/MatMul/ReadVariableOp2^
-sequential_17/dense_60/BiasAdd/ReadVariableOp-sequential_17/dense_60/BiasAdd/ReadVariableOp2\
,sequential_17/dense_60/MatMul/ReadVariableOp,sequential_17/dense_60/MatMul/ReadVariableOp2^
-sequential_17/dense_61/BiasAdd/ReadVariableOp-sequential_17/dense_61/BiasAdd/ReadVariableOp2\
,sequential_17/dense_61/MatMul/ReadVariableOp,sequential_17/dense_61/MatMul/ReadVariableOp2^
-sequential_17/dense_62/BiasAdd/ReadVariableOp-sequential_17/dense_62/BiasAdd/ReadVariableOp2\
,sequential_17/dense_62/MatMul/ReadVariableOp,sequential_17/dense_62/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?)
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_969284

inputs+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource+
'dense_62_matmul_readvariableop_resource,
(dense_62_biasadd_readvariableop_resource
identity??dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/BiasAdd/ReadVariableOp?dense_62/MatMul/ReadVariableOp?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:	d*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMulinputs&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_60/MatMul?
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_60/BiasAdd/ReadVariableOp?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_60/Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_61/MatMul?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_61/BiasAdd/ReadVariableOp?
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_61/BiasAdds
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_61/Relu?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMuldense_61/Relu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_62/MatMul?
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_62/BiasAdd/ReadVariableOp?
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_62/BiasAdd}
dense_62/SigmoidSigmoiddense_62/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_62/Sigmoidf
reshape_8/ShapeShapedense_62/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapedense_62/Sigmoid:y:0 reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_8/Reshape?
IdentityIdentityreshape_8/Reshape:output:0 ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?)
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_969318

inputs+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource+
'dense_62_matmul_readvariableop_resource,
(dense_62_biasadd_readvariableop_resource
identity??dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/BiasAdd/ReadVariableOp?dense_62/MatMul/ReadVariableOp?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:	d*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMulinputs&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_60/MatMul?
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_60/BiasAdd/ReadVariableOp?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_60/Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_61/MatMul?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_61/BiasAdd/ReadVariableOp?
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_61/BiasAdds
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_61/Relu?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMuldense_61/Relu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_62/MatMul?
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_62/BiasAdd/ReadVariableOp?
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_62/BiasAdd}
dense_62/SigmoidSigmoiddense_62/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_62/Sigmoidf
reshape_8/ShapeShapedense_62/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapedense_62/Sigmoid:y:0 reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_8/Reshape?
IdentityIdentityreshape_8/Reshape:output:0 ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
~
)__inference_dense_60_layer_call_fn_969463

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
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_60_layer_call_and_return_conditional_losses_9684702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_968210
input_1G
Cautoencoder_8_sequential_16_dense_56_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_16_dense_56_biasadd_readvariableop_resourceG
Cautoencoder_8_sequential_16_dense_57_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_16_dense_57_biasadd_readvariableop_resourceG
Cautoencoder_8_sequential_16_dense_58_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_16_dense_58_biasadd_readvariableop_resourceG
Cautoencoder_8_sequential_16_dense_59_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_16_dense_59_biasadd_readvariableop_resourceG
Cautoencoder_8_sequential_17_dense_60_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_17_dense_60_biasadd_readvariableop_resourceG
Cautoencoder_8_sequential_17_dense_61_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_17_dense_61_biasadd_readvariableop_resourceG
Cautoencoder_8_sequential_17_dense_62_matmul_readvariableop_resourceH
Dautoencoder_8_sequential_17_dense_62_biasadd_readvariableop_resource
identity??;autoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOp?;autoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOp?;autoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOp?;autoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOp?
+autoencoder_8/sequential_16/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2-
+autoencoder_8/sequential_16/flatten_8/Const?
-autoencoder_8/sequential_16/flatten_8/ReshapeReshapeinput_14autoencoder_8/sequential_16/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2/
-autoencoder_8/sequential_16/flatten_8/Reshape?
:autoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_56_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:autoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOp?
+autoencoder_8/sequential_16/dense_56/MatMulMatMul6autoencoder_8/sequential_16/flatten_8/Reshape:output:0Bautoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_8/sequential_16/dense_56/MatMul?
;autoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_56_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;autoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_16/dense_56/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_56/MatMul:product:0Cautoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_16/dense_56/BiasAdd?
)autoencoder_8/sequential_16/dense_56/ReluRelu5autoencoder_8/sequential_16/dense_56/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2+
)autoencoder_8/sequential_16/dense_56/Relu?
:autoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_57_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02<
:autoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOp?
+autoencoder_8/sequential_16/dense_57/MatMulMatMul7autoencoder_8/sequential_16/dense_56/Relu:activations:0Bautoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_8/sequential_16/dense_57/MatMul?
;autoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_57_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_16/dense_57/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_57/MatMul:product:0Cautoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_8/sequential_16/dense_57/BiasAdd?
)autoencoder_8/sequential_16/dense_57/ReluRelu5autoencoder_8/sequential_16/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_8/sequential_16/dense_57/Relu?
:autoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_58_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02<
:autoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOp?
+autoencoder_8/sequential_16/dense_58/MatMulMatMul7autoencoder_8/sequential_16/dense_57/Relu:activations:0Bautoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_8/sequential_16/dense_58/MatMul?
;autoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_58_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_16/dense_58/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_58/MatMul:product:0Cautoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_8/sequential_16/dense_58/BiasAdd?
)autoencoder_8/sequential_16/dense_58/ReluRelu5autoencoder_8/sequential_16/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_8/sequential_16/dense_58/Relu?
:autoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_59_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype02<
:autoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOp?
+autoencoder_8/sequential_16/dense_59/MatMulMatMul7autoencoder_8/sequential_16/dense_58/Relu:activations:0Bautoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2-
+autoencoder_8/sequential_16/dense_59/MatMul?
;autoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_59_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02=
;autoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_16/dense_59/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_59/MatMul:product:0Cautoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2.
,autoencoder_8/sequential_16/dense_59/BiasAdd?
-autoencoder_8/sequential_16/dense_59/SoftsignSoftsign5autoencoder_8/sequential_16/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2/
-autoencoder_8/sequential_16/dense_59/Softsign?
:autoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_60_matmul_readvariableop_resource*
_output_shapes

:	d*
dtype02<
:autoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOp?
+autoencoder_8/sequential_17/dense_60/MatMulMatMul;autoencoder_8/sequential_16/dense_59/Softsign:activations:0Bautoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_8/sequential_17/dense_60/MatMul?
;autoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_60_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_17/dense_60/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_60/MatMul:product:0Cautoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_8/sequential_17/dense_60/BiasAdd?
)autoencoder_8/sequential_17/dense_60/ReluRelu5autoencoder_8/sequential_17/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_8/sequential_17/dense_60/Relu?
:autoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_61_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02<
:autoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOp?
+autoencoder_8/sequential_17/dense_61/MatMulMatMul7autoencoder_8/sequential_17/dense_60/Relu:activations:0Bautoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_8/sequential_17/dense_61/MatMul?
;autoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_61_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_17/dense_61/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_61/MatMul:product:0Cautoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_8/sequential_17/dense_61/BiasAdd?
)autoencoder_8/sequential_17/dense_61/ReluRelu5autoencoder_8/sequential_17/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_8/sequential_17/dense_61/Relu?
:autoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_62_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02<
:autoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOp?
+autoencoder_8/sequential_17/dense_62/MatMulMatMul7autoencoder_8/sequential_17/dense_61/Relu:activations:0Bautoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_8/sequential_17/dense_62/MatMul?
;autoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;autoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_17/dense_62/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_62/MatMul:product:0Cautoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_17/dense_62/BiasAdd?
,autoencoder_8/sequential_17/dense_62/SigmoidSigmoid5autoencoder_8/sequential_17/dense_62/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_17/dense_62/Sigmoid?
+autoencoder_8/sequential_17/reshape_8/ShapeShape0autoencoder_8/sequential_17/dense_62/Sigmoid:y:0*
T0*
_output_shapes
:2-
+autoencoder_8/sequential_17/reshape_8/Shape?
9autoencoder_8/sequential_17/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9autoencoder_8/sequential_17/reshape_8/strided_slice/stack?
;autoencoder_8/sequential_17/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;autoencoder_8/sequential_17/reshape_8/strided_slice/stack_1?
;autoencoder_8/sequential_17/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;autoencoder_8/sequential_17/reshape_8/strided_slice/stack_2?
3autoencoder_8/sequential_17/reshape_8/strided_sliceStridedSlice4autoencoder_8/sequential_17/reshape_8/Shape:output:0Bautoencoder_8/sequential_17/reshape_8/strided_slice/stack:output:0Dautoencoder_8/sequential_17/reshape_8/strided_slice/stack_1:output:0Dautoencoder_8/sequential_17/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3autoencoder_8/sequential_17/reshape_8/strided_slice?
5autoencoder_8/sequential_17/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5autoencoder_8/sequential_17/reshape_8/Reshape/shape/1?
5autoencoder_8/sequential_17/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5autoencoder_8/sequential_17/reshape_8/Reshape/shape/2?
3autoencoder_8/sequential_17/reshape_8/Reshape/shapePack<autoencoder_8/sequential_17/reshape_8/strided_slice:output:0>autoencoder_8/sequential_17/reshape_8/Reshape/shape/1:output:0>autoencoder_8/sequential_17/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:25
3autoencoder_8/sequential_17/reshape_8/Reshape/shape?
-autoencoder_8/sequential_17/reshape_8/ReshapeReshape0autoencoder_8/sequential_17/dense_62/Sigmoid:y:0<autoencoder_8/sequential_17/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2/
-autoencoder_8/sequential_17/reshape_8/Reshape?
IdentityIdentity6autoencoder_8/sequential_17/reshape_8/Reshape:output:0<^autoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOp<^autoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOp<^autoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOp<^autoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2z
;autoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_56/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_56/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_57/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_57/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_58/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_58/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_59/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_59/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_60/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_60/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_61/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_61/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_62/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_62/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
D__inference_dense_60_layer_call_and_return_conditional_losses_969454

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
~
)__inference_dense_56_layer_call_fn_969383

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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_9682392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968768
input_1
sequential_16_968703
sequential_16_968705
sequential_16_968707
sequential_16_968709
sequential_16_968711
sequential_16_968713
sequential_16_968715
sequential_16_968717
sequential_17_968754
sequential_17_968756
sequential_17_968758
sequential_17_968760
sequential_17_968762
sequential_17_968764
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_968703sequential_16_968705sequential_16_968707sequential_16_968709sequential_16_968711sequential_16_968713sequential_16_968715sequential_16_968717*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9683902'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_968754sequential_17_968756sequential_17_968758sequential_17_968760sequential_17_968762sequential_17_968764*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686052'
%sequential_17/StatefulPartitionedCall?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
D__inference_dense_59_layer_call_and_return_conditional_losses_968320

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_968642

inputs
dense_60_968625
dense_60_968627
dense_61_968630
dense_61_968632
dense_62_968635
dense_62_968637
identity?? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCallinputsdense_60_968625dense_60_968627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_60_layer_call_and_return_conditional_losses_9684702"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_968630dense_61_968632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_61_layer_call_and_return_conditional_losses_9684972"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_968635dense_62_968637*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_62_layer_call_and_return_conditional_losses_9685242"
 dense_62/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCall)dense_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_reshape_8_layer_call_and_return_conditional_losses_9685532
reshape_8/PartitionedCall?
IdentityIdentity"reshape_8/PartitionedCall:output:0!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
a
E__inference_reshape_8_layer_call_and_return_conditional_losses_969516

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_969174

inputs+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource
identity??dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOps
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_8/Const?
flatten_8/ReshapeReshapeinputsflatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_8/Reshape?
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_56/MatMul/ReadVariableOp?
dense_56/MatMulMatMulflatten_8/Reshape:output:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_56/MatMul?
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_56/BiasAdd/ReadVariableOp?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_56/BiasAddt
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_56/Relu?
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_57/MatMul/ReadVariableOp?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_57/MatMul?
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_57/BiasAdd/ReadVariableOp?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_57/BiasAdds
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_57/Relu?
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_58/MatMul/ReadVariableOp?
dense_58/MatMulMatMuldense_57/Relu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_58/MatMul?
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_58/BiasAdd/ReadVariableOp?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_58/BiasAdds
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_58/Relu?
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype02 
dense_59/MatMul/ReadVariableOp?
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_59/MatMul?
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_59/BiasAdd/ReadVariableOp?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_59/BiasAdd
dense_59/SoftsignSoftsigndense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_59/Softsign?
IdentityIdentitydense_59/Softsign:activations:0 ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_59_layer_call_and_return_conditional_losses_969434

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
D__inference_dense_61_layer_call_and_return_conditional_losses_969474

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
F
*__inference_reshape_8_layer_call_fn_969521

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_reshape_8_layer_call_and_return_conditional_losses_9685532
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?_
?
__inference__traced_save_969691
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop.
*savev2_dense_62_kernel_read_readvariableop,
(savev2_dense_62_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableop5
1savev2_adam_dense_62_kernel_m_read_readvariableop3
/savev2_adam_dense_62_bias_m_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableop5
1savev2_adam_dense_62_kernel_v_read_readvariableop3
/savev2_adam_dense_62_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop*savev2_dense_62_kernel_read_readvariableop(savev2_dense_62_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop1savev2_adam_dense_62_kernel_m_read_readvariableop/savev2_adam_dense_62_bias_m_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableop1savev2_adam_dense_62_kernel_v_read_readvariableop/savev2_adam_dense_62_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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
_input_shapes?
?: : : : : : :
??:?:	?d:d:dd:d:d	:	:	d:d:dd:d:	d?:?: : :
??:?:	?d:d:dd:d:d	:	:	d:d:dd:d:	d?:?:
??:?:	?d:d:dd:d:d	:	:	d:d:dd:d:	d?:?: 2(
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
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 	

_output_shapes
:d:$
 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d	: 

_output_shapes
:	:$ 

_output_shapes

:	d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:%!

_output_shapes
:	d?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d	: 

_output_shapes
:	:$ 

_output_shapes

:	d: 

_output_shapes
:d:$  

_output_shapes

:dd: !

_output_shapes
:d:%"!

_output_shapes
:	d?:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?d: '

_output_shapes
:d:$( 

_output_shapes

:dd: )

_output_shapes
:d:$* 

_output_shapes

:d	: +

_output_shapes
:	:$, 

_output_shapes

:	d: -

_output_shapes
:d:$. 

_output_shapes

:dd: /

_output_shapes
:d:%0!

_output_shapes
:	d?:!1

_output_shapes	
:?:2

_output_shapes
: 
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_968582
dense_60_input
dense_60_968565
dense_60_968567
dense_61_968570
dense_61_968572
dense_62_968575
dense_62_968577
identity?? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCalldense_60_inputdense_60_968565dense_60_968567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_60_layer_call_and_return_conditional_losses_9684702"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_968570dense_61_968572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_61_layer_call_and_return_conditional_losses_9684972"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_968575dense_62_968577*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_62_layer_call_and_return_conditional_losses_9685242"
 dense_62/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCall)dense_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_reshape_8_layer_call_and_return_conditional_losses_9685532
reshape_8/PartitionedCall?
IdentityIdentity"reshape_8/PartitionedCall:output:0!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:W S
'
_output_shapes
:?????????	
(
_user_specified_namedense_60_input
?)
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_969208

inputs+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource
identity??dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOps
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_8/Const?
flatten_8/ReshapeReshapeinputsflatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_8/Reshape?
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_56/MatMul/ReadVariableOp?
dense_56/MatMulMatMulflatten_8/Reshape:output:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_56/MatMul?
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_56/BiasAdd/ReadVariableOp?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_56/BiasAddt
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_56/Relu?
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_57/MatMul/ReadVariableOp?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_57/MatMul?
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_57/BiasAdd/ReadVariableOp?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_57/BiasAdds
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_57/Relu?
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_58/MatMul/ReadVariableOp?
dense_58/MatMulMatMuldense_57/Relu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_58/MatMul?
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_58/BiasAdd/ReadVariableOp?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_58/BiasAdds
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_58/Relu?
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype02 
dense_59/MatMul/ReadVariableOp?
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_59/MatMul?
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_59/BiasAdd/ReadVariableOp?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_59/BiasAdd
dense_59/SoftsignSoftsigndense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_59/Softsign?
IdentityIdentitydense_59/Softsign:activations:0 ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_56_layer_call_and_return_conditional_losses_969374

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
D__inference_dense_58_layer_call_and_return_conditional_losses_969414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
.__inference_sequential_16_layer_call_fn_968409
flatten_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9683902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?
~
)__inference_dense_58_layer_call_fn_969423

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
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_9682932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
.__inference_autoencoder_8_layer_call_fn_968903
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_9688392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
.__inference_autoencoder_8_layer_call_fn_968870
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_9688392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
$__inference_signature_wrapper_968946
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_9682102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
D__inference_dense_62_layer_call_and_return_conditional_losses_968524

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
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
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
~
)__inference_dense_57_layer_call_fn_969403

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
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_9682662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_16_layer_call_fn_969229

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
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9683902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_17_layer_call_fn_968620
dense_60_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_60_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????	
(
_user_specified_namedense_60_input
?
F
*__inference_flatten_8_layer_call_fn_969363

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_9682202
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_60_layer_call_and_return_conditional_losses_968470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968802
input_1
sequential_16_968771
sequential_16_968773
sequential_16_968775
sequential_16_968777
sequential_16_968779
sequential_16_968781
sequential_16_968783
sequential_16_968785
sequential_17_968788
sequential_17_968790
sequential_17_968792
sequential_17_968794
sequential_17_968796
sequential_17_968798
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_968771sequential_16_968773sequential_16_968775sequential_16_968777sequential_16_968779sequential_16_968781sequential_16_968783sequential_16_968785*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9684362'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_968788sequential_17_968790sequential_17_968792sequential_17_968794sequential_17_968796sequential_17_968798*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686422'
%sequential_17/StatefulPartitionedCall?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_968562
dense_60_input
dense_60_968481
dense_60_968483
dense_61_968508
dense_61_968510
dense_62_968535
dense_62_968537
identity?? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCalldense_60_inputdense_60_968481dense_60_968483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_60_layer_call_and_return_conditional_losses_9684702"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_968508dense_61_968510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_61_layer_call_and_return_conditional_losses_9684972"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_968535dense_62_968537*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_62_layer_call_and_return_conditional_losses_9685242"
 dense_62/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCall)dense_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_reshape_8_layer_call_and_return_conditional_losses_9685532
reshape_8/PartitionedCall?
IdentityIdentity"reshape_8/PartitionedCall:output:0!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:W S
'
_output_shapes
:?????????	
(
_user_specified_namedense_60_input
?	
?
.__inference_autoencoder_8_layer_call_fn_969107
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_9688392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
.__inference_sequential_17_layer_call_fn_969335

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968839
x
sequential_16_968808
sequential_16_968810
sequential_16_968812
sequential_16_968814
sequential_16_968816
sequential_16_968818
sequential_16_968820
sequential_16_968822
sequential_17_968825
sequential_17_968827
sequential_17_968829
sequential_17_968831
sequential_17_968833
sequential_17_968835
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallxsequential_16_968808sequential_16_968810sequential_16_968812sequential_16_968814sequential_16_968816sequential_16_968818sequential_16_968820sequential_16_968822*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9684362'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_968825sequential_17_968827sequential_17_968829sequential_17_968831sequential_17_968833sequential_17_968835*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686422'
%sequential_17/StatefulPartitionedCall?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
.__inference_sequential_17_layer_call_fn_968657
dense_60_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_60_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_9686422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????	
(
_user_specified_namedense_60_input
?
~
)__inference_dense_62_layer_call_fn_969503

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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_62_layer_call_and_return_conditional_losses_9685242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
D__inference_dense_62_layer_call_and_return_conditional_losses_969494

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
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
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_reshape_8_layer_call_and_return_conditional_losses_968553

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_968605

inputs
dense_60_968588
dense_60_968590
dense_61_968593
dense_61_968595
dense_62_968598
dense_62_968600
identity?? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCallinputsdense_60_968588dense_60_968590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_60_layer_call_and_return_conditional_losses_9684702"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_968593dense_61_968595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_61_layer_call_and_return_conditional_losses_9684972"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_968598dense_62_968600*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_62_layer_call_and_return_conditional_losses_9685242"
 dense_62/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCall)dense_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_reshape_8_layer_call_and_return_conditional_losses_9685532
reshape_8/PartitionedCall?
IdentityIdentity"reshape_8/PartitionedCall:output:0!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????	::::::2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_968362
flatten_8_input
dense_56_968341
dense_56_968343
dense_57_968346
dense_57_968348
dense_58_968351
dense_58_968353
dense_59_968356
dense_59_968358
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallflatten_8_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_9682202
flatten_8/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_56_968341dense_56_968343*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_9682392"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_968346dense_57_968348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_9682662"
 dense_57/StatefulPartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_968351dense_58_968353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_9682932"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_968356dense_59_968358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_9683202"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?h
?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_969074
x9
5sequential_16_dense_56_matmul_readvariableop_resource:
6sequential_16_dense_56_biasadd_readvariableop_resource9
5sequential_16_dense_57_matmul_readvariableop_resource:
6sequential_16_dense_57_biasadd_readvariableop_resource9
5sequential_16_dense_58_matmul_readvariableop_resource:
6sequential_16_dense_58_biasadd_readvariableop_resource9
5sequential_16_dense_59_matmul_readvariableop_resource:
6sequential_16_dense_59_biasadd_readvariableop_resource9
5sequential_17_dense_60_matmul_readvariableop_resource:
6sequential_17_dense_60_biasadd_readvariableop_resource9
5sequential_17_dense_61_matmul_readvariableop_resource:
6sequential_17_dense_61_biasadd_readvariableop_resource9
5sequential_17_dense_62_matmul_readvariableop_resource:
6sequential_17_dense_62_biasadd_readvariableop_resource
identity??-sequential_16/dense_56/BiasAdd/ReadVariableOp?,sequential_16/dense_56/MatMul/ReadVariableOp?-sequential_16/dense_57/BiasAdd/ReadVariableOp?,sequential_16/dense_57/MatMul/ReadVariableOp?-sequential_16/dense_58/BiasAdd/ReadVariableOp?,sequential_16/dense_58/MatMul/ReadVariableOp?-sequential_16/dense_59/BiasAdd/ReadVariableOp?,sequential_16/dense_59/MatMul/ReadVariableOp?-sequential_17/dense_60/BiasAdd/ReadVariableOp?,sequential_17/dense_60/MatMul/ReadVariableOp?-sequential_17/dense_61/BiasAdd/ReadVariableOp?,sequential_17/dense_61/MatMul/ReadVariableOp?-sequential_17/dense_62/BiasAdd/ReadVariableOp?,sequential_17/dense_62/MatMul/ReadVariableOp?
sequential_16/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
sequential_16/flatten_8/Const?
sequential_16/flatten_8/ReshapeReshapex&sequential_16/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2!
sequential_16/flatten_8/Reshape?
,sequential_16/dense_56/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_56_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_16/dense_56/MatMul/ReadVariableOp?
sequential_16/dense_56/MatMulMatMul(sequential_16/flatten_8/Reshape:output:04sequential_16/dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_16/dense_56/MatMul?
-sequential_16/dense_56/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_56_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_16/dense_56/BiasAdd/ReadVariableOp?
sequential_16/dense_56/BiasAddBiasAdd'sequential_16/dense_56/MatMul:product:05sequential_16/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_16/dense_56/BiasAdd?
sequential_16/dense_56/ReluRelu'sequential_16/dense_56/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_16/dense_56/Relu?
,sequential_16/dense_57/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_57_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02.
,sequential_16/dense_57/MatMul/ReadVariableOp?
sequential_16/dense_57/MatMulMatMul)sequential_16/dense_56/Relu:activations:04sequential_16/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_57/MatMul?
-sequential_16/dense_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_57_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_57/BiasAdd/ReadVariableOp?
sequential_16/dense_57/BiasAddBiasAdd'sequential_16/dense_57/MatMul:product:05sequential_16/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_16/dense_57/BiasAdd?
sequential_16/dense_57/ReluRelu'sequential_16/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_57/Relu?
,sequential_16/dense_58/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_58_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_16/dense_58/MatMul/ReadVariableOp?
sequential_16/dense_58/MatMulMatMul)sequential_16/dense_57/Relu:activations:04sequential_16/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_58/MatMul?
-sequential_16/dense_58/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_58_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_58/BiasAdd/ReadVariableOp?
sequential_16/dense_58/BiasAddBiasAdd'sequential_16/dense_58/MatMul:product:05sequential_16/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_16/dense_58/BiasAdd?
sequential_16/dense_58/ReluRelu'sequential_16/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_58/Relu?
,sequential_16/dense_59/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_59_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype02.
,sequential_16/dense_59/MatMul/ReadVariableOp?
sequential_16/dense_59/MatMulMatMul)sequential_16/dense_58/Relu:activations:04sequential_16/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential_16/dense_59/MatMul?
-sequential_16/dense_59/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_59_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02/
-sequential_16/dense_59/BiasAdd/ReadVariableOp?
sequential_16/dense_59/BiasAddBiasAdd'sequential_16/dense_59/MatMul:product:05sequential_16/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2 
sequential_16/dense_59/BiasAdd?
sequential_16/dense_59/SoftsignSoftsign'sequential_16/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2!
sequential_16/dense_59/Softsign?
,sequential_17/dense_60/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_60_matmul_readvariableop_resource*
_output_shapes

:	d*
dtype02.
,sequential_17/dense_60/MatMul/ReadVariableOp?
sequential_17/dense_60/MatMulMatMul-sequential_16/dense_59/Softsign:activations:04sequential_17/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_60/MatMul?
-sequential_17/dense_60/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_60_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_17/dense_60/BiasAdd/ReadVariableOp?
sequential_17/dense_60/BiasAddBiasAdd'sequential_17/dense_60/MatMul:product:05sequential_17/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_17/dense_60/BiasAdd?
sequential_17/dense_60/ReluRelu'sequential_17/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_60/Relu?
,sequential_17/dense_61/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_61_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_17/dense_61/MatMul/ReadVariableOp?
sequential_17/dense_61/MatMulMatMul)sequential_17/dense_60/Relu:activations:04sequential_17/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_61/MatMul?
-sequential_17/dense_61/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_61_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_17/dense_61/BiasAdd/ReadVariableOp?
sequential_17/dense_61/BiasAddBiasAdd'sequential_17/dense_61/MatMul:product:05sequential_17/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_17/dense_61/BiasAdd?
sequential_17/dense_61/ReluRelu'sequential_17/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_61/Relu?
,sequential_17/dense_62/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_62_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02.
,sequential_17/dense_62/MatMul/ReadVariableOp?
sequential_17/dense_62/MatMulMatMul)sequential_17/dense_61/Relu:activations:04sequential_17/dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_17/dense_62/MatMul?
-sequential_17/dense_62/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_17/dense_62/BiasAdd/ReadVariableOp?
sequential_17/dense_62/BiasAddBiasAdd'sequential_17/dense_62/MatMul:product:05sequential_17/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_62/BiasAdd?
sequential_17/dense_62/SigmoidSigmoid'sequential_17/dense_62/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_62/Sigmoid?
sequential_17/reshape_8/ShapeShape"sequential_17/dense_62/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_17/reshape_8/Shape?
+sequential_17/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_17/reshape_8/strided_slice/stack?
-sequential_17/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_17/reshape_8/strided_slice/stack_1?
-sequential_17/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_17/reshape_8/strided_slice/stack_2?
%sequential_17/reshape_8/strided_sliceStridedSlice&sequential_17/reshape_8/Shape:output:04sequential_17/reshape_8/strided_slice/stack:output:06sequential_17/reshape_8/strided_slice/stack_1:output:06sequential_17/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_17/reshape_8/strided_slice?
'sequential_17/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_17/reshape_8/Reshape/shape/1?
'sequential_17/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_17/reshape_8/Reshape/shape/2?
%sequential_17/reshape_8/Reshape/shapePack.sequential_17/reshape_8/strided_slice:output:00sequential_17/reshape_8/Reshape/shape/1:output:00sequential_17/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_17/reshape_8/Reshape/shape?
sequential_17/reshape_8/ReshapeReshape"sequential_17/dense_62/Sigmoid:y:0.sequential_17/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_17/reshape_8/Reshape?
IdentityIdentity(sequential_17/reshape_8/Reshape:output:0.^sequential_16/dense_56/BiasAdd/ReadVariableOp-^sequential_16/dense_56/MatMul/ReadVariableOp.^sequential_16/dense_57/BiasAdd/ReadVariableOp-^sequential_16/dense_57/MatMul/ReadVariableOp.^sequential_16/dense_58/BiasAdd/ReadVariableOp-^sequential_16/dense_58/MatMul/ReadVariableOp.^sequential_16/dense_59/BiasAdd/ReadVariableOp-^sequential_16/dense_59/MatMul/ReadVariableOp.^sequential_17/dense_60/BiasAdd/ReadVariableOp-^sequential_17/dense_60/MatMul/ReadVariableOp.^sequential_17/dense_61/BiasAdd/ReadVariableOp-^sequential_17/dense_61/MatMul/ReadVariableOp.^sequential_17/dense_62/BiasAdd/ReadVariableOp-^sequential_17/dense_62/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2^
-sequential_16/dense_56/BiasAdd/ReadVariableOp-sequential_16/dense_56/BiasAdd/ReadVariableOp2\
,sequential_16/dense_56/MatMul/ReadVariableOp,sequential_16/dense_56/MatMul/ReadVariableOp2^
-sequential_16/dense_57/BiasAdd/ReadVariableOp-sequential_16/dense_57/BiasAdd/ReadVariableOp2\
,sequential_16/dense_57/MatMul/ReadVariableOp,sequential_16/dense_57/MatMul/ReadVariableOp2^
-sequential_16/dense_58/BiasAdd/ReadVariableOp-sequential_16/dense_58/BiasAdd/ReadVariableOp2\
,sequential_16/dense_58/MatMul/ReadVariableOp,sequential_16/dense_58/MatMul/ReadVariableOp2^
-sequential_16/dense_59/BiasAdd/ReadVariableOp-sequential_16/dense_59/BiasAdd/ReadVariableOp2\
,sequential_16/dense_59/MatMul/ReadVariableOp,sequential_16/dense_59/MatMul/ReadVariableOp2^
-sequential_17/dense_60/BiasAdd/ReadVariableOp-sequential_17/dense_60/BiasAdd/ReadVariableOp2\
,sequential_17/dense_60/MatMul/ReadVariableOp,sequential_17/dense_60/MatMul/ReadVariableOp2^
-sequential_17/dense_61/BiasAdd/ReadVariableOp-sequential_17/dense_61/BiasAdd/ReadVariableOp2\
,sequential_17/dense_61/MatMul/ReadVariableOp,sequential_17/dense_61/MatMul/ReadVariableOp2^
-sequential_17/dense_62/BiasAdd/ReadVariableOp-sequential_17/dense_62/BiasAdd/ReadVariableOp2\
,sequential_17/dense_62/MatMul/ReadVariableOp,sequential_17/dense_62/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
.__inference_autoencoder_8_layer_call_fn_969140
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_9688392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_968390

inputs
dense_56_968369
dense_56_968371
dense_57_968374
dense_57_968376
dense_58_968379
dense_58_968381
dense_59_968384
dense_59_968386
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_9682202
flatten_8/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_56_968369dense_56_968371*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_9682392"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_968374dense_57_968376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_9682662"
 dense_57/StatefulPartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_968379dense_58_968381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_9682932"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_968384dense_59_968386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_9683202"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_16_layer_call_fn_969250

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
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9684362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_dense_61_layer_call_fn_969483

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
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_61_layer_call_and_return_conditional_losses_9684972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
D__inference_dense_61_layer_call_and_return_conditional_losses_968497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
~
)__inference_dense_59_layer_call_fn_969443

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
:?????????	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_9683202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_968436

inputs
dense_56_968415
dense_56_968417
dense_57_968420
dense_57_968422
dense_58_968425
dense_58_968427
dense_59_968430
dense_59_968432
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_9682202
flatten_8/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_56_968415dense_56_968417*
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
GPU2*0,1J 8? *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_9682392"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_968420dense_57_968422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_9682662"
 dense_57/StatefulPartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_968425dense_58_968427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_9682932"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_968430dense_59_968432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_9683202"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_968220

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_16_layer_call_fn_968455
flatten_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_9684362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?	
?
D__inference_dense_56_layer_call_and_return_conditional_losses_968239

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
?
"__inference__traced_restore_969848
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate&
"assignvariableop_5_dense_56_kernel$
 assignvariableop_6_dense_56_bias&
"assignvariableop_7_dense_57_kernel$
 assignvariableop_8_dense_57_bias&
"assignvariableop_9_dense_58_kernel%
!assignvariableop_10_dense_58_bias'
#assignvariableop_11_dense_59_kernel%
!assignvariableop_12_dense_59_bias'
#assignvariableop_13_dense_60_kernel%
!assignvariableop_14_dense_60_bias'
#assignvariableop_15_dense_61_kernel%
!assignvariableop_16_dense_61_bias'
#assignvariableop_17_dense_62_kernel%
!assignvariableop_18_dense_62_bias
assignvariableop_19_total
assignvariableop_20_count.
*assignvariableop_21_adam_dense_56_kernel_m,
(assignvariableop_22_adam_dense_56_bias_m.
*assignvariableop_23_adam_dense_57_kernel_m,
(assignvariableop_24_adam_dense_57_bias_m.
*assignvariableop_25_adam_dense_58_kernel_m,
(assignvariableop_26_adam_dense_58_bias_m.
*assignvariableop_27_adam_dense_59_kernel_m,
(assignvariableop_28_adam_dense_59_bias_m.
*assignvariableop_29_adam_dense_60_kernel_m,
(assignvariableop_30_adam_dense_60_bias_m.
*assignvariableop_31_adam_dense_61_kernel_m,
(assignvariableop_32_adam_dense_61_bias_m.
*assignvariableop_33_adam_dense_62_kernel_m,
(assignvariableop_34_adam_dense_62_bias_m.
*assignvariableop_35_adam_dense_56_kernel_v,
(assignvariableop_36_adam_dense_56_bias_v.
*assignvariableop_37_adam_dense_57_kernel_v,
(assignvariableop_38_adam_dense_57_bias_v.
*assignvariableop_39_adam_dense_58_kernel_v,
(assignvariableop_40_adam_dense_58_bias_v.
*assignvariableop_41_adam_dense_59_kernel_v,
(assignvariableop_42_adam_dense_59_bias_v.
*assignvariableop_43_adam_dense_60_kernel_v,
(assignvariableop_44_adam_dense_60_bias_v.
*assignvariableop_45_adam_dense_61_kernel_v,
(assignvariableop_46_adam_dense_61_bias_v.
*assignvariableop_47_adam_dense_62_kernel_v,
(assignvariableop_48_adam_dense_62_bias_v
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
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
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_56_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_56_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_57_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_57_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_58_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_58_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_59_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_59_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_60_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_60_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_61_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_61_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_62_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_62_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_56_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_56_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_57_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_57_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_58_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_58_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_59_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_59_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_60_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_60_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_61_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_61_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_62_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_62_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_56_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_56_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_57_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_57_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_58_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_58_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_59_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_59_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_60_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_60_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_61_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_61_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_62_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_62_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49?	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_969358

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?'
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
?__call__
+?&call_and_return_all_conditional_losses"?$
_tf_keras_sequential?${"class_name": "Sequential", "name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_8_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 9, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_8_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 9, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
? 
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_60_input"}}, {"class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_8", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_60_input"}}, {"class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_8", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}}}
?
iter

beta_1

beta_2
	decay
learning_ratem? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?"
	optimizer
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13"
trackable_list_wrapper
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-layer_metrics
	variables
.metrics

/layers
0layer_regularization_losses
1non_trainable_variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
 bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

!kernel
"bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

#kernel
$bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

%kernel
&bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 9, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
X
0
 1
!2
"3
#4
$5
%6
&7"
trackable_list_wrapper
X
0
 1
!2
"3
#4
$5
%6
&7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Flayer_metrics
	variables
Gmetrics

Hlayers
Ilayer_regularization_losses
Jnon_trainable_variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

'kernel
(bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
?

)kernel
*bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

+kernel
,bias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_8", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[layer_metrics
	variables
\metrics

]layers
^layer_regularization_losses
_non_trainable_variables
trainable_variables
regularization_losses
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
#:!
??2dense_56/kernel
:?2dense_56/bias
": 	?d2dense_57/kernel
:d2dense_57/bias
!:dd2dense_58/kernel
:d2dense_58/bias
!:d	2dense_59/kernel
:	2dense_59/bias
!:	d2dense_60/kernel
:d2dense_60/bias
!:dd2dense_61/kernel
:d2dense_61/bias
": 	d?2dense_62/kernel
:?2dense_62/bias
 "
trackable_dict_wrapper
'
`0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
alayer_metrics
bmetrics
2	variables

clayers
dlayer_regularization_losses
enon_trainable_variables
3trainable_variables
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
flayer_metrics
gmetrics
6	variables

hlayers
ilayer_regularization_losses
jnon_trainable_variables
7trainable_variables
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
klayer_metrics
lmetrics
:	variables

mlayers
nlayer_regularization_losses
onon_trainable_variables
;trainable_variables
<regularization_losses
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
player_metrics
qmetrics
>	variables

rlayers
slayer_regularization_losses
tnon_trainable_variables
?trainable_variables
@regularization_losses
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
ulayer_metrics
vmetrics
B	variables

wlayers
xlayer_regularization_losses
ynon_trainable_variables
Ctrainable_variables
Dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
	0

1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
zlayer_metrics
{metrics
K	variables

|layers
}layer_regularization_losses
~non_trainable_variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
layer_metrics
?metrics
O	variables
?layers
 ?layer_regularization_losses
?non_trainable_variables
Ptrainable_variables
Qregularization_losses
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
?layer_metrics
?metrics
S	variables
?layers
 ?layer_regularization_losses
?non_trainable_variables
Ttrainable_variables
Uregularization_losses
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
?
?layer_metrics
?metrics
W	variables
?layers
 ?layer_regularization_losses
?non_trainable_variables
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_56/kernel/m
!:?2Adam/dense_56/bias/m
':%	?d2Adam/dense_57/kernel/m
 :d2Adam/dense_57/bias/m
&:$dd2Adam/dense_58/kernel/m
 :d2Adam/dense_58/bias/m
&:$d	2Adam/dense_59/kernel/m
 :	2Adam/dense_59/bias/m
&:$	d2Adam/dense_60/kernel/m
 :d2Adam/dense_60/bias/m
&:$dd2Adam/dense_61/kernel/m
 :d2Adam/dense_61/bias/m
':%	d?2Adam/dense_62/kernel/m
!:?2Adam/dense_62/bias/m
(:&
??2Adam/dense_56/kernel/v
!:?2Adam/dense_56/bias/v
':%	?d2Adam/dense_57/kernel/v
 :d2Adam/dense_57/bias/v
&:$dd2Adam/dense_58/kernel/v
 :d2Adam/dense_58/bias/v
&:$d	2Adam/dense_59/kernel/v
 :	2Adam/dense_59/bias/v
&:$	d2Adam/dense_60/kernel/v
 :d2Adam/dense_60/bias/v
&:$dd2Adam/dense_61/kernel/v
 :d2Adam/dense_61/bias/v
':%	d?2Adam/dense_62/kernel/v
!:?2Adam/dense_62/bias/v
?2?
.__inference_autoencoder_8_layer_call_fn_969140
.__inference_autoencoder_8_layer_call_fn_969107
.__inference_autoencoder_8_layer_call_fn_968903
.__inference_autoencoder_8_layer_call_fn_968870?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
!__inference__wrapped_model_968210?
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
annotations? **?'
%?"
input_1?????????
?2?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968802
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_969010
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_969074
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968768?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_sequential_16_layer_call_fn_969250
.__inference_sequential_16_layer_call_fn_968455
.__inference_sequential_16_layer_call_fn_969229
.__inference_sequential_16_layer_call_fn_968409?
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
I__inference_sequential_16_layer_call_and_return_conditional_losses_968337
I__inference_sequential_16_layer_call_and_return_conditional_losses_969174
I__inference_sequential_16_layer_call_and_return_conditional_losses_969208
I__inference_sequential_16_layer_call_and_return_conditional_losses_968362?
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
.__inference_sequential_17_layer_call_fn_969335
.__inference_sequential_17_layer_call_fn_969352
.__inference_sequential_17_layer_call_fn_968620
.__inference_sequential_17_layer_call_fn_968657?
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
I__inference_sequential_17_layer_call_and_return_conditional_losses_969284
I__inference_sequential_17_layer_call_and_return_conditional_losses_969318
I__inference_sequential_17_layer_call_and_return_conditional_losses_968582
I__inference_sequential_17_layer_call_and_return_conditional_losses_968562?
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
$__inference_signature_wrapper_968946input_1"?
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
*__inference_flatten_8_layer_call_fn_969363?
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_969358?
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
)__inference_dense_56_layer_call_fn_969383?
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
D__inference_dense_56_layer_call_and_return_conditional_losses_969374?
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
)__inference_dense_57_layer_call_fn_969403?
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
D__inference_dense_57_layer_call_and_return_conditional_losses_969394?
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
)__inference_dense_58_layer_call_fn_969423?
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
D__inference_dense_58_layer_call_and_return_conditional_losses_969414?
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
)__inference_dense_59_layer_call_fn_969443?
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
D__inference_dense_59_layer_call_and_return_conditional_losses_969434?
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
)__inference_dense_60_layer_call_fn_969463?
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
D__inference_dense_60_layer_call_and_return_conditional_losses_969454?
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
)__inference_dense_61_layer_call_fn_969483?
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
D__inference_dense_61_layer_call_and_return_conditional_losses_969474?
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
)__inference_dense_62_layer_call_fn_969503?
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
D__inference_dense_62_layer_call_and_return_conditional_losses_969494?
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
*__inference_reshape_8_layer_call_fn_969521?
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
E__inference_reshape_8_layer_call_and_return_conditional_losses_969516?
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
!__inference__wrapped_model_968210 !"#$%&'()*+,4?1
*?'
%?"
input_1?????????
? "7?4
2
output_1&?#
output_1??????????
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968768u !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p
? ")?&
?
0?????????
? ?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_968802u !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p 
? ")?&
?
0?????????
? ?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_969010o !"#$%&'()*+,2?/
(?%
?
x?????????
p
? ")?&
?
0?????????
? ?
I__inference_autoencoder_8_layer_call_and_return_conditional_losses_969074o !"#$%&'()*+,2?/
(?%
?
x?????????
p 
? ")?&
?
0?????????
? ?
.__inference_autoencoder_8_layer_call_fn_968870h !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p
? "???????????
.__inference_autoencoder_8_layer_call_fn_968903h !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p 
? "???????????
.__inference_autoencoder_8_layer_call_fn_969107b !"#$%&'()*+,2?/
(?%
?
x?????????
p
? "???????????
.__inference_autoencoder_8_layer_call_fn_969140b !"#$%&'()*+,2?/
(?%
?
x?????????
p 
? "???????????
D__inference_dense_56_layer_call_and_return_conditional_losses_969374^ 0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_56_layer_call_fn_969383Q 0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_57_layer_call_and_return_conditional_losses_969394]!"0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? }
)__inference_dense_57_layer_call_fn_969403P!"0?-
&?#
!?
inputs??????????
? "??????????d?
D__inference_dense_58_layer_call_and_return_conditional_losses_969414\#$/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? |
)__inference_dense_58_layer_call_fn_969423O#$/?,
%?"
 ?
inputs?????????d
? "??????????d?
D__inference_dense_59_layer_call_and_return_conditional_losses_969434\%&/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????	
? |
)__inference_dense_59_layer_call_fn_969443O%&/?,
%?"
 ?
inputs?????????d
? "??????????	?
D__inference_dense_60_layer_call_and_return_conditional_losses_969454\'(/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????d
? |
)__inference_dense_60_layer_call_fn_969463O'(/?,
%?"
 ?
inputs?????????	
? "??????????d?
D__inference_dense_61_layer_call_and_return_conditional_losses_969474\)*/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? |
)__inference_dense_61_layer_call_fn_969483O)*/?,
%?"
 ?
inputs?????????d
? "??????????d?
D__inference_dense_62_layer_call_and_return_conditional_losses_969494]+,/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? }
)__inference_dense_62_layer_call_fn_969503P+,/?,
%?"
 ?
inputs?????????d
? "????????????
E__inference_flatten_8_layer_call_and_return_conditional_losses_969358]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_flatten_8_layer_call_fn_969363P3?0
)?&
$?!
inputs?????????
? "????????????
E__inference_reshape_8_layer_call_and_return_conditional_losses_969516]0?-
&?#
!?
inputs??????????
? ")?&
?
0?????????
? ~
*__inference_reshape_8_layer_call_fn_969521P0?-
&?#
!?
inputs??????????
? "???????????
I__inference_sequential_16_layer_call_and_return_conditional_losses_968337w !"#$%&D?A
:?7
-?*
flatten_8_input?????????
p

 
? "%?"
?
0?????????	
? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_968362w !"#$%&D?A
:?7
-?*
flatten_8_input?????????
p 

 
? "%?"
?
0?????????	
? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_969174n !"#$%&;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????	
? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_969208n !"#$%&;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????	
? ?
.__inference_sequential_16_layer_call_fn_968409j !"#$%&D?A
:?7
-?*
flatten_8_input?????????
p

 
? "??????????	?
.__inference_sequential_16_layer_call_fn_968455j !"#$%&D?A
:?7
-?*
flatten_8_input?????????
p 

 
? "??????????	?
.__inference_sequential_16_layer_call_fn_969229a !"#$%&;?8
1?.
$?!
inputs?????????
p

 
? "??????????	?
.__inference_sequential_16_layer_call_fn_969250a !"#$%&;?8
1?.
$?!
inputs?????????
p 

 
? "??????????	?
I__inference_sequential_17_layer_call_and_return_conditional_losses_968562t'()*+,??<
5?2
(?%
dense_60_input?????????	
p

 
? ")?&
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_968582t'()*+,??<
5?2
(?%
dense_60_input?????????	
p 

 
? ")?&
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_969284l'()*+,7?4
-?*
 ?
inputs?????????	
p

 
? ")?&
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_969318l'()*+,7?4
-?*
 ?
inputs?????????	
p 

 
? ")?&
?
0?????????
? ?
.__inference_sequential_17_layer_call_fn_968620g'()*+,??<
5?2
(?%
dense_60_input?????????	
p

 
? "???????????
.__inference_sequential_17_layer_call_fn_968657g'()*+,??<
5?2
(?%
dense_60_input?????????	
p 

 
? "???????????
.__inference_sequential_17_layer_call_fn_969335_'()*+,7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
.__inference_sequential_17_layer_call_fn_969352_'()*+,7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
$__inference_signature_wrapper_968946? !"#$%&'()*+,??<
? 
5?2
0
input_1%?"
input_1?????????"7?4
2
output_1&?#
output_1?????????