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
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_63/kernel
u
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel* 
_output_shapes
:
??*
dtype0
s
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_63/bias
l
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
_output_shapes	
:?*
dtype0
{
dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d* 
shared_namedense_64/kernel
t
#dense_64/kernel/Read/ReadVariableOpReadVariableOpdense_64/kernel*
_output_shapes
:	?d*
dtype0
r
dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_64/bias
k
!dense_64/bias/Read/ReadVariableOpReadVariableOpdense_64/bias*
_output_shapes
:d*
dtype0
z
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_65/kernel
s
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes

:dd*
dtype0
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes
:d*
dtype0
z
dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
* 
shared_namedense_66/kernel
s
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel*
_output_shapes

:d
*
dtype0
r
dense_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_66/bias
k
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes
:
*
dtype0
z
dense_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d* 
shared_namedense_67/kernel
s
#dense_67/kernel/Read/ReadVariableOpReadVariableOpdense_67/kernel*
_output_shapes

:
d*
dtype0
r
dense_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_67/bias
k
!dense_67/bias/Read/ReadVariableOpReadVariableOpdense_67/bias*
_output_shapes
:d*
dtype0
z
dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_68/kernel
s
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes

:dd*
dtype0
r
dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_68/bias
k
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes
:d*
dtype0
{
dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?* 
shared_namedense_69/kernel
t
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel*
_output_shapes
:	d?*
dtype0
s
dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_69/bias
l
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
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
Adam/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_63/kernel/m
?
*Adam/dense_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_63/bias/m
z
(Adam/dense_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameAdam/dense_64/kernel/m
?
*Adam/dense_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/dense_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_64/bias/m
y
(Adam/dense_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_65/kernel/m
?
*Adam/dense_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_65/bias/m
y
(Adam/dense_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*'
shared_nameAdam/dense_66/kernel/m
?
*Adam/dense_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/m*
_output_shapes

:d
*
dtype0
?
Adam/dense_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_66/bias/m
y
(Adam/dense_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*'
shared_nameAdam/dense_67/kernel/m
?
*Adam/dense_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/m*
_output_shapes

:
d*
dtype0
?
Adam/dense_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_67/bias/m
y
(Adam/dense_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_68/kernel/m
?
*Adam/dense_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_68/bias/m
y
(Adam/dense_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*'
shared_nameAdam/dense_69/kernel/m
?
*Adam/dense_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/m*
_output_shapes
:	d?*
dtype0
?
Adam/dense_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_69/bias/m
z
(Adam/dense_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_63/kernel/v
?
*Adam/dense_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_63/bias/v
z
(Adam/dense_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameAdam/dense_64/kernel/v
?
*Adam/dense_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/dense_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_64/bias/v
y
(Adam/dense_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_65/kernel/v
?
*Adam/dense_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_65/bias/v
y
(Adam/dense_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*'
shared_nameAdam/dense_66/kernel/v
?
*Adam/dense_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/v*
_output_shapes

:d
*
dtype0
?
Adam/dense_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_66/bias/v
y
(Adam/dense_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*'
shared_nameAdam/dense_67/kernel/v
?
*Adam/dense_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/v*
_output_shapes

:
d*
dtype0
?
Adam/dense_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_67/bias/v
y
(Adam/dense_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameAdam/dense_68/kernel/v
?
*Adam/dense_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_68/bias/v
y
(Adam/dense_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*'
shared_nameAdam/dense_69/kernel/v
?
*Adam/dense_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/v*
_output_shapes
:	d?*
dtype0
?
Adam/dense_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_69/bias/v
z
(Adam/dense_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/v*
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
VARIABLE_VALUEdense_63/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_63/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_64/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_64/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_65/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_65/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_66/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_66/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_67/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_67/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_68/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_68/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_69/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_69/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_63/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_63/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_64/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_64/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_65/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_65/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_66/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_66/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_67/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_67/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_68/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_68/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_69/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_69/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_63/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_63/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_64/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_64/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_65/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_65/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_66/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_66/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_67/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_67/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_68/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_68/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_69/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_69/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/bias*
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
GPU2*0,1J 8? *.
f)R'
%__inference_signature_wrapper_1121684
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOp#dense_64/kernel/Read/ReadVariableOp!dense_64/bias/Read/ReadVariableOp#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOp#dense_66/kernel/Read/ReadVariableOp!dense_66/bias/Read/ReadVariableOp#dense_67/kernel/Read/ReadVariableOp!dense_67/bias/Read/ReadVariableOp#dense_68/kernel/Read/ReadVariableOp!dense_68/bias/Read/ReadVariableOp#dense_69/kernel/Read/ReadVariableOp!dense_69/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_63/kernel/m/Read/ReadVariableOp(Adam/dense_63/bias/m/Read/ReadVariableOp*Adam/dense_64/kernel/m/Read/ReadVariableOp(Adam/dense_64/bias/m/Read/ReadVariableOp*Adam/dense_65/kernel/m/Read/ReadVariableOp(Adam/dense_65/bias/m/Read/ReadVariableOp*Adam/dense_66/kernel/m/Read/ReadVariableOp(Adam/dense_66/bias/m/Read/ReadVariableOp*Adam/dense_67/kernel/m/Read/ReadVariableOp(Adam/dense_67/bias/m/Read/ReadVariableOp*Adam/dense_68/kernel/m/Read/ReadVariableOp(Adam/dense_68/bias/m/Read/ReadVariableOp*Adam/dense_69/kernel/m/Read/ReadVariableOp(Adam/dense_69/bias/m/Read/ReadVariableOp*Adam/dense_63/kernel/v/Read/ReadVariableOp(Adam/dense_63/bias/v/Read/ReadVariableOp*Adam/dense_64/kernel/v/Read/ReadVariableOp(Adam/dense_64/bias/v/Read/ReadVariableOp*Adam/dense_65/kernel/v/Read/ReadVariableOp(Adam/dense_65/bias/v/Read/ReadVariableOp*Adam/dense_66/kernel/v/Read/ReadVariableOp(Adam/dense_66/bias/v/Read/ReadVariableOp*Adam/dense_67/kernel/v/Read/ReadVariableOp(Adam/dense_67/bias/v/Read/ReadVariableOp*Adam/dense_68/kernel/v/Read/ReadVariableOp(Adam/dense_68/bias/v/Read/ReadVariableOp*Adam/dense_69/kernel/v/Read/ReadVariableOp(Adam/dense_69/bias/v/Read/ReadVariableOpConst*>
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
GPU2*0,1J 8? *)
f$R"
 __inference__traced_save_1122429
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/biastotalcountAdam/dense_63/kernel/mAdam/dense_63/bias/mAdam/dense_64/kernel/mAdam/dense_64/bias/mAdam/dense_65/kernel/mAdam/dense_65/bias/mAdam/dense_66/kernel/mAdam/dense_66/bias/mAdam/dense_67/kernel/mAdam/dense_67/bias/mAdam/dense_68/kernel/mAdam/dense_68/bias/mAdam/dense_69/kernel/mAdam/dense_69/bias/mAdam/dense_63/kernel/vAdam/dense_63/bias/vAdam/dense_64/kernel/vAdam/dense_64/bias/vAdam/dense_65/kernel/vAdam/dense_65/bias/vAdam/dense_66/kernel/vAdam/dense_66/bias/vAdam/dense_67/kernel/vAdam/dense_67/bias/vAdam/dense_68/kernel/vAdam/dense_68/bias/vAdam/dense_69/kernel/vAdam/dense_69/bias/v*=
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
GPU2*0,1J 8? *,
f'R%
#__inference__traced_restore_1122586??

?	
?
E__inference_dense_69_layer_call_and_return_conditional_losses_1121262

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

*__inference_dense_68_layer_call_fn_1122221

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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_11212352
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
E__inference_dense_68_layer_call_and_return_conditional_losses_1122212

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
?
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121128

inputs
dense_63_1121107
dense_63_1121109
dense_64_1121112
dense_64_1121114
dense_65_1121117
dense_65_1121119
dense_66_1121122
dense_66_1121124
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallinputs*
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
GPU2*0,1J 8? *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_11209582
flatten_9/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_63_1121107dense_63_1121109*
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
E__inference_dense_63_layer_call_and_return_conditional_losses_11209772"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_1121112dense_64_1121114*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_11210042"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_1121117dense_65_1121119*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_11210312"
 dense_65/StatefulPartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_1121122dense_66_1121124*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_11210582"
 dense_66/StatefulPartitionedCall?
IdentityIdentity)dense_66/StatefulPartitionedCall:output:0!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
/__inference_autoencoder_9_layer_call_fn_1121641
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
GPU2*0,1J 8? *S
fNRL
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_11215772
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
?
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121320
dense_67_input
dense_67_1121303
dense_67_1121305
dense_68_1121308
dense_68_1121310
dense_69_1121313
dense_69_1121315
identity?? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall? dense_69/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCalldense_67_inputdense_67_1121303dense_67_1121305*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_11212082"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_1121308dense_68_1121310*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_11212352"
 dense_68/StatefulPartitionedCall?
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_1121313dense_69_1121315*
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
E__inference_dense_69_layer_call_and_return_conditional_losses_11212622"
 dense_69/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall)dense_69/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *O
fJRH
F__inference_reshape_9_layer_call_and_return_conditional_losses_11212912
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:W S
'
_output_shapes
:?????????

(
_user_specified_namedense_67_input
?
b
F__inference_reshape_9_layer_call_and_return_conditional_losses_1121291

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
?

?
/__inference_autoencoder_9_layer_call_fn_1121608
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
GPU2*0,1J 8? *S
fNRL
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_11215772
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
?
?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121540
input_1
sequential_18_1121509
sequential_18_1121511
sequential_18_1121513
sequential_18_1121515
sequential_18_1121517
sequential_18_1121519
sequential_18_1121521
sequential_18_1121523
sequential_19_1121526
sequential_19_1121528
sequential_19_1121530
sequential_19_1121532
sequential_19_1121534
sequential_19_1121536
identity??%sequential_18/StatefulPartitionedCall?%sequential_19/StatefulPartitionedCall?
%sequential_18/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_18_1121509sequential_18_1121511sequential_18_1121513sequential_18_1121515sequential_18_1121517sequential_18_1121519sequential_18_1121521sequential_18_1121523*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211742'
%sequential_18/StatefulPartitionedCall?
%sequential_19/StatefulPartitionedCallStatefulPartitionedCall.sequential_18/StatefulPartitionedCall:output:0sequential_19_1121526sequential_19_1121528sequential_19_1121530sequential_19_1121532sequential_19_1121534sequential_19_1121536*
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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213802'
%sequential_19/StatefulPartitionedCall?
IdentityIdentity.sequential_19/StatefulPartitionedCall:output:0&^sequential_18/StatefulPartitionedCall&^sequential_19/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_18/StatefulPartitionedCall%sequential_18/StatefulPartitionedCall2N
%sequential_19/StatefulPartitionedCall%sequential_19/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_dense_69_layer_call_and_return_conditional_losses_1122232

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
?_
?
 __inference__traced_save_1122429
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop.
*savev2_dense_64_kernel_read_readvariableop,
(savev2_dense_64_bias_read_readvariableop.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop.
*savev2_dense_66_kernel_read_readvariableop,
(savev2_dense_66_bias_read_readvariableop.
*savev2_dense_67_kernel_read_readvariableop,
(savev2_dense_67_bias_read_readvariableop.
*savev2_dense_68_kernel_read_readvariableop,
(savev2_dense_68_bias_read_readvariableop.
*savev2_dense_69_kernel_read_readvariableop,
(savev2_dense_69_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_63_kernel_m_read_readvariableop3
/savev2_adam_dense_63_bias_m_read_readvariableop5
1savev2_adam_dense_64_kernel_m_read_readvariableop3
/savev2_adam_dense_64_bias_m_read_readvariableop5
1savev2_adam_dense_65_kernel_m_read_readvariableop3
/savev2_adam_dense_65_bias_m_read_readvariableop5
1savev2_adam_dense_66_kernel_m_read_readvariableop3
/savev2_adam_dense_66_bias_m_read_readvariableop5
1savev2_adam_dense_67_kernel_m_read_readvariableop3
/savev2_adam_dense_67_bias_m_read_readvariableop5
1savev2_adam_dense_68_kernel_m_read_readvariableop3
/savev2_adam_dense_68_bias_m_read_readvariableop5
1savev2_adam_dense_69_kernel_m_read_readvariableop3
/savev2_adam_dense_69_bias_m_read_readvariableop5
1savev2_adam_dense_63_kernel_v_read_readvariableop3
/savev2_adam_dense_63_bias_v_read_readvariableop5
1savev2_adam_dense_64_kernel_v_read_readvariableop3
/savev2_adam_dense_64_bias_v_read_readvariableop5
1savev2_adam_dense_65_kernel_v_read_readvariableop3
/savev2_adam_dense_65_bias_v_read_readvariableop5
1savev2_adam_dense_66_kernel_v_read_readvariableop3
/savev2_adam_dense_66_bias_v_read_readvariableop5
1savev2_adam_dense_67_kernel_v_read_readvariableop3
/savev2_adam_dense_67_bias_v_read_readvariableop5
1savev2_adam_dense_68_kernel_v_read_readvariableop3
/savev2_adam_dense_68_bias_v_read_readvariableop5
1savev2_adam_dense_69_kernel_v_read_readvariableop3
/savev2_adam_dense_69_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop*savev2_dense_64_kernel_read_readvariableop(savev2_dense_64_bias_read_readvariableop*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableop*savev2_dense_66_kernel_read_readvariableop(savev2_dense_66_bias_read_readvariableop*savev2_dense_67_kernel_read_readvariableop(savev2_dense_67_bias_read_readvariableop*savev2_dense_68_kernel_read_readvariableop(savev2_dense_68_bias_read_readvariableop*savev2_dense_69_kernel_read_readvariableop(savev2_dense_69_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_63_kernel_m_read_readvariableop/savev2_adam_dense_63_bias_m_read_readvariableop1savev2_adam_dense_64_kernel_m_read_readvariableop/savev2_adam_dense_64_bias_m_read_readvariableop1savev2_adam_dense_65_kernel_m_read_readvariableop/savev2_adam_dense_65_bias_m_read_readvariableop1savev2_adam_dense_66_kernel_m_read_readvariableop/savev2_adam_dense_66_bias_m_read_readvariableop1savev2_adam_dense_67_kernel_m_read_readvariableop/savev2_adam_dense_67_bias_m_read_readvariableop1savev2_adam_dense_68_kernel_m_read_readvariableop/savev2_adam_dense_68_bias_m_read_readvariableop1savev2_adam_dense_69_kernel_m_read_readvariableop/savev2_adam_dense_69_bias_m_read_readvariableop1savev2_adam_dense_63_kernel_v_read_readvariableop/savev2_adam_dense_63_bias_v_read_readvariableop1savev2_adam_dense_64_kernel_v_read_readvariableop/savev2_adam_dense_64_bias_v_read_readvariableop1savev2_adam_dense_65_kernel_v_read_readvariableop/savev2_adam_dense_65_bias_v_read_readvariableop1savev2_adam_dense_66_kernel_v_read_readvariableop/savev2_adam_dense_66_bias_v_read_readvariableop1savev2_adam_dense_67_kernel_v_read_readvariableop/savev2_adam_dense_67_bias_v_read_readvariableop1savev2_adam_dense_68_kernel_v_read_readvariableop/savev2_adam_dense_68_bias_v_read_readvariableop1savev2_adam_dense_69_kernel_v_read_readvariableop/savev2_adam_dense_69_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
??:?:	?d:d:dd:d:d
:
:
d:d:dd:d:	d?:?: : :
??:?:	?d:d:dd:d:d
:
:
d:d:dd:d:	d?:?:
??:?:	?d:d:dd:d:d
:
:
d:d:dd:d:	d?:?: 2(
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

:d
: 

_output_shapes
:
:$ 

_output_shapes

:
d: 
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

:d
: 

_output_shapes
:
:$ 

_output_shapes

:
d: 
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

:d
: +

_output_shapes
:
:$, 

_output_shapes

:
d: -
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
?

*__inference_dense_65_layer_call_fn_1122161

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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_11210312
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
?

*__inference_dense_63_layer_call_fn_1122121

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
E__inference_dense_63_layer_call_and_return_conditional_losses_11209772
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
?
?
/__inference_sequential_18_layer_call_fn_1121193
flatten_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_9_input
?
G
+__inference_flatten_9_layer_call_fn_1122101

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
GPU2*0,1J 8? *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_11209582
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
?

*__inference_dense_66_layer_call_fn_1122181

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
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_11210582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?)
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121946

inputs+
'dense_63_matmul_readvariableop_resource,
(dense_63_biasadd_readvariableop_resource+
'dense_64_matmul_readvariableop_resource,
(dense_64_biasadd_readvariableop_resource+
'dense_65_matmul_readvariableop_resource,
(dense_65_biasadd_readvariableop_resource+
'dense_66_matmul_readvariableop_resource,
(dense_66_biasadd_readvariableop_resource
identity??dense_63/BiasAdd/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?dense_65/MatMul/ReadVariableOp?dense_66/BiasAdd/ReadVariableOp?dense_66/MatMul/ReadVariableOps
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_9/Const?
flatten_9/ReshapeReshapeinputsflatten_9/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_9/Reshape?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulflatten_9/Reshape:output:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_63/MatMul?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_63/BiasAdd/ReadVariableOp?
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_63/BiasAddt
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_63/Relu?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMuldense_63/Relu:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_64/BiasAdds
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_64/Relu?
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_65/MatMul/ReadVariableOp?
dense_65/MatMulMatMuldense_64/Relu:activations:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_65/MatMul?
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_65/BiasAdd/ReadVariableOp?
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_65/BiasAdds
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_65/Relu?
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02 
dense_66/MatMul/ReadVariableOp?
dense_66/MatMulMatMuldense_65/Relu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_66/MatMul?
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_66/BiasAdd/ReadVariableOp?
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_66/BiasAdd
dense_66/SoftsignSoftsigndense_66/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_66/Softsign?
IdentityIdentitydense_66/Softsign:activations:0 ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_64_layer_call_and_return_conditional_losses_1121004

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
?
?
/__inference_sequential_19_layer_call_fn_1121395
dense_67_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_67_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????

(
_user_specified_namedense_67_input
?
?
/__inference_sequential_19_layer_call_fn_1122090

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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
E__inference_dense_65_layer_call_and_return_conditional_losses_1122152

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
?	
?
E__inference_dense_63_layer_call_and_return_conditional_losses_1120977

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
E__inference_dense_68_layer_call_and_return_conditional_losses_1121235

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
G
+__inference_reshape_9_layer_call_fn_1122259

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
GPU2*0,1J 8? *O
fJRH
F__inference_reshape_9_layer_call_and_return_conditional_losses_11212912
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
?

*__inference_dense_64_layer_call_fn_1122141

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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_11210042
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
??
?
#__inference__traced_restore_1122586
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate&
"assignvariableop_5_dense_63_kernel$
 assignvariableop_6_dense_63_bias&
"assignvariableop_7_dense_64_kernel$
 assignvariableop_8_dense_64_bias&
"assignvariableop_9_dense_65_kernel%
!assignvariableop_10_dense_65_bias'
#assignvariableop_11_dense_66_kernel%
!assignvariableop_12_dense_66_bias'
#assignvariableop_13_dense_67_kernel%
!assignvariableop_14_dense_67_bias'
#assignvariableop_15_dense_68_kernel%
!assignvariableop_16_dense_68_bias'
#assignvariableop_17_dense_69_kernel%
!assignvariableop_18_dense_69_bias
assignvariableop_19_total
assignvariableop_20_count.
*assignvariableop_21_adam_dense_63_kernel_m,
(assignvariableop_22_adam_dense_63_bias_m.
*assignvariableop_23_adam_dense_64_kernel_m,
(assignvariableop_24_adam_dense_64_bias_m.
*assignvariableop_25_adam_dense_65_kernel_m,
(assignvariableop_26_adam_dense_65_bias_m.
*assignvariableop_27_adam_dense_66_kernel_m,
(assignvariableop_28_adam_dense_66_bias_m.
*assignvariableop_29_adam_dense_67_kernel_m,
(assignvariableop_30_adam_dense_67_bias_m.
*assignvariableop_31_adam_dense_68_kernel_m,
(assignvariableop_32_adam_dense_68_bias_m.
*assignvariableop_33_adam_dense_69_kernel_m,
(assignvariableop_34_adam_dense_69_bias_m.
*assignvariableop_35_adam_dense_63_kernel_v,
(assignvariableop_36_adam_dense_63_bias_v.
*assignvariableop_37_adam_dense_64_kernel_v,
(assignvariableop_38_adam_dense_64_bias_v.
*assignvariableop_39_adam_dense_65_kernel_v,
(assignvariableop_40_adam_dense_65_bias_v.
*assignvariableop_41_adam_dense_66_kernel_v,
(assignvariableop_42_adam_dense_66_bias_v.
*assignvariableop_43_adam_dense_67_kernel_v,
(assignvariableop_44_adam_dense_67_bias_v.
*assignvariableop_45_adam_dense_68_kernel_v,
(assignvariableop_46_adam_dense_68_bias_v.
*assignvariableop_47_adam_dense_69_kernel_v,
(assignvariableop_48_adam_dense_69_bias_v
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
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_63_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_63_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_64_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_64_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_65_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_65_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_66_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_66_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_67_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_67_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_68_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_68_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_69_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_69_biasIdentity_18:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_63_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_63_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_64_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_64_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_65_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_65_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_66_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_66_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_67_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_67_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_68_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_68_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_69_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_69_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_63_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_63_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_64_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_64_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_65_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_65_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_66_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_66_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_67_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_67_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_68_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_68_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_69_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_69_bias_vIdentity_48:output:0"/device:CPU:0*
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
?	
?
E__inference_dense_67_layer_call_and_return_conditional_losses_1122192

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
d*
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
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
/__inference_sequential_19_layer_call_fn_1121358
dense_67_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_67_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????

(
_user_specified_namedense_67_input
?h
?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121748
x9
5sequential_18_dense_63_matmul_readvariableop_resource:
6sequential_18_dense_63_biasadd_readvariableop_resource9
5sequential_18_dense_64_matmul_readvariableop_resource:
6sequential_18_dense_64_biasadd_readvariableop_resource9
5sequential_18_dense_65_matmul_readvariableop_resource:
6sequential_18_dense_65_biasadd_readvariableop_resource9
5sequential_18_dense_66_matmul_readvariableop_resource:
6sequential_18_dense_66_biasadd_readvariableop_resource9
5sequential_19_dense_67_matmul_readvariableop_resource:
6sequential_19_dense_67_biasadd_readvariableop_resource9
5sequential_19_dense_68_matmul_readvariableop_resource:
6sequential_19_dense_68_biasadd_readvariableop_resource9
5sequential_19_dense_69_matmul_readvariableop_resource:
6sequential_19_dense_69_biasadd_readvariableop_resource
identity??-sequential_18/dense_63/BiasAdd/ReadVariableOp?,sequential_18/dense_63/MatMul/ReadVariableOp?-sequential_18/dense_64/BiasAdd/ReadVariableOp?,sequential_18/dense_64/MatMul/ReadVariableOp?-sequential_18/dense_65/BiasAdd/ReadVariableOp?,sequential_18/dense_65/MatMul/ReadVariableOp?-sequential_18/dense_66/BiasAdd/ReadVariableOp?,sequential_18/dense_66/MatMul/ReadVariableOp?-sequential_19/dense_67/BiasAdd/ReadVariableOp?,sequential_19/dense_67/MatMul/ReadVariableOp?-sequential_19/dense_68/BiasAdd/ReadVariableOp?,sequential_19/dense_68/MatMul/ReadVariableOp?-sequential_19/dense_69/BiasAdd/ReadVariableOp?,sequential_19/dense_69/MatMul/ReadVariableOp?
sequential_18/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
sequential_18/flatten_9/Const?
sequential_18/flatten_9/ReshapeReshapex&sequential_18/flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????2!
sequential_18/flatten_9/Reshape?
,sequential_18/dense_63/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_63_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_18/dense_63/MatMul/ReadVariableOp?
sequential_18/dense_63/MatMulMatMul(sequential_18/flatten_9/Reshape:output:04sequential_18/dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_18/dense_63/MatMul?
-sequential_18/dense_63/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_18/dense_63/BiasAdd/ReadVariableOp?
sequential_18/dense_63/BiasAddBiasAdd'sequential_18/dense_63/MatMul:product:05sequential_18/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_18/dense_63/BiasAdd?
sequential_18/dense_63/ReluRelu'sequential_18/dense_63/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_18/dense_63/Relu?
,sequential_18/dense_64/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_64_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02.
,sequential_18/dense_64/MatMul/ReadVariableOp?
sequential_18/dense_64/MatMulMatMul)sequential_18/dense_63/Relu:activations:04sequential_18/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_64/MatMul?
-sequential_18/dense_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_64_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_18/dense_64/BiasAdd/ReadVariableOp?
sequential_18/dense_64/BiasAddBiasAdd'sequential_18/dense_64/MatMul:product:05sequential_18/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_18/dense_64/BiasAdd?
sequential_18/dense_64/ReluRelu'sequential_18/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_64/Relu?
,sequential_18/dense_65/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_65_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_18/dense_65/MatMul/ReadVariableOp?
sequential_18/dense_65/MatMulMatMul)sequential_18/dense_64/Relu:activations:04sequential_18/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_65/MatMul?
-sequential_18/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_65_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_18/dense_65/BiasAdd/ReadVariableOp?
sequential_18/dense_65/BiasAddBiasAdd'sequential_18/dense_65/MatMul:product:05sequential_18/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_18/dense_65/BiasAdd?
sequential_18/dense_65/ReluRelu'sequential_18/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_65/Relu?
,sequential_18/dense_66/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_66_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02.
,sequential_18/dense_66/MatMul/ReadVariableOp?
sequential_18/dense_66/MatMulMatMul)sequential_18/dense_65/Relu:activations:04sequential_18/dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_18/dense_66/MatMul?
-sequential_18/dense_66/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_66_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_18/dense_66/BiasAdd/ReadVariableOp?
sequential_18/dense_66/BiasAddBiasAdd'sequential_18/dense_66/MatMul:product:05sequential_18/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential_18/dense_66/BiasAdd?
sequential_18/dense_66/SoftsignSoftsign'sequential_18/dense_66/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2!
sequential_18/dense_66/Softsign?
,sequential_19/dense_67/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_67_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02.
,sequential_19/dense_67/MatMul/ReadVariableOp?
sequential_19/dense_67/MatMulMatMul-sequential_18/dense_66/Softsign:activations:04sequential_19/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_67/MatMul?
-sequential_19/dense_67/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_67_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_19/dense_67/BiasAdd/ReadVariableOp?
sequential_19/dense_67/BiasAddBiasAdd'sequential_19/dense_67/MatMul:product:05sequential_19/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_19/dense_67/BiasAdd?
sequential_19/dense_67/ReluRelu'sequential_19/dense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_67/Relu?
,sequential_19/dense_68/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_68_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_19/dense_68/MatMul/ReadVariableOp?
sequential_19/dense_68/MatMulMatMul)sequential_19/dense_67/Relu:activations:04sequential_19/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_68/MatMul?
-sequential_19/dense_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_68_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_19/dense_68/BiasAdd/ReadVariableOp?
sequential_19/dense_68/BiasAddBiasAdd'sequential_19/dense_68/MatMul:product:05sequential_19/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_19/dense_68/BiasAdd?
sequential_19/dense_68/ReluRelu'sequential_19/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_68/Relu?
,sequential_19/dense_69/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_69_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02.
,sequential_19/dense_69/MatMul/ReadVariableOp?
sequential_19/dense_69/MatMulMatMul)sequential_19/dense_68/Relu:activations:04sequential_19/dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_19/dense_69/MatMul?
-sequential_19/dense_69/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_69_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_19/dense_69/BiasAdd/ReadVariableOp?
sequential_19/dense_69/BiasAddBiasAdd'sequential_19/dense_69/MatMul:product:05sequential_19/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_19/dense_69/BiasAdd?
sequential_19/dense_69/SigmoidSigmoid'sequential_19/dense_69/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_19/dense_69/Sigmoid?
sequential_19/reshape_9/ShapeShape"sequential_19/dense_69/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_19/reshape_9/Shape?
+sequential_19/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_19/reshape_9/strided_slice/stack?
-sequential_19/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_19/reshape_9/strided_slice/stack_1?
-sequential_19/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_19/reshape_9/strided_slice/stack_2?
%sequential_19/reshape_9/strided_sliceStridedSlice&sequential_19/reshape_9/Shape:output:04sequential_19/reshape_9/strided_slice/stack:output:06sequential_19/reshape_9/strided_slice/stack_1:output:06sequential_19/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_19/reshape_9/strided_slice?
'sequential_19/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/reshape_9/Reshape/shape/1?
'sequential_19/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/reshape_9/Reshape/shape/2?
%sequential_19/reshape_9/Reshape/shapePack.sequential_19/reshape_9/strided_slice:output:00sequential_19/reshape_9/Reshape/shape/1:output:00sequential_19/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_19/reshape_9/Reshape/shape?
sequential_19/reshape_9/ReshapeReshape"sequential_19/dense_69/Sigmoid:y:0.sequential_19/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_19/reshape_9/Reshape?
IdentityIdentity(sequential_19/reshape_9/Reshape:output:0.^sequential_18/dense_63/BiasAdd/ReadVariableOp-^sequential_18/dense_63/MatMul/ReadVariableOp.^sequential_18/dense_64/BiasAdd/ReadVariableOp-^sequential_18/dense_64/MatMul/ReadVariableOp.^sequential_18/dense_65/BiasAdd/ReadVariableOp-^sequential_18/dense_65/MatMul/ReadVariableOp.^sequential_18/dense_66/BiasAdd/ReadVariableOp-^sequential_18/dense_66/MatMul/ReadVariableOp.^sequential_19/dense_67/BiasAdd/ReadVariableOp-^sequential_19/dense_67/MatMul/ReadVariableOp.^sequential_19/dense_68/BiasAdd/ReadVariableOp-^sequential_19/dense_68/MatMul/ReadVariableOp.^sequential_19/dense_69/BiasAdd/ReadVariableOp-^sequential_19/dense_69/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2^
-sequential_18/dense_63/BiasAdd/ReadVariableOp-sequential_18/dense_63/BiasAdd/ReadVariableOp2\
,sequential_18/dense_63/MatMul/ReadVariableOp,sequential_18/dense_63/MatMul/ReadVariableOp2^
-sequential_18/dense_64/BiasAdd/ReadVariableOp-sequential_18/dense_64/BiasAdd/ReadVariableOp2\
,sequential_18/dense_64/MatMul/ReadVariableOp,sequential_18/dense_64/MatMul/ReadVariableOp2^
-sequential_18/dense_65/BiasAdd/ReadVariableOp-sequential_18/dense_65/BiasAdd/ReadVariableOp2\
,sequential_18/dense_65/MatMul/ReadVariableOp,sequential_18/dense_65/MatMul/ReadVariableOp2^
-sequential_18/dense_66/BiasAdd/ReadVariableOp-sequential_18/dense_66/BiasAdd/ReadVariableOp2\
,sequential_18/dense_66/MatMul/ReadVariableOp,sequential_18/dense_66/MatMul/ReadVariableOp2^
-sequential_19/dense_67/BiasAdd/ReadVariableOp-sequential_19/dense_67/BiasAdd/ReadVariableOp2\
,sequential_19/dense_67/MatMul/ReadVariableOp,sequential_19/dense_67/MatMul/ReadVariableOp2^
-sequential_19/dense_68/BiasAdd/ReadVariableOp-sequential_19/dense_68/BiasAdd/ReadVariableOp2\
,sequential_19/dense_68/MatMul/ReadVariableOp,sequential_19/dense_68/MatMul/ReadVariableOp2^
-sequential_19/dense_69/BiasAdd/ReadVariableOp-sequential_19/dense_69/BiasAdd/ReadVariableOp2\
,sequential_19/dense_69/MatMul/ReadVariableOp,sequential_19/dense_69/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
/__inference_autoencoder_9_layer_call_fn_1121878
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
GPU2*0,1J 8? *S
fNRL
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_11215772
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
?h
?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121812
x9
5sequential_18_dense_63_matmul_readvariableop_resource:
6sequential_18_dense_63_biasadd_readvariableop_resource9
5sequential_18_dense_64_matmul_readvariableop_resource:
6sequential_18_dense_64_biasadd_readvariableop_resource9
5sequential_18_dense_65_matmul_readvariableop_resource:
6sequential_18_dense_65_biasadd_readvariableop_resource9
5sequential_18_dense_66_matmul_readvariableop_resource:
6sequential_18_dense_66_biasadd_readvariableop_resource9
5sequential_19_dense_67_matmul_readvariableop_resource:
6sequential_19_dense_67_biasadd_readvariableop_resource9
5sequential_19_dense_68_matmul_readvariableop_resource:
6sequential_19_dense_68_biasadd_readvariableop_resource9
5sequential_19_dense_69_matmul_readvariableop_resource:
6sequential_19_dense_69_biasadd_readvariableop_resource
identity??-sequential_18/dense_63/BiasAdd/ReadVariableOp?,sequential_18/dense_63/MatMul/ReadVariableOp?-sequential_18/dense_64/BiasAdd/ReadVariableOp?,sequential_18/dense_64/MatMul/ReadVariableOp?-sequential_18/dense_65/BiasAdd/ReadVariableOp?,sequential_18/dense_65/MatMul/ReadVariableOp?-sequential_18/dense_66/BiasAdd/ReadVariableOp?,sequential_18/dense_66/MatMul/ReadVariableOp?-sequential_19/dense_67/BiasAdd/ReadVariableOp?,sequential_19/dense_67/MatMul/ReadVariableOp?-sequential_19/dense_68/BiasAdd/ReadVariableOp?,sequential_19/dense_68/MatMul/ReadVariableOp?-sequential_19/dense_69/BiasAdd/ReadVariableOp?,sequential_19/dense_69/MatMul/ReadVariableOp?
sequential_18/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
sequential_18/flatten_9/Const?
sequential_18/flatten_9/ReshapeReshapex&sequential_18/flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????2!
sequential_18/flatten_9/Reshape?
,sequential_18/dense_63/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_63_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_18/dense_63/MatMul/ReadVariableOp?
sequential_18/dense_63/MatMulMatMul(sequential_18/flatten_9/Reshape:output:04sequential_18/dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_18/dense_63/MatMul?
-sequential_18/dense_63/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_18/dense_63/BiasAdd/ReadVariableOp?
sequential_18/dense_63/BiasAddBiasAdd'sequential_18/dense_63/MatMul:product:05sequential_18/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_18/dense_63/BiasAdd?
sequential_18/dense_63/ReluRelu'sequential_18/dense_63/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_18/dense_63/Relu?
,sequential_18/dense_64/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_64_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02.
,sequential_18/dense_64/MatMul/ReadVariableOp?
sequential_18/dense_64/MatMulMatMul)sequential_18/dense_63/Relu:activations:04sequential_18/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_64/MatMul?
-sequential_18/dense_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_64_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_18/dense_64/BiasAdd/ReadVariableOp?
sequential_18/dense_64/BiasAddBiasAdd'sequential_18/dense_64/MatMul:product:05sequential_18/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_18/dense_64/BiasAdd?
sequential_18/dense_64/ReluRelu'sequential_18/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_64/Relu?
,sequential_18/dense_65/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_65_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_18/dense_65/MatMul/ReadVariableOp?
sequential_18/dense_65/MatMulMatMul)sequential_18/dense_64/Relu:activations:04sequential_18/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_65/MatMul?
-sequential_18/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_65_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_18/dense_65/BiasAdd/ReadVariableOp?
sequential_18/dense_65/BiasAddBiasAdd'sequential_18/dense_65/MatMul:product:05sequential_18/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_18/dense_65/BiasAdd?
sequential_18/dense_65/ReluRelu'sequential_18/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_18/dense_65/Relu?
,sequential_18/dense_66/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_66_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02.
,sequential_18/dense_66/MatMul/ReadVariableOp?
sequential_18/dense_66/MatMulMatMul)sequential_18/dense_65/Relu:activations:04sequential_18/dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_18/dense_66/MatMul?
-sequential_18/dense_66/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_66_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_18/dense_66/BiasAdd/ReadVariableOp?
sequential_18/dense_66/BiasAddBiasAdd'sequential_18/dense_66/MatMul:product:05sequential_18/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential_18/dense_66/BiasAdd?
sequential_18/dense_66/SoftsignSoftsign'sequential_18/dense_66/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2!
sequential_18/dense_66/Softsign?
,sequential_19/dense_67/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_67_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02.
,sequential_19/dense_67/MatMul/ReadVariableOp?
sequential_19/dense_67/MatMulMatMul-sequential_18/dense_66/Softsign:activations:04sequential_19/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_67/MatMul?
-sequential_19/dense_67/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_67_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_19/dense_67/BiasAdd/ReadVariableOp?
sequential_19/dense_67/BiasAddBiasAdd'sequential_19/dense_67/MatMul:product:05sequential_19/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_19/dense_67/BiasAdd?
sequential_19/dense_67/ReluRelu'sequential_19/dense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_67/Relu?
,sequential_19/dense_68/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_68_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,sequential_19/dense_68/MatMul/ReadVariableOp?
sequential_19/dense_68/MatMulMatMul)sequential_19/dense_67/Relu:activations:04sequential_19/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_68/MatMul?
-sequential_19/dense_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_68_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_19/dense_68/BiasAdd/ReadVariableOp?
sequential_19/dense_68/BiasAddBiasAdd'sequential_19/dense_68/MatMul:product:05sequential_19/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_19/dense_68/BiasAdd?
sequential_19/dense_68/ReluRelu'sequential_19/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_19/dense_68/Relu?
,sequential_19/dense_69/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_69_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02.
,sequential_19/dense_69/MatMul/ReadVariableOp?
sequential_19/dense_69/MatMulMatMul)sequential_19/dense_68/Relu:activations:04sequential_19/dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_19/dense_69/MatMul?
-sequential_19/dense_69/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_69_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_19/dense_69/BiasAdd/ReadVariableOp?
sequential_19/dense_69/BiasAddBiasAdd'sequential_19/dense_69/MatMul:product:05sequential_19/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_19/dense_69/BiasAdd?
sequential_19/dense_69/SigmoidSigmoid'sequential_19/dense_69/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_19/dense_69/Sigmoid?
sequential_19/reshape_9/ShapeShape"sequential_19/dense_69/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_19/reshape_9/Shape?
+sequential_19/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_19/reshape_9/strided_slice/stack?
-sequential_19/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_19/reshape_9/strided_slice/stack_1?
-sequential_19/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_19/reshape_9/strided_slice/stack_2?
%sequential_19/reshape_9/strided_sliceStridedSlice&sequential_19/reshape_9/Shape:output:04sequential_19/reshape_9/strided_slice/stack:output:06sequential_19/reshape_9/strided_slice/stack_1:output:06sequential_19/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_19/reshape_9/strided_slice?
'sequential_19/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/reshape_9/Reshape/shape/1?
'sequential_19/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/reshape_9/Reshape/shape/2?
%sequential_19/reshape_9/Reshape/shapePack.sequential_19/reshape_9/strided_slice:output:00sequential_19/reshape_9/Reshape/shape/1:output:00sequential_19/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_19/reshape_9/Reshape/shape?
sequential_19/reshape_9/ReshapeReshape"sequential_19/dense_69/Sigmoid:y:0.sequential_19/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_19/reshape_9/Reshape?
IdentityIdentity(sequential_19/reshape_9/Reshape:output:0.^sequential_18/dense_63/BiasAdd/ReadVariableOp-^sequential_18/dense_63/MatMul/ReadVariableOp.^sequential_18/dense_64/BiasAdd/ReadVariableOp-^sequential_18/dense_64/MatMul/ReadVariableOp.^sequential_18/dense_65/BiasAdd/ReadVariableOp-^sequential_18/dense_65/MatMul/ReadVariableOp.^sequential_18/dense_66/BiasAdd/ReadVariableOp-^sequential_18/dense_66/MatMul/ReadVariableOp.^sequential_19/dense_67/BiasAdd/ReadVariableOp-^sequential_19/dense_67/MatMul/ReadVariableOp.^sequential_19/dense_68/BiasAdd/ReadVariableOp-^sequential_19/dense_68/MatMul/ReadVariableOp.^sequential_19/dense_69/BiasAdd/ReadVariableOp-^sequential_19/dense_69/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2^
-sequential_18/dense_63/BiasAdd/ReadVariableOp-sequential_18/dense_63/BiasAdd/ReadVariableOp2\
,sequential_18/dense_63/MatMul/ReadVariableOp,sequential_18/dense_63/MatMul/ReadVariableOp2^
-sequential_18/dense_64/BiasAdd/ReadVariableOp-sequential_18/dense_64/BiasAdd/ReadVariableOp2\
,sequential_18/dense_64/MatMul/ReadVariableOp,sequential_18/dense_64/MatMul/ReadVariableOp2^
-sequential_18/dense_65/BiasAdd/ReadVariableOp-sequential_18/dense_65/BiasAdd/ReadVariableOp2\
,sequential_18/dense_65/MatMul/ReadVariableOp,sequential_18/dense_65/MatMul/ReadVariableOp2^
-sequential_18/dense_66/BiasAdd/ReadVariableOp-sequential_18/dense_66/BiasAdd/ReadVariableOp2\
,sequential_18/dense_66/MatMul/ReadVariableOp,sequential_18/dense_66/MatMul/ReadVariableOp2^
-sequential_19/dense_67/BiasAdd/ReadVariableOp-sequential_19/dense_67/BiasAdd/ReadVariableOp2\
,sequential_19/dense_67/MatMul/ReadVariableOp,sequential_19/dense_67/MatMul/ReadVariableOp2^
-sequential_19/dense_68/BiasAdd/ReadVariableOp-sequential_19/dense_68/BiasAdd/ReadVariableOp2\
,sequential_19/dense_68/MatMul/ReadVariableOp,sequential_19/dense_68/MatMul/ReadVariableOp2^
-sequential_19/dense_69/BiasAdd/ReadVariableOp-sequential_19/dense_69/BiasAdd/ReadVariableOp2\
,sequential_19/dense_69/MatMul/ReadVariableOp,sequential_19/dense_69/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121075
flatten_9_input
dense_63_1120988
dense_63_1120990
dense_64_1121015
dense_64_1121017
dense_65_1121042
dense_65_1121044
dense_66_1121069
dense_66_1121071
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallflatten_9_input*
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
GPU2*0,1J 8? *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_11209582
flatten_9/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_63_1120988dense_63_1120990*
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
E__inference_dense_63_layer_call_and_return_conditional_losses_11209772"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_1121015dense_64_1121017*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_11210042"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_1121042dense_65_1121044*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_11210312"
 dense_65/StatefulPartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_1121069dense_66_1121071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_11210582"
 dense_66/StatefulPartitionedCall?
IdentityIdentity)dense_66/StatefulPartitionedCall:output:0!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_9_input
?	
?
E__inference_dense_64_layer_call_and_return_conditional_losses_1122132

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
E__inference_dense_66_layer_call_and_return_conditional_losses_1121058

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

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
??
?
"__inference__wrapped_model_1120948
input_1G
Cautoencoder_9_sequential_18_dense_63_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_18_dense_63_biasadd_readvariableop_resourceG
Cautoencoder_9_sequential_18_dense_64_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_18_dense_64_biasadd_readvariableop_resourceG
Cautoencoder_9_sequential_18_dense_65_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_18_dense_65_biasadd_readvariableop_resourceG
Cautoencoder_9_sequential_18_dense_66_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_18_dense_66_biasadd_readvariableop_resourceG
Cautoencoder_9_sequential_19_dense_67_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_19_dense_67_biasadd_readvariableop_resourceG
Cautoencoder_9_sequential_19_dense_68_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_19_dense_68_biasadd_readvariableop_resourceG
Cautoencoder_9_sequential_19_dense_69_matmul_readvariableop_resourceH
Dautoencoder_9_sequential_19_dense_69_biasadd_readvariableop_resource
identity??;autoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOp?;autoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOp?;autoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOp?;autoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOp?;autoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOp?;autoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOp?;autoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOp?:autoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOp?
+autoencoder_9/sequential_18/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2-
+autoencoder_9/sequential_18/flatten_9/Const?
-autoencoder_9/sequential_18/flatten_9/ReshapeReshapeinput_14autoencoder_9/sequential_18/flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????2/
-autoencoder_9/sequential_18/flatten_9/Reshape?
:autoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_18_dense_63_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:autoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOp?
+autoencoder_9/sequential_18/dense_63/MatMulMatMul6autoencoder_9/sequential_18/flatten_9/Reshape:output:0Bautoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_9/sequential_18/dense_63/MatMul?
;autoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_18_dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;autoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_18/dense_63/BiasAddBiasAdd5autoencoder_9/sequential_18/dense_63/MatMul:product:0Cautoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_9/sequential_18/dense_63/BiasAdd?
)autoencoder_9/sequential_18/dense_63/ReluRelu5autoencoder_9/sequential_18/dense_63/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2+
)autoencoder_9/sequential_18/dense_63/Relu?
:autoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_18_dense_64_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02<
:autoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOp?
+autoencoder_9/sequential_18/dense_64/MatMulMatMul7autoencoder_9/sequential_18/dense_63/Relu:activations:0Bautoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_9/sequential_18/dense_64/MatMul?
;autoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_18_dense_64_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_18/dense_64/BiasAddBiasAdd5autoencoder_9/sequential_18/dense_64/MatMul:product:0Cautoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_9/sequential_18/dense_64/BiasAdd?
)autoencoder_9/sequential_18/dense_64/ReluRelu5autoencoder_9/sequential_18/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_9/sequential_18/dense_64/Relu?
:autoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_18_dense_65_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02<
:autoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOp?
+autoencoder_9/sequential_18/dense_65/MatMulMatMul7autoencoder_9/sequential_18/dense_64/Relu:activations:0Bautoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_9/sequential_18/dense_65/MatMul?
;autoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_18_dense_65_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_18/dense_65/BiasAddBiasAdd5autoencoder_9/sequential_18/dense_65/MatMul:product:0Cautoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_9/sequential_18/dense_65/BiasAdd?
)autoencoder_9/sequential_18/dense_65/ReluRelu5autoencoder_9/sequential_18/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_9/sequential_18/dense_65/Relu?
:autoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_18_dense_66_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02<
:autoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOp?
+autoencoder_9/sequential_18/dense_66/MatMulMatMul7autoencoder_9/sequential_18/dense_65/Relu:activations:0Bautoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+autoencoder_9/sequential_18/dense_66/MatMul?
;autoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_18_dense_66_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;autoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_18/dense_66/BiasAddBiasAdd5autoencoder_9/sequential_18/dense_66/MatMul:product:0Cautoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,autoencoder_9/sequential_18/dense_66/BiasAdd?
-autoencoder_9/sequential_18/dense_66/SoftsignSoftsign5autoencoder_9/sequential_18/dense_66/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2/
-autoencoder_9/sequential_18/dense_66/Softsign?
:autoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_19_dense_67_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02<
:autoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOp?
+autoencoder_9/sequential_19/dense_67/MatMulMatMul;autoencoder_9/sequential_18/dense_66/Softsign:activations:0Bautoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_9/sequential_19/dense_67/MatMul?
;autoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_19_dense_67_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_19/dense_67/BiasAddBiasAdd5autoencoder_9/sequential_19/dense_67/MatMul:product:0Cautoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_9/sequential_19/dense_67/BiasAdd?
)autoencoder_9/sequential_19/dense_67/ReluRelu5autoencoder_9/sequential_19/dense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_9/sequential_19/dense_67/Relu?
:autoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_19_dense_68_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02<
:autoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOp?
+autoencoder_9/sequential_19/dense_68/MatMulMatMul7autoencoder_9/sequential_19/dense_67/Relu:activations:0Bautoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_9/sequential_19/dense_68/MatMul?
;autoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_19_dense_68_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;autoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_19/dense_68/BiasAddBiasAdd5autoencoder_9/sequential_19/dense_68/MatMul:product:0Cautoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,autoencoder_9/sequential_19/dense_68/BiasAdd?
)autoencoder_9/sequential_19/dense_68/ReluRelu5autoencoder_9/sequential_19/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2+
)autoencoder_9/sequential_19/dense_68/Relu?
:autoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOpReadVariableOpCautoencoder_9_sequential_19_dense_69_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02<
:autoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOp?
+autoencoder_9/sequential_19/dense_69/MatMulMatMul7autoencoder_9/sequential_19/dense_68/Relu:activations:0Bautoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_9/sequential_19/dense_69/MatMul?
;autoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_9_sequential_19_dense_69_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;autoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOp?
,autoencoder_9/sequential_19/dense_69/BiasAddBiasAdd5autoencoder_9/sequential_19/dense_69/MatMul:product:0Cautoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_9/sequential_19/dense_69/BiasAdd?
,autoencoder_9/sequential_19/dense_69/SigmoidSigmoid5autoencoder_9/sequential_19/dense_69/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_9/sequential_19/dense_69/Sigmoid?
+autoencoder_9/sequential_19/reshape_9/ShapeShape0autoencoder_9/sequential_19/dense_69/Sigmoid:y:0*
T0*
_output_shapes
:2-
+autoencoder_9/sequential_19/reshape_9/Shape?
9autoencoder_9/sequential_19/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9autoencoder_9/sequential_19/reshape_9/strided_slice/stack?
;autoencoder_9/sequential_19/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;autoencoder_9/sequential_19/reshape_9/strided_slice/stack_1?
;autoencoder_9/sequential_19/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;autoencoder_9/sequential_19/reshape_9/strided_slice/stack_2?
3autoencoder_9/sequential_19/reshape_9/strided_sliceStridedSlice4autoencoder_9/sequential_19/reshape_9/Shape:output:0Bautoencoder_9/sequential_19/reshape_9/strided_slice/stack:output:0Dautoencoder_9/sequential_19/reshape_9/strided_slice/stack_1:output:0Dautoencoder_9/sequential_19/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3autoencoder_9/sequential_19/reshape_9/strided_slice?
5autoencoder_9/sequential_19/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5autoencoder_9/sequential_19/reshape_9/Reshape/shape/1?
5autoencoder_9/sequential_19/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5autoencoder_9/sequential_19/reshape_9/Reshape/shape/2?
3autoencoder_9/sequential_19/reshape_9/Reshape/shapePack<autoencoder_9/sequential_19/reshape_9/strided_slice:output:0>autoencoder_9/sequential_19/reshape_9/Reshape/shape/1:output:0>autoencoder_9/sequential_19/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:25
3autoencoder_9/sequential_19/reshape_9/Reshape/shape?
-autoencoder_9/sequential_19/reshape_9/ReshapeReshape0autoencoder_9/sequential_19/dense_69/Sigmoid:y:0<autoencoder_9/sequential_19/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2/
-autoencoder_9/sequential_19/reshape_9/Reshape?
IdentityIdentity6autoencoder_9/sequential_19/reshape_9/Reshape:output:0<^autoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOp<^autoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOp<^autoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOp<^autoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOp<^autoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOp<^autoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOp<^autoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOp;^autoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2z
;autoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOp;autoencoder_9/sequential_18/dense_63/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOp:autoencoder_9/sequential_18/dense_63/MatMul/ReadVariableOp2z
;autoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOp;autoencoder_9/sequential_18/dense_64/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOp:autoencoder_9/sequential_18/dense_64/MatMul/ReadVariableOp2z
;autoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOp;autoencoder_9/sequential_18/dense_65/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOp:autoencoder_9/sequential_18/dense_65/MatMul/ReadVariableOp2z
;autoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOp;autoencoder_9/sequential_18/dense_66/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOp:autoencoder_9/sequential_18/dense_66/MatMul/ReadVariableOp2z
;autoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOp;autoencoder_9/sequential_19/dense_67/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOp:autoencoder_9/sequential_19/dense_67/MatMul/ReadVariableOp2z
;autoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOp;autoencoder_9/sequential_19/dense_68/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOp:autoencoder_9/sequential_19/dense_68/MatMul/ReadVariableOp2z
;autoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOp;autoencoder_9/sequential_19/dense_69/BiasAdd/ReadVariableOp2x
:autoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOp:autoencoder_9/sequential_19/dense_69/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?)
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121912

inputs+
'dense_63_matmul_readvariableop_resource,
(dense_63_biasadd_readvariableop_resource+
'dense_64_matmul_readvariableop_resource,
(dense_64_biasadd_readvariableop_resource+
'dense_65_matmul_readvariableop_resource,
(dense_65_biasadd_readvariableop_resource+
'dense_66_matmul_readvariableop_resource,
(dense_66_biasadd_readvariableop_resource
identity??dense_63/BiasAdd/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?dense_65/MatMul/ReadVariableOp?dense_66/BiasAdd/ReadVariableOp?dense_66/MatMul/ReadVariableOps
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_9/Const?
flatten_9/ReshapeReshapeinputsflatten_9/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_9/Reshape?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulflatten_9/Reshape:output:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_63/MatMul?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_63/BiasAdd/ReadVariableOp?
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_63/BiasAddt
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_63/Relu?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMuldense_63/Relu:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_64/BiasAdds
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_64/Relu?
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_65/MatMul/ReadVariableOp?
dense_65/MatMulMatMuldense_64/Relu:activations:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_65/MatMul?
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_65/BiasAdd/ReadVariableOp?
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_65/BiasAdds
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_65/Relu?
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02 
dense_66/MatMul/ReadVariableOp?
dense_66/MatMulMatMuldense_65/Relu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_66/MatMul?
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_66/BiasAdd/ReadVariableOp?
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_66/BiasAdd
dense_66/SoftsignSoftsigndense_66/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_66/Softsign?
IdentityIdentitydense_66/Softsign:activations:0 ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_66_layer_call_and_return_conditional_losses_1122172

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

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
?
?
/__inference_sequential_19_layer_call_fn_1122073

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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1120958

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
?
b
F__inference_reshape_9_layer_call_and_return_conditional_losses_1122254

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
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1122022

inputs+
'dense_67_matmul_readvariableop_resource,
(dense_67_biasadd_readvariableop_resource+
'dense_68_matmul_readvariableop_resource,
(dense_68_biasadd_readvariableop_resource+
'dense_69_matmul_readvariableop_resource,
(dense_69_biasadd_readvariableop_resource
identity??dense_67/BiasAdd/ReadVariableOp?dense_67/MatMul/ReadVariableOp?dense_68/BiasAdd/ReadVariableOp?dense_68/MatMul/ReadVariableOp?dense_69/BiasAdd/ReadVariableOp?dense_69/MatMul/ReadVariableOp?
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02 
dense_67/MatMul/ReadVariableOp?
dense_67/MatMulMatMulinputs&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_67/MatMul?
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_67/BiasAdd/ReadVariableOp?
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_67/BiasAdds
dense_67/ReluReludense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_67/Relu?
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_68/MatMul/ReadVariableOp?
dense_68/MatMulMatMuldense_67/Relu:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_68/MatMul?
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_68/BiasAdd/ReadVariableOp?
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_68/BiasAdds
dense_68/ReluReludense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_68/Relu?
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_69/MatMul/ReadVariableOp?
dense_69/MatMulMatMuldense_68/Relu:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_69/MatMul?
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_69/BiasAdd/ReadVariableOp?
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_69/BiasAdd}
dense_69/SigmoidSigmoiddense_69/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_69/Sigmoidf
reshape_9/ShapeShapedense_69/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_9/Shape?
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stack?
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1?
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2?
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2?
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shape?
reshape_9/ReshapeReshapedense_69/Sigmoid:y:0 reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_9/Reshape?
IdentityIdentityreshape_9/Reshape:output:0 ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121380

inputs
dense_67_1121363
dense_67_1121365
dense_68_1121368
dense_68_1121370
dense_69_1121373
dense_69_1121375
identity?? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall? dense_69/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCallinputsdense_67_1121363dense_67_1121365*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_11212082"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_1121368dense_68_1121370*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_11212352"
 dense_68/StatefulPartitionedCall?
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_1121373dense_69_1121375*
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
E__inference_dense_69_layer_call_and_return_conditional_losses_11212622"
 dense_69/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall)dense_69/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *O
fJRH
F__inference_reshape_9_layer_call_and_return_conditional_losses_11212912
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
E__inference_dense_63_layer_call_and_return_conditional_losses_1122112

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
?
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121343

inputs
dense_67_1121326
dense_67_1121328
dense_68_1121331
dense_68_1121333
dense_69_1121336
dense_69_1121338
identity?? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall? dense_69/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCallinputsdense_67_1121326dense_67_1121328*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_11212082"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_1121331dense_68_1121333*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_11212352"
 dense_68/StatefulPartitionedCall?
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_1121336dense_69_1121338*
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
E__inference_dense_69_layer_call_and_return_conditional_losses_11212622"
 dense_69/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall)dense_69/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *O
fJRH
F__inference_reshape_9_layer_call_and_return_conditional_losses_11212912
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121174

inputs
dense_63_1121153
dense_63_1121155
dense_64_1121158
dense_64_1121160
dense_65_1121163
dense_65_1121165
dense_66_1121168
dense_66_1121170
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallinputs*
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
GPU2*0,1J 8? *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_11209582
flatten_9/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_63_1121153dense_63_1121155*
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
E__inference_dense_63_layer_call_and_return_conditional_losses_11209772"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_1121158dense_64_1121160*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_11210042"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_1121163dense_65_1121165*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_11210312"
 dense_65/StatefulPartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_1121168dense_66_1121170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_11210582"
 dense_66/StatefulPartitionedCall?
IdentityIdentity)dense_66/StatefulPartitionedCall:output:0!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121506
input_1
sequential_18_1121441
sequential_18_1121443
sequential_18_1121445
sequential_18_1121447
sequential_18_1121449
sequential_18_1121451
sequential_18_1121453
sequential_18_1121455
sequential_19_1121492
sequential_19_1121494
sequential_19_1121496
sequential_19_1121498
sequential_19_1121500
sequential_19_1121502
identity??%sequential_18/StatefulPartitionedCall?%sequential_19/StatefulPartitionedCall?
%sequential_18/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_18_1121441sequential_18_1121443sequential_18_1121445sequential_18_1121447sequential_18_1121449sequential_18_1121451sequential_18_1121453sequential_18_1121455*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211282'
%sequential_18/StatefulPartitionedCall?
%sequential_19/StatefulPartitionedCallStatefulPartitionedCall.sequential_18/StatefulPartitionedCall:output:0sequential_19_1121492sequential_19_1121494sequential_19_1121496sequential_19_1121498sequential_19_1121500sequential_19_1121502*
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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213432'
%sequential_19/StatefulPartitionedCall?
IdentityIdentity.sequential_19/StatefulPartitionedCall:output:0&^sequential_18/StatefulPartitionedCall&^sequential_19/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_18/StatefulPartitionedCall%sequential_18/StatefulPartitionedCall2N
%sequential_19/StatefulPartitionedCall%sequential_19/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121100
flatten_9_input
dense_63_1121079
dense_63_1121081
dense_64_1121084
dense_64_1121086
dense_65_1121089
dense_65_1121091
dense_66_1121094
dense_66_1121096
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallflatten_9_input*
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
GPU2*0,1J 8? *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_11209582
flatten_9/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_63_1121079dense_63_1121081*
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
E__inference_dense_63_layer_call_and_return_conditional_losses_11209772"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_1121084dense_64_1121086*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_11210042"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_1121089dense_65_1121091*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_11210312"
 dense_65/StatefulPartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_1121094dense_66_1121096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_11210582"
 dense_66/StatefulPartitionedCall?
IdentityIdentity)dense_66/StatefulPartitionedCall:output:0!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_9_input
?
?
/__inference_sequential_18_layer_call_fn_1121988

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
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_autoencoder_9_layer_call_fn_1121845
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
GPU2*0,1J 8? *S
fNRL
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_11215772
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
?
?
/__inference_sequential_18_layer_call_fn_1121147
flatten_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_9_input
?	
?
E__inference_dense_65_layer_call_and_return_conditional_losses_1121031

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
?
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121300
dense_67_input
dense_67_1121219
dense_67_1121221
dense_68_1121246
dense_68_1121248
dense_69_1121273
dense_69_1121275
identity?? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall? dense_69/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCalldense_67_inputdense_67_1121219dense_67_1121221*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_11212082"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_1121246dense_68_1121248*
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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_11212352"
 dense_68/StatefulPartitionedCall?
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_1121273dense_69_1121275*
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
E__inference_dense_69_layer_call_and_return_conditional_losses_11212622"
 dense_69/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall)dense_69/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *O
fJRH
F__inference_reshape_9_layer_call_and_return_conditional_losses_11212912
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:W S
'
_output_shapes
:?????????

(
_user_specified_namedense_67_input
?)
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1122056

inputs+
'dense_67_matmul_readvariableop_resource,
(dense_67_biasadd_readvariableop_resource+
'dense_68_matmul_readvariableop_resource,
(dense_68_biasadd_readvariableop_resource+
'dense_69_matmul_readvariableop_resource,
(dense_69_biasadd_readvariableop_resource
identity??dense_67/BiasAdd/ReadVariableOp?dense_67/MatMul/ReadVariableOp?dense_68/BiasAdd/ReadVariableOp?dense_68/MatMul/ReadVariableOp?dense_69/BiasAdd/ReadVariableOp?dense_69/MatMul/ReadVariableOp?
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02 
dense_67/MatMul/ReadVariableOp?
dense_67/MatMulMatMulinputs&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_67/MatMul?
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_67/BiasAdd/ReadVariableOp?
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_67/BiasAdds
dense_67/ReluReludense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_67/Relu?
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_68/MatMul/ReadVariableOp?
dense_68/MatMulMatMuldense_67/Relu:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_68/MatMul?
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_68/BiasAdd/ReadVariableOp?
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_68/BiasAdds
dense_68/ReluReludense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_68/Relu?
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_69/MatMul/ReadVariableOp?
dense_69/MatMulMatMuldense_68/Relu:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_69/MatMul?
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_69/BiasAdd/ReadVariableOp?
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_69/BiasAdd}
dense_69/SigmoidSigmoiddense_69/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_69/Sigmoidf
reshape_9/ShapeShapedense_69/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_9/Shape?
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stack?
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1?
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2?
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2?
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shape?
reshape_9/ReshapeReshapedense_69/Sigmoid:y:0 reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_9/Reshape?
IdentityIdentityreshape_9/Reshape:output:0 ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1122096

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
?
?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121577
x
sequential_18_1121546
sequential_18_1121548
sequential_18_1121550
sequential_18_1121552
sequential_18_1121554
sequential_18_1121556
sequential_18_1121558
sequential_18_1121560
sequential_19_1121563
sequential_19_1121565
sequential_19_1121567
sequential_19_1121569
sequential_19_1121571
sequential_19_1121573
identity??%sequential_18/StatefulPartitionedCall?%sequential_19/StatefulPartitionedCall?
%sequential_18/StatefulPartitionedCallStatefulPartitionedCallxsequential_18_1121546sequential_18_1121548sequential_18_1121550sequential_18_1121552sequential_18_1121554sequential_18_1121556sequential_18_1121558sequential_18_1121560*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211742'
%sequential_18/StatefulPartitionedCall?
%sequential_19/StatefulPartitionedCallStatefulPartitionedCall.sequential_18/StatefulPartitionedCall:output:0sequential_19_1121563sequential_19_1121565sequential_19_1121567sequential_19_1121569sequential_19_1121571sequential_19_1121573*
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
GPU2*0,1J 8? *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_11213802'
%sequential_19/StatefulPartitionedCall?
IdentityIdentity.sequential_19/StatefulPartitionedCall:output:0&^sequential_18/StatefulPartitionedCall&^sequential_19/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_18/StatefulPartitionedCall%sequential_18/StatefulPartitionedCall2N
%sequential_19/StatefulPartitionedCall%sequential_19/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
%__inference_signature_wrapper_1121684
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
GPU2*0,1J 8? *+
f&R$
"__inference__wrapped_model_11209482
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
?

*__inference_dense_69_layer_call_fn_1122241

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
E__inference_dense_69_layer_call_and_return_conditional_losses_11212622
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
?

*__inference_dense_67_layer_call_fn_1122201

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
GPU2*0,1J 8? *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_11212082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
E__inference_dense_67_layer_call_and_return_conditional_losses_1121208

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
d*
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
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
/__inference_sequential_18_layer_call_fn_1121967

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
:?????????
**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_11211282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
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
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_sequential?${"class_name": "Sequential", "name": "sequential_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_9_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 10, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_9_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 10, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_67_input"}}, {"class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_69", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_9", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_67_input"}}, {"class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_69", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_9", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}}}
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
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
 bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

!kernel
"bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

#kernel
$bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

%kernel
&bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 10, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

)kernel
*bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

+kernel
,bias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_69", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_69", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_9", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}
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
??2dense_63/kernel
:?2dense_63/bias
": 	?d2dense_64/kernel
:d2dense_64/bias
!:dd2dense_65/kernel
:d2dense_65/bias
!:d
2dense_66/kernel
:
2dense_66/bias
!:
d2dense_67/kernel
:d2dense_67/bias
!:dd2dense_68/kernel
:d2dense_68/bias
": 	d?2dense_69/kernel
:?2dense_69/bias
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
??2Adam/dense_63/kernel/m
!:?2Adam/dense_63/bias/m
':%	?d2Adam/dense_64/kernel/m
 :d2Adam/dense_64/bias/m
&:$dd2Adam/dense_65/kernel/m
 :d2Adam/dense_65/bias/m
&:$d
2Adam/dense_66/kernel/m
 :
2Adam/dense_66/bias/m
&:$
d2Adam/dense_67/kernel/m
 :d2Adam/dense_67/bias/m
&:$dd2Adam/dense_68/kernel/m
 :d2Adam/dense_68/bias/m
':%	d?2Adam/dense_69/kernel/m
!:?2Adam/dense_69/bias/m
(:&
??2Adam/dense_63/kernel/v
!:?2Adam/dense_63/bias/v
':%	?d2Adam/dense_64/kernel/v
 :d2Adam/dense_64/bias/v
&:$dd2Adam/dense_65/kernel/v
 :d2Adam/dense_65/bias/v
&:$d
2Adam/dense_66/kernel/v
 :
2Adam/dense_66/bias/v
&:$
d2Adam/dense_67/kernel/v
 :d2Adam/dense_67/bias/v
&:$dd2Adam/dense_68/kernel/v
 :d2Adam/dense_68/bias/v
':%	d?2Adam/dense_69/kernel/v
!:?2Adam/dense_69/bias/v
?2?
/__inference_autoencoder_9_layer_call_fn_1121845
/__inference_autoencoder_9_layer_call_fn_1121878
/__inference_autoencoder_9_layer_call_fn_1121608
/__inference_autoencoder_9_layer_call_fn_1121641?
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
"__inference__wrapped_model_1120948?
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
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121540
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121812
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121748
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121506?
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
/__inference_sequential_18_layer_call_fn_1121988
/__inference_sequential_18_layer_call_fn_1121147
/__inference_sequential_18_layer_call_fn_1121193
/__inference_sequential_18_layer_call_fn_1121967?
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121075
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121100
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121946
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121912?
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
/__inference_sequential_19_layer_call_fn_1122073
/__inference_sequential_19_layer_call_fn_1121395
/__inference_sequential_19_layer_call_fn_1121358
/__inference_sequential_19_layer_call_fn_1122090?
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
J__inference_sequential_19_layer_call_and_return_conditional_losses_1122056
J__inference_sequential_19_layer_call_and_return_conditional_losses_1122022
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121320
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121300?
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
%__inference_signature_wrapper_1121684input_1"?
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
+__inference_flatten_9_layer_call_fn_1122101?
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
F__inference_flatten_9_layer_call_and_return_conditional_losses_1122096?
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
*__inference_dense_63_layer_call_fn_1122121?
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
E__inference_dense_63_layer_call_and_return_conditional_losses_1122112?
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
*__inference_dense_64_layer_call_fn_1122141?
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
E__inference_dense_64_layer_call_and_return_conditional_losses_1122132?
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
*__inference_dense_65_layer_call_fn_1122161?
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
E__inference_dense_65_layer_call_and_return_conditional_losses_1122152?
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
*__inference_dense_66_layer_call_fn_1122181?
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
E__inference_dense_66_layer_call_and_return_conditional_losses_1122172?
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
*__inference_dense_67_layer_call_fn_1122201?
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
E__inference_dense_67_layer_call_and_return_conditional_losses_1122192?
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
*__inference_dense_68_layer_call_fn_1122221?
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
E__inference_dense_68_layer_call_and_return_conditional_losses_1122212?
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
*__inference_dense_69_layer_call_fn_1122241?
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
E__inference_dense_69_layer_call_and_return_conditional_losses_1122232?
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
+__inference_reshape_9_layer_call_fn_1122259?
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
F__inference_reshape_9_layer_call_and_return_conditional_losses_1122254?
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
"__inference__wrapped_model_1120948 !"#$%&'()*+,4?1
*?'
%?"
input_1?????????
? "7?4
2
output_1&?#
output_1??????????
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121506u !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p
? ")?&
?
0?????????
? ?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121540u !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p 
? ")?&
?
0?????????
? ?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121748o !"#$%&'()*+,2?/
(?%
?
x?????????
p
? ")?&
?
0?????????
? ?
J__inference_autoencoder_9_layer_call_and_return_conditional_losses_1121812o !"#$%&'()*+,2?/
(?%
?
x?????????
p 
? ")?&
?
0?????????
? ?
/__inference_autoencoder_9_layer_call_fn_1121608h !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p
? "???????????
/__inference_autoencoder_9_layer_call_fn_1121641h !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p 
? "???????????
/__inference_autoencoder_9_layer_call_fn_1121845b !"#$%&'()*+,2?/
(?%
?
x?????????
p
? "???????????
/__inference_autoencoder_9_layer_call_fn_1121878b !"#$%&'()*+,2?/
(?%
?
x?????????
p 
? "???????????
E__inference_dense_63_layer_call_and_return_conditional_losses_1122112^ 0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_63_layer_call_fn_1122121Q 0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_64_layer_call_and_return_conditional_losses_1122132]!"0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ~
*__inference_dense_64_layer_call_fn_1122141P!"0?-
&?#
!?
inputs??????????
? "??????????d?
E__inference_dense_65_layer_call_and_return_conditional_losses_1122152\#$/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? }
*__inference_dense_65_layer_call_fn_1122161O#$/?,
%?"
 ?
inputs?????????d
? "??????????d?
E__inference_dense_66_layer_call_and_return_conditional_losses_1122172\%&/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? }
*__inference_dense_66_layer_call_fn_1122181O%&/?,
%?"
 ?
inputs?????????d
? "??????????
?
E__inference_dense_67_layer_call_and_return_conditional_losses_1122192\'(/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????d
? }
*__inference_dense_67_layer_call_fn_1122201O'(/?,
%?"
 ?
inputs?????????

? "??????????d?
E__inference_dense_68_layer_call_and_return_conditional_losses_1122212\)*/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? }
*__inference_dense_68_layer_call_fn_1122221O)*/?,
%?"
 ?
inputs?????????d
? "??????????d?
E__inference_dense_69_layer_call_and_return_conditional_losses_1122232]+,/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? ~
*__inference_dense_69_layer_call_fn_1122241P+,/?,
%?"
 ?
inputs?????????d
? "????????????
F__inference_flatten_9_layer_call_and_return_conditional_losses_1122096]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? 
+__inference_flatten_9_layer_call_fn_1122101P3?0
)?&
$?!
inputs?????????
? "????????????
F__inference_reshape_9_layer_call_and_return_conditional_losses_1122254]0?-
&?#
!?
inputs??????????
? ")?&
?
0?????????
? 
+__inference_reshape_9_layer_call_fn_1122259P0?-
&?#
!?
inputs??????????
? "???????????
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121075w !"#$%&D?A
:?7
-?*
flatten_9_input?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121100w !"#$%&D?A
:?7
-?*
flatten_9_input?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121912n !"#$%&;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_18_layer_call_and_return_conditional_losses_1121946n !"#$%&;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
/__inference_sequential_18_layer_call_fn_1121147j !"#$%&D?A
:?7
-?*
flatten_9_input?????????
p

 
? "??????????
?
/__inference_sequential_18_layer_call_fn_1121193j !"#$%&D?A
:?7
-?*
flatten_9_input?????????
p 

 
? "??????????
?
/__inference_sequential_18_layer_call_fn_1121967a !"#$%&;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
/__inference_sequential_18_layer_call_fn_1121988a !"#$%&;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121300t'()*+,??<
5?2
(?%
dense_67_input?????????

p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1121320t'()*+,??<
5?2
(?%
dense_67_input?????????

p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1122022l'()*+,7?4
-?*
 ?
inputs?????????

p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_19_layer_call_and_return_conditional_losses_1122056l'()*+,7?4
-?*
 ?
inputs?????????

p 

 
? ")?&
?
0?????????
? ?
/__inference_sequential_19_layer_call_fn_1121358g'()*+,??<
5?2
(?%
dense_67_input?????????

p

 
? "???????????
/__inference_sequential_19_layer_call_fn_1121395g'()*+,??<
5?2
(?%
dense_67_input?????????

p 

 
? "???????????
/__inference_sequential_19_layer_call_fn_1122073_'()*+,7?4
-?*
 ?
inputs?????????

p

 
? "???????????
/__inference_sequential_19_layer_call_fn_1122090_'()*+,7?4
-?*
 ?
inputs?????????

p 

 
? "???????????
%__inference_signature_wrapper_1121684? !"#$%&'()*+,??<
? 
5?2
0
input_1%?"
input_1?????????"7?4
2
output_1&?#
output_1?????????