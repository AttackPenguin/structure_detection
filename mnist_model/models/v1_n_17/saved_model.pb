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
~
dense_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_112/kernel
w
$dense_112/kernel/Read/ReadVariableOpReadVariableOpdense_112/kernel* 
_output_shapes
:
??*
dtype0
u
dense_112/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_112/bias
n
"dense_112/bias/Read/ReadVariableOpReadVariableOpdense_112/bias*
_output_shapes	
:?*
dtype0
}
dense_113/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*!
shared_namedense_113/kernel
v
$dense_113/kernel/Read/ReadVariableOpReadVariableOpdense_113/kernel*
_output_shapes
:	?d*
dtype0
t
dense_113/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_113/bias
m
"dense_113/bias/Read/ReadVariableOpReadVariableOpdense_113/bias*
_output_shapes
:d*
dtype0
|
dense_114/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*!
shared_namedense_114/kernel
u
$dense_114/kernel/Read/ReadVariableOpReadVariableOpdense_114/kernel*
_output_shapes

:dd*
dtype0
t
dense_114/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_114/bias
m
"dense_114/bias/Read/ReadVariableOpReadVariableOpdense_114/bias*
_output_shapes
:d*
dtype0
|
dense_115/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_115/kernel
u
$dense_115/kernel/Read/ReadVariableOpReadVariableOpdense_115/kernel*
_output_shapes

:d*
dtype0
t
dense_115/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_115/bias
m
"dense_115/bias/Read/ReadVariableOpReadVariableOpdense_115/bias*
_output_shapes
:*
dtype0
|
dense_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_116/kernel
u
$dense_116/kernel/Read/ReadVariableOpReadVariableOpdense_116/kernel*
_output_shapes

:d*
dtype0
t
dense_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_116/bias
m
"dense_116/bias/Read/ReadVariableOpReadVariableOpdense_116/bias*
_output_shapes
:d*
dtype0
|
dense_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*!
shared_namedense_117/kernel
u
$dense_117/kernel/Read/ReadVariableOpReadVariableOpdense_117/kernel*
_output_shapes

:dd*
dtype0
t
dense_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_117/bias
m
"dense_117/bias/Read/ReadVariableOpReadVariableOpdense_117/bias*
_output_shapes
:d*
dtype0
}
dense_118/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*!
shared_namedense_118/kernel
v
$dense_118/kernel/Read/ReadVariableOpReadVariableOpdense_118/kernel*
_output_shapes
:	d?*
dtype0
u
dense_118/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_118/bias
n
"dense_118/bias/Read/ReadVariableOpReadVariableOpdense_118/bias*
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
Adam/dense_112/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_112/kernel/m
?
+Adam/dense_112/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_112/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_112/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_112/bias/m
|
)Adam/dense_112/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_112/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_113/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*(
shared_nameAdam/dense_113/kernel/m
?
+Adam/dense_113/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_113/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/dense_113/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_113/bias/m
{
)Adam/dense_113/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_113/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_114/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameAdam/dense_114/kernel/m
?
+Adam/dense_114/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_114/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_114/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_114/bias/m
{
)Adam/dense_114/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_114/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_115/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_115/kernel/m
?
+Adam/dense_115/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_115/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_115/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_115/bias/m
{
)Adam/dense_115/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_115/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_116/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_116/kernel/m
?
+Adam/dense_116/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_116/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_116/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_116/bias/m
{
)Adam/dense_116/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_116/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_117/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameAdam/dense_117/kernel/m
?
+Adam/dense_117/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_117/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_117/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_117/bias/m
{
)Adam/dense_117/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_117/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_118/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_nameAdam/dense_118/kernel/m
?
+Adam/dense_118/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_118/kernel/m*
_output_shapes
:	d?*
dtype0
?
Adam/dense_118/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_118/bias/m
|
)Adam/dense_118/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_118/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_112/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_112/kernel/v
?
+Adam/dense_112/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_112/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_112/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_112/bias/v
|
)Adam/dense_112/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_112/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_113/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*(
shared_nameAdam/dense_113/kernel/v
?
+Adam/dense_113/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_113/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/dense_113/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_113/bias/v
{
)Adam/dense_113/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_113/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_114/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameAdam/dense_114/kernel/v
?
+Adam/dense_114/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_114/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_114/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_114/bias/v
{
)Adam/dense_114/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_114/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_115/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_115/kernel/v
?
+Adam/dense_115/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_115/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_115/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_115/bias/v
{
)Adam/dense_115/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_115/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_116/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_116/kernel/v
?
+Adam/dense_116/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_116/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_116/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_116/bias/v
{
)Adam/dense_116/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_116/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_117/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameAdam/dense_117/kernel/v
?
+Adam/dense_117/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_117/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_117/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_117/bias/v
{
)Adam/dense_117/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_117/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_118/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_nameAdam/dense_118/kernel/v
?
+Adam/dense_118/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_118/kernel/v*
_output_shapes
:	d?*
dtype0
?
Adam/dense_118/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_118/bias/v
|
)Adam/dense_118/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_118/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?L
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?L
value?LB?L B?L
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
LJ
VARIABLE_VALUEdense_112/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_112/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_113/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_113/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_114/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_114/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_115/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_115/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_116/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_116/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_117/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_117/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_118/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_118/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
om
VARIABLE_VALUEAdam/dense_112/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_112/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_113/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_113/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_114/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_114/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_115/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_115/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_116/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_116/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dense_117/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_117/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dense_118/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_118/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_112/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_112/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_113/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_113/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_114/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_114/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_115/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_115/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_116/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_116/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dense_117/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_117/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dense_118/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_118/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_112/kerneldense_112/biasdense_113/kerneldense_113/biasdense_114/kerneldense_114/biasdense_115/kerneldense_115/biasdense_116/kerneldense_116/biasdense_117/kerneldense_117/biasdense_118/kerneldense_118/bias*
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
%__inference_signature_wrapper_1968258
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$dense_112/kernel/Read/ReadVariableOp"dense_112/bias/Read/ReadVariableOp$dense_113/kernel/Read/ReadVariableOp"dense_113/bias/Read/ReadVariableOp$dense_114/kernel/Read/ReadVariableOp"dense_114/bias/Read/ReadVariableOp$dense_115/kernel/Read/ReadVariableOp"dense_115/bias/Read/ReadVariableOp$dense_116/kernel/Read/ReadVariableOp"dense_116/bias/Read/ReadVariableOp$dense_117/kernel/Read/ReadVariableOp"dense_117/bias/Read/ReadVariableOp$dense_118/kernel/Read/ReadVariableOp"dense_118/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_112/kernel/m/Read/ReadVariableOp)Adam/dense_112/bias/m/Read/ReadVariableOp+Adam/dense_113/kernel/m/Read/ReadVariableOp)Adam/dense_113/bias/m/Read/ReadVariableOp+Adam/dense_114/kernel/m/Read/ReadVariableOp)Adam/dense_114/bias/m/Read/ReadVariableOp+Adam/dense_115/kernel/m/Read/ReadVariableOp)Adam/dense_115/bias/m/Read/ReadVariableOp+Adam/dense_116/kernel/m/Read/ReadVariableOp)Adam/dense_116/bias/m/Read/ReadVariableOp+Adam/dense_117/kernel/m/Read/ReadVariableOp)Adam/dense_117/bias/m/Read/ReadVariableOp+Adam/dense_118/kernel/m/Read/ReadVariableOp)Adam/dense_118/bias/m/Read/ReadVariableOp+Adam/dense_112/kernel/v/Read/ReadVariableOp)Adam/dense_112/bias/v/Read/ReadVariableOp+Adam/dense_113/kernel/v/Read/ReadVariableOp)Adam/dense_113/bias/v/Read/ReadVariableOp+Adam/dense_114/kernel/v/Read/ReadVariableOp)Adam/dense_114/bias/v/Read/ReadVariableOp+Adam/dense_115/kernel/v/Read/ReadVariableOp)Adam/dense_115/bias/v/Read/ReadVariableOp+Adam/dense_116/kernel/v/Read/ReadVariableOp)Adam/dense_116/bias/v/Read/ReadVariableOp+Adam/dense_117/kernel/v/Read/ReadVariableOp)Adam/dense_117/bias/v/Read/ReadVariableOp+Adam/dense_118/kernel/v/Read/ReadVariableOp)Adam/dense_118/bias/v/Read/ReadVariableOpConst*>
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
 __inference__traced_save_1969003
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_112/kerneldense_112/biasdense_113/kerneldense_113/biasdense_114/kerneldense_114/biasdense_115/kerneldense_115/biasdense_116/kerneldense_116/biasdense_117/kerneldense_117/biasdense_118/kerneldense_118/biastotalcountAdam/dense_112/kernel/mAdam/dense_112/bias/mAdam/dense_113/kernel/mAdam/dense_113/bias/mAdam/dense_114/kernel/mAdam/dense_114/bias/mAdam/dense_115/kernel/mAdam/dense_115/bias/mAdam/dense_116/kernel/mAdam/dense_116/bias/mAdam/dense_117/kernel/mAdam/dense_117/bias/mAdam/dense_118/kernel/mAdam/dense_118/bias/mAdam/dense_112/kernel/vAdam/dense_112/bias/vAdam/dense_113/kernel/vAdam/dense_113/bias/vAdam/dense_114/kernel/vAdam/dense_114/bias/vAdam/dense_115/kernel/vAdam/dense_115/bias/vAdam/dense_116/kernel/vAdam/dense_116/bias/vAdam/dense_117/kernel/vAdam/dense_117/bias/vAdam/dense_118/kernel/vAdam/dense_118/bias/v*=
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
#__inference__traced_restore_1969160??

?i
?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968386
x:
6sequential_32_dense_112_matmul_readvariableop_resource;
7sequential_32_dense_112_biasadd_readvariableop_resource:
6sequential_32_dense_113_matmul_readvariableop_resource;
7sequential_32_dense_113_biasadd_readvariableop_resource:
6sequential_32_dense_114_matmul_readvariableop_resource;
7sequential_32_dense_114_biasadd_readvariableop_resource:
6sequential_32_dense_115_matmul_readvariableop_resource;
7sequential_32_dense_115_biasadd_readvariableop_resource:
6sequential_33_dense_116_matmul_readvariableop_resource;
7sequential_33_dense_116_biasadd_readvariableop_resource:
6sequential_33_dense_117_matmul_readvariableop_resource;
7sequential_33_dense_117_biasadd_readvariableop_resource:
6sequential_33_dense_118_matmul_readvariableop_resource;
7sequential_33_dense_118_biasadd_readvariableop_resource
identity??.sequential_32/dense_112/BiasAdd/ReadVariableOp?-sequential_32/dense_112/MatMul/ReadVariableOp?.sequential_32/dense_113/BiasAdd/ReadVariableOp?-sequential_32/dense_113/MatMul/ReadVariableOp?.sequential_32/dense_114/BiasAdd/ReadVariableOp?-sequential_32/dense_114/MatMul/ReadVariableOp?.sequential_32/dense_115/BiasAdd/ReadVariableOp?-sequential_32/dense_115/MatMul/ReadVariableOp?.sequential_33/dense_116/BiasAdd/ReadVariableOp?-sequential_33/dense_116/MatMul/ReadVariableOp?.sequential_33/dense_117/BiasAdd/ReadVariableOp?-sequential_33/dense_117/MatMul/ReadVariableOp?.sequential_33/dense_118/BiasAdd/ReadVariableOp?-sequential_33/dense_118/MatMul/ReadVariableOp?
sequential_32/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2 
sequential_32/flatten_16/Const?
 sequential_32/flatten_16/ReshapeReshapex'sequential_32/flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_32/flatten_16/Reshape?
-sequential_32/dense_112/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_112_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential_32/dense_112/MatMul/ReadVariableOp?
sequential_32/dense_112/MatMulMatMul)sequential_32/flatten_16/Reshape:output:05sequential_32/dense_112/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_32/dense_112/MatMul?
.sequential_32/dense_112/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_112_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_32/dense_112/BiasAdd/ReadVariableOp?
sequential_32/dense_112/BiasAddBiasAdd(sequential_32/dense_112/MatMul:product:06sequential_32/dense_112/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_32/dense_112/BiasAdd?
sequential_32/dense_112/ReluRelu(sequential_32/dense_112/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_32/dense_112/Relu?
-sequential_32/dense_113/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_113_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02/
-sequential_32/dense_113/MatMul/ReadVariableOp?
sequential_32/dense_113/MatMulMatMul*sequential_32/dense_112/Relu:activations:05sequential_32/dense_113/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_32/dense_113/MatMul?
.sequential_32/dense_113/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_113_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_32/dense_113/BiasAdd/ReadVariableOp?
sequential_32/dense_113/BiasAddBiasAdd(sequential_32/dense_113/MatMul:product:06sequential_32/dense_113/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_32/dense_113/BiasAdd?
sequential_32/dense_113/ReluRelu(sequential_32/dense_113/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_32/dense_113/Relu?
-sequential_32/dense_114/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_114_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02/
-sequential_32/dense_114/MatMul/ReadVariableOp?
sequential_32/dense_114/MatMulMatMul*sequential_32/dense_113/Relu:activations:05sequential_32/dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_32/dense_114/MatMul?
.sequential_32/dense_114/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_114_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_32/dense_114/BiasAdd/ReadVariableOp?
sequential_32/dense_114/BiasAddBiasAdd(sequential_32/dense_114/MatMul:product:06sequential_32/dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_32/dense_114/BiasAdd?
sequential_32/dense_114/ReluRelu(sequential_32/dense_114/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_32/dense_114/Relu?
-sequential_32/dense_115/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_115_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_32/dense_115/MatMul/ReadVariableOp?
sequential_32/dense_115/MatMulMatMul*sequential_32/dense_114/Relu:activations:05sequential_32/dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_32/dense_115/MatMul?
.sequential_32/dense_115/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_32/dense_115/BiasAdd/ReadVariableOp?
sequential_32/dense_115/BiasAddBiasAdd(sequential_32/dense_115/MatMul:product:06sequential_32/dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_32/dense_115/BiasAdd?
 sequential_32/dense_115/SoftsignSoftsign(sequential_32/dense_115/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 sequential_32/dense_115/Softsign?
-sequential_33/dense_116/MatMul/ReadVariableOpReadVariableOp6sequential_33_dense_116_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_33/dense_116/MatMul/ReadVariableOp?
sequential_33/dense_116/MatMulMatMul.sequential_32/dense_115/Softsign:activations:05sequential_33/dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_33/dense_116/MatMul?
.sequential_33/dense_116/BiasAdd/ReadVariableOpReadVariableOp7sequential_33_dense_116_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_33/dense_116/BiasAdd/ReadVariableOp?
sequential_33/dense_116/BiasAddBiasAdd(sequential_33/dense_116/MatMul:product:06sequential_33/dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_33/dense_116/BiasAdd?
sequential_33/dense_116/ReluRelu(sequential_33/dense_116/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_33/dense_116/Relu?
-sequential_33/dense_117/MatMul/ReadVariableOpReadVariableOp6sequential_33_dense_117_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02/
-sequential_33/dense_117/MatMul/ReadVariableOp?
sequential_33/dense_117/MatMulMatMul*sequential_33/dense_116/Relu:activations:05sequential_33/dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_33/dense_117/MatMul?
.sequential_33/dense_117/BiasAdd/ReadVariableOpReadVariableOp7sequential_33_dense_117_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_33/dense_117/BiasAdd/ReadVariableOp?
sequential_33/dense_117/BiasAddBiasAdd(sequential_33/dense_117/MatMul:product:06sequential_33/dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_33/dense_117/BiasAdd?
sequential_33/dense_117/ReluRelu(sequential_33/dense_117/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_33/dense_117/Relu?
-sequential_33/dense_118/MatMul/ReadVariableOpReadVariableOp6sequential_33_dense_118_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02/
-sequential_33/dense_118/MatMul/ReadVariableOp?
sequential_33/dense_118/MatMulMatMul*sequential_33/dense_117/Relu:activations:05sequential_33/dense_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_33/dense_118/MatMul?
.sequential_33/dense_118/BiasAdd/ReadVariableOpReadVariableOp7sequential_33_dense_118_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_33/dense_118/BiasAdd/ReadVariableOp?
sequential_33/dense_118/BiasAddBiasAdd(sequential_33/dense_118/MatMul:product:06sequential_33/dense_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_33/dense_118/BiasAdd?
sequential_33/dense_118/SigmoidSigmoid(sequential_33/dense_118/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_33/dense_118/Sigmoid?
sequential_33/reshape_16/ShapeShape#sequential_33/dense_118/Sigmoid:y:0*
T0*
_output_shapes
:2 
sequential_33/reshape_16/Shape?
,sequential_33/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_33/reshape_16/strided_slice/stack?
.sequential_33/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_33/reshape_16/strided_slice/stack_1?
.sequential_33/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_33/reshape_16/strided_slice/stack_2?
&sequential_33/reshape_16/strided_sliceStridedSlice'sequential_33/reshape_16/Shape:output:05sequential_33/reshape_16/strided_slice/stack:output:07sequential_33/reshape_16/strided_slice/stack_1:output:07sequential_33/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_33/reshape_16/strided_slice?
(sequential_33/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_33/reshape_16/Reshape/shape/1?
(sequential_33/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_33/reshape_16/Reshape/shape/2?
&sequential_33/reshape_16/Reshape/shapePack/sequential_33/reshape_16/strided_slice:output:01sequential_33/reshape_16/Reshape/shape/1:output:01sequential_33/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_33/reshape_16/Reshape/shape?
 sequential_33/reshape_16/ReshapeReshape#sequential_33/dense_118/Sigmoid:y:0/sequential_33/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2"
 sequential_33/reshape_16/Reshape?
IdentityIdentity)sequential_33/reshape_16/Reshape:output:0/^sequential_32/dense_112/BiasAdd/ReadVariableOp.^sequential_32/dense_112/MatMul/ReadVariableOp/^sequential_32/dense_113/BiasAdd/ReadVariableOp.^sequential_32/dense_113/MatMul/ReadVariableOp/^sequential_32/dense_114/BiasAdd/ReadVariableOp.^sequential_32/dense_114/MatMul/ReadVariableOp/^sequential_32/dense_115/BiasAdd/ReadVariableOp.^sequential_32/dense_115/MatMul/ReadVariableOp/^sequential_33/dense_116/BiasAdd/ReadVariableOp.^sequential_33/dense_116/MatMul/ReadVariableOp/^sequential_33/dense_117/BiasAdd/ReadVariableOp.^sequential_33/dense_117/MatMul/ReadVariableOp/^sequential_33/dense_118/BiasAdd/ReadVariableOp.^sequential_33/dense_118/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2`
.sequential_32/dense_112/BiasAdd/ReadVariableOp.sequential_32/dense_112/BiasAdd/ReadVariableOp2^
-sequential_32/dense_112/MatMul/ReadVariableOp-sequential_32/dense_112/MatMul/ReadVariableOp2`
.sequential_32/dense_113/BiasAdd/ReadVariableOp.sequential_32/dense_113/BiasAdd/ReadVariableOp2^
-sequential_32/dense_113/MatMul/ReadVariableOp-sequential_32/dense_113/MatMul/ReadVariableOp2`
.sequential_32/dense_114/BiasAdd/ReadVariableOp.sequential_32/dense_114/BiasAdd/ReadVariableOp2^
-sequential_32/dense_114/MatMul/ReadVariableOp-sequential_32/dense_114/MatMul/ReadVariableOp2`
.sequential_32/dense_115/BiasAdd/ReadVariableOp.sequential_32/dense_115/BiasAdd/ReadVariableOp2^
-sequential_32/dense_115/MatMul/ReadVariableOp-sequential_32/dense_115/MatMul/ReadVariableOp2`
.sequential_33/dense_116/BiasAdd/ReadVariableOp.sequential_33/dense_116/BiasAdd/ReadVariableOp2^
-sequential_33/dense_116/MatMul/ReadVariableOp-sequential_33/dense_116/MatMul/ReadVariableOp2`
.sequential_33/dense_117/BiasAdd/ReadVariableOp.sequential_33/dense_117/BiasAdd/ReadVariableOp2^
-sequential_33/dense_117/MatMul/ReadVariableOp-sequential_33/dense_117/MatMul/ReadVariableOp2`
.sequential_33/dense_118/BiasAdd/ReadVariableOp.sequential_33/dense_118/BiasAdd/ReadVariableOp2^
-sequential_33/dense_118/MatMul/ReadVariableOp-sequential_33/dense_118/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?
c
G__inference_flatten_16_layer_call_and_return_conditional_losses_1967532

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
??
?
#__inference__traced_restore_1969160
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate'
#assignvariableop_5_dense_112_kernel%
!assignvariableop_6_dense_112_bias'
#assignvariableop_7_dense_113_kernel%
!assignvariableop_8_dense_113_bias'
#assignvariableop_9_dense_114_kernel&
"assignvariableop_10_dense_114_bias(
$assignvariableop_11_dense_115_kernel&
"assignvariableop_12_dense_115_bias(
$assignvariableop_13_dense_116_kernel&
"assignvariableop_14_dense_116_bias(
$assignvariableop_15_dense_117_kernel&
"assignvariableop_16_dense_117_bias(
$assignvariableop_17_dense_118_kernel&
"assignvariableop_18_dense_118_bias
assignvariableop_19_total
assignvariableop_20_count/
+assignvariableop_21_adam_dense_112_kernel_m-
)assignvariableop_22_adam_dense_112_bias_m/
+assignvariableop_23_adam_dense_113_kernel_m-
)assignvariableop_24_adam_dense_113_bias_m/
+assignvariableop_25_adam_dense_114_kernel_m-
)assignvariableop_26_adam_dense_114_bias_m/
+assignvariableop_27_adam_dense_115_kernel_m-
)assignvariableop_28_adam_dense_115_bias_m/
+assignvariableop_29_adam_dense_116_kernel_m-
)assignvariableop_30_adam_dense_116_bias_m/
+assignvariableop_31_adam_dense_117_kernel_m-
)assignvariableop_32_adam_dense_117_bias_m/
+assignvariableop_33_adam_dense_118_kernel_m-
)assignvariableop_34_adam_dense_118_bias_m/
+assignvariableop_35_adam_dense_112_kernel_v-
)assignvariableop_36_adam_dense_112_bias_v/
+assignvariableop_37_adam_dense_113_kernel_v-
)assignvariableop_38_adam_dense_113_bias_v/
+assignvariableop_39_adam_dense_114_kernel_v-
)assignvariableop_40_adam_dense_114_bias_v/
+assignvariableop_41_adam_dense_115_kernel_v-
)assignvariableop_42_adam_dense_115_bias_v/
+assignvariableop_43_adam_dense_116_kernel_v-
)assignvariableop_44_adam_dense_116_bias_v/
+assignvariableop_45_adam_dense_117_kernel_v-
)assignvariableop_46_adam_dense_117_bias_v/
+assignvariableop_47_adam_dense_118_kernel_v-
)assignvariableop_48_adam_dense_118_bias_v
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
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_112_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_112_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_113_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_113_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_114_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_114_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_115_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_115_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_116_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_116_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_117_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_117_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_dense_118_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_118_biasIdentity_18:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_112_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_112_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_113_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_113_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_114_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_114_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_115_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_115_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_116_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_116_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_117_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_117_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_118_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_118_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_112_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_112_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_113_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_113_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_114_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_114_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_115_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_115_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_dense_116_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_116_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_117_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_117_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_118_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_118_bias_vIdentity_48:output:0"/device:CPU:0*
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
?
?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967894
dense_116_input
dense_116_1967877
dense_116_1967879
dense_117_1967882
dense_117_1967884
dense_118_1967887
dense_118_1967889
identity??!dense_116/StatefulPartitionedCall?!dense_117/StatefulPartitionedCall?!dense_118/StatefulPartitionedCall?
!dense_116/StatefulPartitionedCallStatefulPartitionedCalldense_116_inputdense_116_1967877dense_116_1967879*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_116_layer_call_and_return_conditional_losses_19677822#
!dense_116/StatefulPartitionedCall?
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*dense_116/StatefulPartitionedCall:output:0dense_117_1967882dense_117_1967884*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_19678092#
!dense_117/StatefulPartitionedCall?
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_1967887dense_118_1967889*
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
F__inference_dense_118_layer_call_and_return_conditional_losses_19678362#
!dense_118/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCall*dense_118/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *P
fKRI
G__inference_reshape_16_layer_call_and_return_conditional_losses_19678652
reshape_16/PartitionedCall?
IdentityIdentity#reshape_16/PartitionedCall:output:0"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedense_116_input
?

?
0__inference_autoencoder_16_layer_call_fn_1968215
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
GPU2*0,1J 8? *T
fORM
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_19681512
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
0__inference_autoencoder_16_layer_call_fn_1968419
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
GPU2*0,1J 8? *T
fORM
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_19681512
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
?
?
+__inference_dense_116_layer_call_fn_1968775

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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_116_layer_call_and_return_conditional_losses_19677822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_118_layer_call_and_return_conditional_losses_1968806

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
?
?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967649
flatten_16_input
dense_112_1967562
dense_112_1967564
dense_113_1967589
dense_113_1967591
dense_114_1967616
dense_114_1967618
dense_115_1967643
dense_115_1967645
identity??!dense_112/StatefulPartitionedCall?!dense_113/StatefulPartitionedCall?!dense_114/StatefulPartitionedCall?!dense_115/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCallflatten_16_input*
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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_19675322
flatten_16/PartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_112_1967562dense_112_1967564*
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
F__inference_dense_112_layer_call_and_return_conditional_losses_19675512#
!dense_112/StatefulPartitionedCall?
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1967589dense_113_1967591*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_19675782#
!dense_113/StatefulPartitionedCall?
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1967616dense_114_1967618*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_19676052#
!dense_114/StatefulPartitionedCall?
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1967643dense_115_1967645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_19676322#
!dense_115/StatefulPartitionedCall?
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameflatten_16_input
?

?
F__inference_dense_115_layer_call_and_return_conditional_losses_1967632

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
F__inference_dense_115_layer_call_and_return_conditional_losses_1968746

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
SoftsignSoftsignBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Softsign?
IdentityIdentitySoftsign:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
c
G__inference_reshape_16_layer_call_and_return_conditional_losses_1968828

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
?
?
/__inference_sequential_33_layer_call_fn_1967932
dense_116_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_116_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedense_116_input
?
?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967954

inputs
dense_116_1967937
dense_116_1967939
dense_117_1967942
dense_117_1967944
dense_118_1967947
dense_118_1967949
identity??!dense_116/StatefulPartitionedCall?!dense_117/StatefulPartitionedCall?!dense_118/StatefulPartitionedCall?
!dense_116/StatefulPartitionedCallStatefulPartitionedCallinputsdense_116_1967937dense_116_1967939*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_116_layer_call_and_return_conditional_losses_19677822#
!dense_116/StatefulPartitionedCall?
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*dense_116/StatefulPartitionedCall:output:0dense_117_1967942dense_117_1967944*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_19678092#
!dense_117/StatefulPartitionedCall?
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_1967947dense_118_1967949*
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
F__inference_dense_118_layer_call_and_return_conditional_losses_19678362#
!dense_118/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCall*dense_118/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *P
fKRI
G__inference_reshape_16_layer_call_and_return_conditional_losses_19678652
reshape_16/PartitionedCall?
IdentityIdentity#reshape_16/PartitionedCall:output:0"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967874
dense_116_input
dense_116_1967793
dense_116_1967795
dense_117_1967820
dense_117_1967822
dense_118_1967847
dense_118_1967849
identity??!dense_116/StatefulPartitionedCall?!dense_117/StatefulPartitionedCall?!dense_118/StatefulPartitionedCall?
!dense_116/StatefulPartitionedCallStatefulPartitionedCalldense_116_inputdense_116_1967793dense_116_1967795*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_116_layer_call_and_return_conditional_losses_19677822#
!dense_116/StatefulPartitionedCall?
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*dense_116/StatefulPartitionedCall:output:0dense_117_1967820dense_117_1967822*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_19678092#
!dense_117/StatefulPartitionedCall?
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_1967847dense_118_1967849*
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
F__inference_dense_118_layer_call_and_return_conditional_losses_19678362#
!dense_118/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCall*dense_118/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *P
fKRI
G__inference_reshape_16_layer_call_and_return_conditional_losses_19678652
reshape_16/PartitionedCall?
IdentityIdentity#reshape_16/PartitionedCall:output:0"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedense_116_input
?
?
+__inference_dense_113_layer_call_fn_1968715

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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_19675782
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
?
"__inference__wrapped_model_1967522
input_1I
Eautoencoder_16_sequential_32_dense_112_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_32_dense_112_biasadd_readvariableop_resourceI
Eautoencoder_16_sequential_32_dense_113_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_32_dense_113_biasadd_readvariableop_resourceI
Eautoencoder_16_sequential_32_dense_114_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_32_dense_114_biasadd_readvariableop_resourceI
Eautoencoder_16_sequential_32_dense_115_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_32_dense_115_biasadd_readvariableop_resourceI
Eautoencoder_16_sequential_33_dense_116_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_33_dense_116_biasadd_readvariableop_resourceI
Eautoencoder_16_sequential_33_dense_117_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_33_dense_117_biasadd_readvariableop_resourceI
Eautoencoder_16_sequential_33_dense_118_matmul_readvariableop_resourceJ
Fautoencoder_16_sequential_33_dense_118_biasadd_readvariableop_resource
identity??=autoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOp?=autoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOp?=autoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOp?=autoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOp?=autoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOp?=autoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOp?=autoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOp?<autoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOp?
-autoencoder_16/sequential_32/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2/
-autoencoder_16/sequential_32/flatten_16/Const?
/autoencoder_16/sequential_32/flatten_16/ReshapeReshapeinput_16autoencoder_16/sequential_32/flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????21
/autoencoder_16/sequential_32/flatten_16/Reshape?
<autoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_32_dense_112_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02>
<autoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOp?
-autoencoder_16/sequential_32/dense_112/MatMulMatMul8autoencoder_16/sequential_32/flatten_16/Reshape:output:0Dautoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-autoencoder_16/sequential_32/dense_112/MatMul?
=autoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_32_dense_112_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=autoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_32/dense_112/BiasAddBiasAdd7autoencoder_16/sequential_32/dense_112/MatMul:product:0Eautoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.autoencoder_16/sequential_32/dense_112/BiasAdd?
+autoencoder_16/sequential_32/dense_112/ReluRelu7autoencoder_16/sequential_32/dense_112/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_16/sequential_32/dense_112/Relu?
<autoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_32_dense_113_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02>
<autoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOp?
-autoencoder_16/sequential_32/dense_113/MatMulMatMul9autoencoder_16/sequential_32/dense_112/Relu:activations:0Dautoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-autoencoder_16/sequential_32/dense_113/MatMul?
=autoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_32_dense_113_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02?
=autoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_32/dense_113/BiasAddBiasAdd7autoencoder_16/sequential_32/dense_113/MatMul:product:0Eautoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d20
.autoencoder_16/sequential_32/dense_113/BiasAdd?
+autoencoder_16/sequential_32/dense_113/ReluRelu7autoencoder_16/sequential_32/dense_113/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_16/sequential_32/dense_113/Relu?
<autoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_32_dense_114_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02>
<autoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOp?
-autoencoder_16/sequential_32/dense_114/MatMulMatMul9autoencoder_16/sequential_32/dense_113/Relu:activations:0Dautoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-autoencoder_16/sequential_32/dense_114/MatMul?
=autoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_32_dense_114_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02?
=autoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_32/dense_114/BiasAddBiasAdd7autoencoder_16/sequential_32/dense_114/MatMul:product:0Eautoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d20
.autoencoder_16/sequential_32/dense_114/BiasAdd?
+autoencoder_16/sequential_32/dense_114/ReluRelu7autoencoder_16/sequential_32/dense_114/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_16/sequential_32/dense_114/Relu?
<autoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_32_dense_115_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02>
<autoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOp?
-autoencoder_16/sequential_32/dense_115/MatMulMatMul9autoencoder_16/sequential_32/dense_114/Relu:activations:0Dautoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-autoencoder_16/sequential_32/dense_115/MatMul?
=autoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_32_dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=autoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_32/dense_115/BiasAddBiasAdd7autoencoder_16/sequential_32/dense_115/MatMul:product:0Eautoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.autoencoder_16/sequential_32/dense_115/BiasAdd?
/autoencoder_16/sequential_32/dense_115/SoftsignSoftsign7autoencoder_16/sequential_32/dense_115/BiasAdd:output:0*
T0*'
_output_shapes
:?????????21
/autoencoder_16/sequential_32/dense_115/Softsign?
<autoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_33_dense_116_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02>
<autoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOp?
-autoencoder_16/sequential_33/dense_116/MatMulMatMul=autoencoder_16/sequential_32/dense_115/Softsign:activations:0Dautoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-autoencoder_16/sequential_33/dense_116/MatMul?
=autoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_33_dense_116_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02?
=autoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_33/dense_116/BiasAddBiasAdd7autoencoder_16/sequential_33/dense_116/MatMul:product:0Eautoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d20
.autoencoder_16/sequential_33/dense_116/BiasAdd?
+autoencoder_16/sequential_33/dense_116/ReluRelu7autoencoder_16/sequential_33/dense_116/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_16/sequential_33/dense_116/Relu?
<autoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_33_dense_117_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02>
<autoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOp?
-autoencoder_16/sequential_33/dense_117/MatMulMatMul9autoencoder_16/sequential_33/dense_116/Relu:activations:0Dautoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-autoencoder_16/sequential_33/dense_117/MatMul?
=autoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_33_dense_117_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02?
=autoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_33/dense_117/BiasAddBiasAdd7autoencoder_16/sequential_33/dense_117/MatMul:product:0Eautoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d20
.autoencoder_16/sequential_33/dense_117/BiasAdd?
+autoencoder_16/sequential_33/dense_117/ReluRelu7autoencoder_16/sequential_33/dense_117/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2-
+autoencoder_16/sequential_33/dense_117/Relu?
<autoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOpReadVariableOpEautoencoder_16_sequential_33_dense_118_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02>
<autoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOp?
-autoencoder_16/sequential_33/dense_118/MatMulMatMul9autoencoder_16/sequential_33/dense_117/Relu:activations:0Dautoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-autoencoder_16/sequential_33/dense_118/MatMul?
=autoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_16_sequential_33_dense_118_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=autoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOp?
.autoencoder_16/sequential_33/dense_118/BiasAddBiasAdd7autoencoder_16/sequential_33/dense_118/MatMul:product:0Eautoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.autoencoder_16/sequential_33/dense_118/BiasAdd?
.autoencoder_16/sequential_33/dense_118/SigmoidSigmoid7autoencoder_16/sequential_33/dense_118/BiasAdd:output:0*
T0*(
_output_shapes
:??????????20
.autoencoder_16/sequential_33/dense_118/Sigmoid?
-autoencoder_16/sequential_33/reshape_16/ShapeShape2autoencoder_16/sequential_33/dense_118/Sigmoid:y:0*
T0*
_output_shapes
:2/
-autoencoder_16/sequential_33/reshape_16/Shape?
;autoencoder_16/sequential_33/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;autoencoder_16/sequential_33/reshape_16/strided_slice/stack?
=autoencoder_16/sequential_33/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder_16/sequential_33/reshape_16/strided_slice/stack_1?
=autoencoder_16/sequential_33/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder_16/sequential_33/reshape_16/strided_slice/stack_2?
5autoencoder_16/sequential_33/reshape_16/strided_sliceStridedSlice6autoencoder_16/sequential_33/reshape_16/Shape:output:0Dautoencoder_16/sequential_33/reshape_16/strided_slice/stack:output:0Fautoencoder_16/sequential_33/reshape_16/strided_slice/stack_1:output:0Fautoencoder_16/sequential_33/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5autoencoder_16/sequential_33/reshape_16/strided_slice?
7autoencoder_16/sequential_33/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :29
7autoencoder_16/sequential_33/reshape_16/Reshape/shape/1?
7autoencoder_16/sequential_33/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :29
7autoencoder_16/sequential_33/reshape_16/Reshape/shape/2?
5autoencoder_16/sequential_33/reshape_16/Reshape/shapePack>autoencoder_16/sequential_33/reshape_16/strided_slice:output:0@autoencoder_16/sequential_33/reshape_16/Reshape/shape/1:output:0@autoencoder_16/sequential_33/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:27
5autoencoder_16/sequential_33/reshape_16/Reshape/shape?
/autoencoder_16/sequential_33/reshape_16/ReshapeReshape2autoencoder_16/sequential_33/dense_118/Sigmoid:y:0>autoencoder_16/sequential_33/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????21
/autoencoder_16/sequential_33/reshape_16/Reshape?
IdentityIdentity8autoencoder_16/sequential_33/reshape_16/Reshape:output:0>^autoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOp>^autoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOp>^autoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOp>^autoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOp>^autoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOp>^autoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOp>^autoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOp=^autoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2~
=autoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOp=autoencoder_16/sequential_32/dense_112/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOp<autoencoder_16/sequential_32/dense_112/MatMul/ReadVariableOp2~
=autoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOp=autoencoder_16/sequential_32/dense_113/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOp<autoencoder_16/sequential_32/dense_113/MatMul/ReadVariableOp2~
=autoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOp=autoencoder_16/sequential_32/dense_114/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOp<autoencoder_16/sequential_32/dense_114/MatMul/ReadVariableOp2~
=autoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOp=autoencoder_16/sequential_32/dense_115/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOp<autoencoder_16/sequential_32/dense_115/MatMul/ReadVariableOp2~
=autoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOp=autoencoder_16/sequential_33/dense_116/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOp<autoencoder_16/sequential_33/dense_116/MatMul/ReadVariableOp2~
=autoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOp=autoencoder_16/sequential_33/dense_117/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOp<autoencoder_16/sequential_33/dense_117/MatMul/ReadVariableOp2~
=autoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOp=autoencoder_16/sequential_33/dense_118/BiasAdd/ReadVariableOp2|
<autoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOp<autoencoder_16/sequential_33/dense_118/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?i
?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968322
x:
6sequential_32_dense_112_matmul_readvariableop_resource;
7sequential_32_dense_112_biasadd_readvariableop_resource:
6sequential_32_dense_113_matmul_readvariableop_resource;
7sequential_32_dense_113_biasadd_readvariableop_resource:
6sequential_32_dense_114_matmul_readvariableop_resource;
7sequential_32_dense_114_biasadd_readvariableop_resource:
6sequential_32_dense_115_matmul_readvariableop_resource;
7sequential_32_dense_115_biasadd_readvariableop_resource:
6sequential_33_dense_116_matmul_readvariableop_resource;
7sequential_33_dense_116_biasadd_readvariableop_resource:
6sequential_33_dense_117_matmul_readvariableop_resource;
7sequential_33_dense_117_biasadd_readvariableop_resource:
6sequential_33_dense_118_matmul_readvariableop_resource;
7sequential_33_dense_118_biasadd_readvariableop_resource
identity??.sequential_32/dense_112/BiasAdd/ReadVariableOp?-sequential_32/dense_112/MatMul/ReadVariableOp?.sequential_32/dense_113/BiasAdd/ReadVariableOp?-sequential_32/dense_113/MatMul/ReadVariableOp?.sequential_32/dense_114/BiasAdd/ReadVariableOp?-sequential_32/dense_114/MatMul/ReadVariableOp?.sequential_32/dense_115/BiasAdd/ReadVariableOp?-sequential_32/dense_115/MatMul/ReadVariableOp?.sequential_33/dense_116/BiasAdd/ReadVariableOp?-sequential_33/dense_116/MatMul/ReadVariableOp?.sequential_33/dense_117/BiasAdd/ReadVariableOp?-sequential_33/dense_117/MatMul/ReadVariableOp?.sequential_33/dense_118/BiasAdd/ReadVariableOp?-sequential_33/dense_118/MatMul/ReadVariableOp?
sequential_32/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2 
sequential_32/flatten_16/Const?
 sequential_32/flatten_16/ReshapeReshapex'sequential_32/flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_32/flatten_16/Reshape?
-sequential_32/dense_112/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_112_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential_32/dense_112/MatMul/ReadVariableOp?
sequential_32/dense_112/MatMulMatMul)sequential_32/flatten_16/Reshape:output:05sequential_32/dense_112/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_32/dense_112/MatMul?
.sequential_32/dense_112/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_112_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_32/dense_112/BiasAdd/ReadVariableOp?
sequential_32/dense_112/BiasAddBiasAdd(sequential_32/dense_112/MatMul:product:06sequential_32/dense_112/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_32/dense_112/BiasAdd?
sequential_32/dense_112/ReluRelu(sequential_32/dense_112/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_32/dense_112/Relu?
-sequential_32/dense_113/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_113_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02/
-sequential_32/dense_113/MatMul/ReadVariableOp?
sequential_32/dense_113/MatMulMatMul*sequential_32/dense_112/Relu:activations:05sequential_32/dense_113/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_32/dense_113/MatMul?
.sequential_32/dense_113/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_113_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_32/dense_113/BiasAdd/ReadVariableOp?
sequential_32/dense_113/BiasAddBiasAdd(sequential_32/dense_113/MatMul:product:06sequential_32/dense_113/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_32/dense_113/BiasAdd?
sequential_32/dense_113/ReluRelu(sequential_32/dense_113/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_32/dense_113/Relu?
-sequential_32/dense_114/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_114_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02/
-sequential_32/dense_114/MatMul/ReadVariableOp?
sequential_32/dense_114/MatMulMatMul*sequential_32/dense_113/Relu:activations:05sequential_32/dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_32/dense_114/MatMul?
.sequential_32/dense_114/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_114_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_32/dense_114/BiasAdd/ReadVariableOp?
sequential_32/dense_114/BiasAddBiasAdd(sequential_32/dense_114/MatMul:product:06sequential_32/dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_32/dense_114/BiasAdd?
sequential_32/dense_114/ReluRelu(sequential_32/dense_114/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_32/dense_114/Relu?
-sequential_32/dense_115/MatMul/ReadVariableOpReadVariableOp6sequential_32_dense_115_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_32/dense_115/MatMul/ReadVariableOp?
sequential_32/dense_115/MatMulMatMul*sequential_32/dense_114/Relu:activations:05sequential_32/dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_32/dense_115/MatMul?
.sequential_32/dense_115/BiasAdd/ReadVariableOpReadVariableOp7sequential_32_dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_32/dense_115/BiasAdd/ReadVariableOp?
sequential_32/dense_115/BiasAddBiasAdd(sequential_32/dense_115/MatMul:product:06sequential_32/dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_32/dense_115/BiasAdd?
 sequential_32/dense_115/SoftsignSoftsign(sequential_32/dense_115/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 sequential_32/dense_115/Softsign?
-sequential_33/dense_116/MatMul/ReadVariableOpReadVariableOp6sequential_33_dense_116_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_33/dense_116/MatMul/ReadVariableOp?
sequential_33/dense_116/MatMulMatMul.sequential_32/dense_115/Softsign:activations:05sequential_33/dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_33/dense_116/MatMul?
.sequential_33/dense_116/BiasAdd/ReadVariableOpReadVariableOp7sequential_33_dense_116_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_33/dense_116/BiasAdd/ReadVariableOp?
sequential_33/dense_116/BiasAddBiasAdd(sequential_33/dense_116/MatMul:product:06sequential_33/dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_33/dense_116/BiasAdd?
sequential_33/dense_116/ReluRelu(sequential_33/dense_116/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_33/dense_116/Relu?
-sequential_33/dense_117/MatMul/ReadVariableOpReadVariableOp6sequential_33_dense_117_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02/
-sequential_33/dense_117/MatMul/ReadVariableOp?
sequential_33/dense_117/MatMulMatMul*sequential_33/dense_116/Relu:activations:05sequential_33/dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_33/dense_117/MatMul?
.sequential_33/dense_117/BiasAdd/ReadVariableOpReadVariableOp7sequential_33_dense_117_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_33/dense_117/BiasAdd/ReadVariableOp?
sequential_33/dense_117/BiasAddBiasAdd(sequential_33/dense_117/MatMul:product:06sequential_33/dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_33/dense_117/BiasAdd?
sequential_33/dense_117/ReluRelu(sequential_33/dense_117/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_33/dense_117/Relu?
-sequential_33/dense_118/MatMul/ReadVariableOpReadVariableOp6sequential_33_dense_118_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02/
-sequential_33/dense_118/MatMul/ReadVariableOp?
sequential_33/dense_118/MatMulMatMul*sequential_33/dense_117/Relu:activations:05sequential_33/dense_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_33/dense_118/MatMul?
.sequential_33/dense_118/BiasAdd/ReadVariableOpReadVariableOp7sequential_33_dense_118_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_33/dense_118/BiasAdd/ReadVariableOp?
sequential_33/dense_118/BiasAddBiasAdd(sequential_33/dense_118/MatMul:product:06sequential_33/dense_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_33/dense_118/BiasAdd?
sequential_33/dense_118/SigmoidSigmoid(sequential_33/dense_118/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_33/dense_118/Sigmoid?
sequential_33/reshape_16/ShapeShape#sequential_33/dense_118/Sigmoid:y:0*
T0*
_output_shapes
:2 
sequential_33/reshape_16/Shape?
,sequential_33/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_33/reshape_16/strided_slice/stack?
.sequential_33/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_33/reshape_16/strided_slice/stack_1?
.sequential_33/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_33/reshape_16/strided_slice/stack_2?
&sequential_33/reshape_16/strided_sliceStridedSlice'sequential_33/reshape_16/Shape:output:05sequential_33/reshape_16/strided_slice/stack:output:07sequential_33/reshape_16/strided_slice/stack_1:output:07sequential_33/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_33/reshape_16/strided_slice?
(sequential_33/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_33/reshape_16/Reshape/shape/1?
(sequential_33/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_33/reshape_16/Reshape/shape/2?
&sequential_33/reshape_16/Reshape/shapePack/sequential_33/reshape_16/strided_slice:output:01sequential_33/reshape_16/Reshape/shape/1:output:01sequential_33/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_33/reshape_16/Reshape/shape?
 sequential_33/reshape_16/ReshapeReshape#sequential_33/dense_118/Sigmoid:y:0/sequential_33/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2"
 sequential_33/reshape_16/Reshape?
IdentityIdentity)sequential_33/reshape_16/Reshape:output:0/^sequential_32/dense_112/BiasAdd/ReadVariableOp.^sequential_32/dense_112/MatMul/ReadVariableOp/^sequential_32/dense_113/BiasAdd/ReadVariableOp.^sequential_32/dense_113/MatMul/ReadVariableOp/^sequential_32/dense_114/BiasAdd/ReadVariableOp.^sequential_32/dense_114/MatMul/ReadVariableOp/^sequential_32/dense_115/BiasAdd/ReadVariableOp.^sequential_32/dense_115/MatMul/ReadVariableOp/^sequential_33/dense_116/BiasAdd/ReadVariableOp.^sequential_33/dense_116/MatMul/ReadVariableOp/^sequential_33/dense_117/BiasAdd/ReadVariableOp.^sequential_33/dense_117/MatMul/ReadVariableOp/^sequential_33/dense_118/BiasAdd/ReadVariableOp.^sequential_33/dense_118/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2`
.sequential_32/dense_112/BiasAdd/ReadVariableOp.sequential_32/dense_112/BiasAdd/ReadVariableOp2^
-sequential_32/dense_112/MatMul/ReadVariableOp-sequential_32/dense_112/MatMul/ReadVariableOp2`
.sequential_32/dense_113/BiasAdd/ReadVariableOp.sequential_32/dense_113/BiasAdd/ReadVariableOp2^
-sequential_32/dense_113/MatMul/ReadVariableOp-sequential_32/dense_113/MatMul/ReadVariableOp2`
.sequential_32/dense_114/BiasAdd/ReadVariableOp.sequential_32/dense_114/BiasAdd/ReadVariableOp2^
-sequential_32/dense_114/MatMul/ReadVariableOp-sequential_32/dense_114/MatMul/ReadVariableOp2`
.sequential_32/dense_115/BiasAdd/ReadVariableOp.sequential_32/dense_115/BiasAdd/ReadVariableOp2^
-sequential_32/dense_115/MatMul/ReadVariableOp-sequential_32/dense_115/MatMul/ReadVariableOp2`
.sequential_33/dense_116/BiasAdd/ReadVariableOp.sequential_33/dense_116/BiasAdd/ReadVariableOp2^
-sequential_33/dense_116/MatMul/ReadVariableOp-sequential_33/dense_116/MatMul/ReadVariableOp2`
.sequential_33/dense_117/BiasAdd/ReadVariableOp.sequential_33/dense_117/BiasAdd/ReadVariableOp2^
-sequential_33/dense_117/MatMul/ReadVariableOp-sequential_33/dense_117/MatMul/ReadVariableOp2`
.sequential_33/dense_118/BiasAdd/ReadVariableOp.sequential_33/dense_118/BiasAdd/ReadVariableOp2^
-sequential_33/dense_118/MatMul/ReadVariableOp-sequential_33/dense_118/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
F__inference_dense_113_layer_call_and_return_conditional_losses_1968706

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
F__inference_dense_118_layer_call_and_return_conditional_losses_1967836

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
?
c
G__inference_flatten_16_layer_call_and_return_conditional_losses_1968670

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
?
?
/__inference_sequential_33_layer_call_fn_1968647

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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968114
input_1
sequential_32_1968083
sequential_32_1968085
sequential_32_1968087
sequential_32_1968089
sequential_32_1968091
sequential_32_1968093
sequential_32_1968095
sequential_32_1968097
sequential_33_1968100
sequential_33_1968102
sequential_33_1968104
sequential_33_1968106
sequential_33_1968108
sequential_33_1968110
identity??%sequential_32/StatefulPartitionedCall?%sequential_33/StatefulPartitionedCall?
%sequential_32/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_32_1968083sequential_32_1968085sequential_32_1968087sequential_32_1968089sequential_32_1968091sequential_32_1968093sequential_32_1968095sequential_32_1968097*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677482'
%sequential_32/StatefulPartitionedCall?
%sequential_33/StatefulPartitionedCallStatefulPartitionedCall.sequential_32/StatefulPartitionedCall:output:0sequential_33_1968100sequential_33_1968102sequential_33_1968104sequential_33_1968106sequential_33_1968108sequential_33_1968110*
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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679542'
%sequential_33/StatefulPartitionedCall?
IdentityIdentity.sequential_33/StatefulPartitionedCall:output:0&^sequential_32/StatefulPartitionedCall&^sequential_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_32/StatefulPartitionedCall%sequential_32/StatefulPartitionedCall2N
%sequential_33/StatefulPartitionedCall%sequential_33/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
H
,__inference_reshape_16_layer_call_fn_1968833

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
GPU2*0,1J 8? *P
fKRI
G__inference_reshape_16_layer_call_and_return_conditional_losses_19678652
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
?
?
/__inference_sequential_32_layer_call_fn_1968541

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
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1968596

inputs,
(dense_116_matmul_readvariableop_resource-
)dense_116_biasadd_readvariableop_resource,
(dense_117_matmul_readvariableop_resource-
)dense_117_biasadd_readvariableop_resource,
(dense_118_matmul_readvariableop_resource-
)dense_118_biasadd_readvariableop_resource
identity?? dense_116/BiasAdd/ReadVariableOp?dense_116/MatMul/ReadVariableOp? dense_117/BiasAdd/ReadVariableOp?dense_117/MatMul/ReadVariableOp? dense_118/BiasAdd/ReadVariableOp?dense_118/MatMul/ReadVariableOp?
dense_116/MatMul/ReadVariableOpReadVariableOp(dense_116_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_116/MatMul/ReadVariableOp?
dense_116/MatMulMatMulinputs'dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_116/MatMul?
 dense_116/BiasAdd/ReadVariableOpReadVariableOp)dense_116_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_116/BiasAdd/ReadVariableOp?
dense_116/BiasAddBiasAdddense_116/MatMul:product:0(dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_116/BiasAddv
dense_116/ReluReludense_116/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_116/Relu?
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02!
dense_117/MatMul/ReadVariableOp?
dense_117/MatMulMatMuldense_116/Relu:activations:0'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_117/MatMul?
 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_117/BiasAdd/ReadVariableOp?
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_117/BiasAddv
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_117/Relu?
dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02!
dense_118/MatMul/ReadVariableOp?
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_118/MatMul?
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_118/BiasAdd/ReadVariableOp?
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_118/BiasAdd?
dense_118/SigmoidSigmoiddense_118/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_118/Sigmoidi
reshape_16/ShapeShapedense_118/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapedense_118/Sigmoid:y:0!reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_16/Reshape?
IdentityIdentityreshape_16/Reshape:output:0!^dense_116/BiasAdd/ReadVariableOp ^dense_116/MatMul/ReadVariableOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_116/BiasAdd/ReadVariableOp dense_116/BiasAdd/ReadVariableOp2B
dense_116/MatMul/ReadVariableOpdense_116/MatMul/ReadVariableOp2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_32_layer_call_fn_1967767
flatten_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameflatten_16_input
?
?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968151
x
sequential_32_1968120
sequential_32_1968122
sequential_32_1968124
sequential_32_1968126
sequential_32_1968128
sequential_32_1968130
sequential_32_1968132
sequential_32_1968134
sequential_33_1968137
sequential_33_1968139
sequential_33_1968141
sequential_33_1968143
sequential_33_1968145
sequential_33_1968147
identity??%sequential_32/StatefulPartitionedCall?%sequential_33/StatefulPartitionedCall?
%sequential_32/StatefulPartitionedCallStatefulPartitionedCallxsequential_32_1968120sequential_32_1968122sequential_32_1968124sequential_32_1968126sequential_32_1968128sequential_32_1968130sequential_32_1968132sequential_32_1968134*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677482'
%sequential_32/StatefulPartitionedCall?
%sequential_33/StatefulPartitionedCallStatefulPartitionedCall.sequential_32/StatefulPartitionedCall:output:0sequential_33_1968137sequential_33_1968139sequential_33_1968141sequential_33_1968143sequential_33_1968145sequential_33_1968147*
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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679542'
%sequential_33/StatefulPartitionedCall?
IdentityIdentity.sequential_33/StatefulPartitionedCall:output:0&^sequential_32/StatefulPartitionedCall&^sequential_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_32/StatefulPartitionedCall%sequential_32/StatefulPartitionedCall2N
%sequential_33/StatefulPartitionedCall%sequential_33/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
%__inference_signature_wrapper_1968258
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
"__inference__wrapped_model_19675222
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
?*
?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1968520

inputs,
(dense_112_matmul_readvariableop_resource-
)dense_112_biasadd_readvariableop_resource,
(dense_113_matmul_readvariableop_resource-
)dense_113_biasadd_readvariableop_resource,
(dense_114_matmul_readvariableop_resource-
)dense_114_biasadd_readvariableop_resource,
(dense_115_matmul_readvariableop_resource-
)dense_115_biasadd_readvariableop_resource
identity?? dense_112/BiasAdd/ReadVariableOp?dense_112/MatMul/ReadVariableOp? dense_113/BiasAdd/ReadVariableOp?dense_113/MatMul/ReadVariableOp? dense_114/BiasAdd/ReadVariableOp?dense_114/MatMul/ReadVariableOp? dense_115/BiasAdd/ReadVariableOp?dense_115/MatMul/ReadVariableOpu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_16/Const?
flatten_16/ReshapeReshapeinputsflatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_16/Reshape?
dense_112/MatMul/ReadVariableOpReadVariableOp(dense_112_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_112/MatMul/ReadVariableOp?
dense_112/MatMulMatMulflatten_16/Reshape:output:0'dense_112/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_112/MatMul?
 dense_112/BiasAdd/ReadVariableOpReadVariableOp)dense_112_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_112/BiasAdd/ReadVariableOp?
dense_112/BiasAddBiasAdddense_112/MatMul:product:0(dense_112/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_112/BiasAddw
dense_112/ReluReludense_112/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_112/Relu?
dense_113/MatMul/ReadVariableOpReadVariableOp(dense_113_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02!
dense_113/MatMul/ReadVariableOp?
dense_113/MatMulMatMuldense_112/Relu:activations:0'dense_113/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_113/MatMul?
 dense_113/BiasAdd/ReadVariableOpReadVariableOp)dense_113_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_113/BiasAdd/ReadVariableOp?
dense_113/BiasAddBiasAdddense_113/MatMul:product:0(dense_113/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_113/BiasAddv
dense_113/ReluReludense_113/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_113/Relu?
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02!
dense_114/MatMul/ReadVariableOp?
dense_114/MatMulMatMuldense_113/Relu:activations:0'dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_114/MatMul?
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_114/BiasAdd/ReadVariableOp?
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_114/BiasAddv
dense_114/ReluReludense_114/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_114/Relu?
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_115/MatMul/ReadVariableOp?
dense_115/MatMulMatMuldense_114/Relu:activations:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_115/MatMul?
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_115/BiasAdd/ReadVariableOp?
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_115/BiasAdd?
dense_115/SoftsignSoftsigndense_115/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_115/Softsign?
IdentityIdentity dense_115/Softsign:activations:0!^dense_112/BiasAdd/ReadVariableOp ^dense_112/MatMul/ReadVariableOp!^dense_113/BiasAdd/ReadVariableOp ^dense_113/MatMul/ReadVariableOp!^dense_114/BiasAdd/ReadVariableOp ^dense_114/MatMul/ReadVariableOp!^dense_115/BiasAdd/ReadVariableOp ^dense_115/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_112/BiasAdd/ReadVariableOp dense_112/BiasAdd/ReadVariableOp2B
dense_112/MatMul/ReadVariableOpdense_112/MatMul/ReadVariableOp2D
 dense_113/BiasAdd/ReadVariableOp dense_113/BiasAdd/ReadVariableOp2B
dense_113/MatMul/ReadVariableOpdense_113/MatMul/ReadVariableOp2D
 dense_114/BiasAdd/ReadVariableOp dense_114/BiasAdd/ReadVariableOp2B
dense_114/MatMul/ReadVariableOpdense_114/MatMul/ReadVariableOp2D
 dense_115/BiasAdd/ReadVariableOp dense_115/BiasAdd/ReadVariableOp2B
dense_115/MatMul/ReadVariableOpdense_115/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
0__inference_autoencoder_16_layer_call_fn_1968182
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
GPU2*0,1J 8? *T
fORM
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_19681512
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
?
+__inference_dense_114_layer_call_fn_1968735

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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_19676052
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
?
+__inference_dense_117_layer_call_fn_1968795

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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_19678092
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
?
+__inference_dense_115_layer_call_fn_1968755

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
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_19676322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
c
G__inference_reshape_16_layer_call_and_return_conditional_losses_1967865

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
?
?
/__inference_sequential_33_layer_call_fn_1967969
dense_116_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_116_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namedense_116_input
?	
?
F__inference_dense_117_layer_call_and_return_conditional_losses_1968786

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
?
?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967702

inputs
dense_112_1967681
dense_112_1967683
dense_113_1967686
dense_113_1967688
dense_114_1967691
dense_114_1967693
dense_115_1967696
dense_115_1967698
identity??!dense_112/StatefulPartitionedCall?!dense_113/StatefulPartitionedCall?!dense_114/StatefulPartitionedCall?!dense_115/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCallinputs*
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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_19675322
flatten_16/PartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_112_1967681dense_112_1967683*
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
F__inference_dense_112_layer_call_and_return_conditional_losses_19675512#
!dense_112/StatefulPartitionedCall?
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1967686dense_113_1967688*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_19675782#
!dense_113/StatefulPartitionedCall?
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1967691dense_114_1967693*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_19676052#
!dense_114/StatefulPartitionedCall?
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1967696dense_115_1967698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_19676322#
!dense_115/StatefulPartitionedCall?
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967674
flatten_16_input
dense_112_1967653
dense_112_1967655
dense_113_1967658
dense_113_1967660
dense_114_1967663
dense_114_1967665
dense_115_1967668
dense_115_1967670
identity??!dense_112/StatefulPartitionedCall?!dense_113/StatefulPartitionedCall?!dense_114/StatefulPartitionedCall?!dense_115/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCallflatten_16_input*
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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_19675322
flatten_16/PartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_112_1967653dense_112_1967655*
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
F__inference_dense_112_layer_call_and_return_conditional_losses_19675512#
!dense_112/StatefulPartitionedCall?
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1967658dense_113_1967660*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_19675782#
!dense_113/StatefulPartitionedCall?
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1967663dense_114_1967665*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_19676052#
!dense_114/StatefulPartitionedCall?
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1967668dense_115_1967670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_19676322#
!dense_115/StatefulPartitionedCall?
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameflatten_16_input
?
?
+__inference_dense_118_layer_call_fn_1968815

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
F__inference_dense_118_layer_call_and_return_conditional_losses_19678362
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
?
?
/__inference_sequential_33_layer_call_fn_1968664

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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
0__inference_autoencoder_16_layer_call_fn_1968452
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
GPU2*0,1J 8? *T
fORM
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_19681512
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
?
?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967748

inputs
dense_112_1967727
dense_112_1967729
dense_113_1967732
dense_113_1967734
dense_114_1967737
dense_114_1967739
dense_115_1967742
dense_115_1967744
identity??!dense_112/StatefulPartitionedCall?!dense_113/StatefulPartitionedCall?!dense_114/StatefulPartitionedCall?!dense_115/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCallinputs*
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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_19675322
flatten_16/PartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_112_1967727dense_112_1967729*
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
F__inference_dense_112_layer_call_and_return_conditional_losses_19675512#
!dense_112/StatefulPartitionedCall?
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1967732dense_113_1967734*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_19675782#
!dense_113/StatefulPartitionedCall?
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1967737dense_114_1967739*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_19676052#
!dense_114/StatefulPartitionedCall?
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1967742dense_115_1967744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_19676322#
!dense_115/StatefulPartitionedCall?
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_117_layer_call_and_return_conditional_losses_1967809

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
F__inference_dense_116_layer_call_and_return_conditional_losses_1968766

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_114_layer_call_and_return_conditional_losses_1968726

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
?`
?
 __inference__traced_save_1969003
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_dense_112_kernel_read_readvariableop-
)savev2_dense_112_bias_read_readvariableop/
+savev2_dense_113_kernel_read_readvariableop-
)savev2_dense_113_bias_read_readvariableop/
+savev2_dense_114_kernel_read_readvariableop-
)savev2_dense_114_bias_read_readvariableop/
+savev2_dense_115_kernel_read_readvariableop-
)savev2_dense_115_bias_read_readvariableop/
+savev2_dense_116_kernel_read_readvariableop-
)savev2_dense_116_bias_read_readvariableop/
+savev2_dense_117_kernel_read_readvariableop-
)savev2_dense_117_bias_read_readvariableop/
+savev2_dense_118_kernel_read_readvariableop-
)savev2_dense_118_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_112_kernel_m_read_readvariableop4
0savev2_adam_dense_112_bias_m_read_readvariableop6
2savev2_adam_dense_113_kernel_m_read_readvariableop4
0savev2_adam_dense_113_bias_m_read_readvariableop6
2savev2_adam_dense_114_kernel_m_read_readvariableop4
0savev2_adam_dense_114_bias_m_read_readvariableop6
2savev2_adam_dense_115_kernel_m_read_readvariableop4
0savev2_adam_dense_115_bias_m_read_readvariableop6
2savev2_adam_dense_116_kernel_m_read_readvariableop4
0savev2_adam_dense_116_bias_m_read_readvariableop6
2savev2_adam_dense_117_kernel_m_read_readvariableop4
0savev2_adam_dense_117_bias_m_read_readvariableop6
2savev2_adam_dense_118_kernel_m_read_readvariableop4
0savev2_adam_dense_118_bias_m_read_readvariableop6
2savev2_adam_dense_112_kernel_v_read_readvariableop4
0savev2_adam_dense_112_bias_v_read_readvariableop6
2savev2_adam_dense_113_kernel_v_read_readvariableop4
0savev2_adam_dense_113_bias_v_read_readvariableop6
2savev2_adam_dense_114_kernel_v_read_readvariableop4
0savev2_adam_dense_114_bias_v_read_readvariableop6
2savev2_adam_dense_115_kernel_v_read_readvariableop4
0savev2_adam_dense_115_bias_v_read_readvariableop6
2savev2_adam_dense_116_kernel_v_read_readvariableop4
0savev2_adam_dense_116_bias_v_read_readvariableop6
2savev2_adam_dense_117_kernel_v_read_readvariableop4
0savev2_adam_dense_117_bias_v_read_readvariableop6
2savev2_adam_dense_118_kernel_v_read_readvariableop4
0savev2_adam_dense_118_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_dense_112_kernel_read_readvariableop)savev2_dense_112_bias_read_readvariableop+savev2_dense_113_kernel_read_readvariableop)savev2_dense_113_bias_read_readvariableop+savev2_dense_114_kernel_read_readvariableop)savev2_dense_114_bias_read_readvariableop+savev2_dense_115_kernel_read_readvariableop)savev2_dense_115_bias_read_readvariableop+savev2_dense_116_kernel_read_readvariableop)savev2_dense_116_bias_read_readvariableop+savev2_dense_117_kernel_read_readvariableop)savev2_dense_117_bias_read_readvariableop+savev2_dense_118_kernel_read_readvariableop)savev2_dense_118_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_112_kernel_m_read_readvariableop0savev2_adam_dense_112_bias_m_read_readvariableop2savev2_adam_dense_113_kernel_m_read_readvariableop0savev2_adam_dense_113_bias_m_read_readvariableop2savev2_adam_dense_114_kernel_m_read_readvariableop0savev2_adam_dense_114_bias_m_read_readvariableop2savev2_adam_dense_115_kernel_m_read_readvariableop0savev2_adam_dense_115_bias_m_read_readvariableop2savev2_adam_dense_116_kernel_m_read_readvariableop0savev2_adam_dense_116_bias_m_read_readvariableop2savev2_adam_dense_117_kernel_m_read_readvariableop0savev2_adam_dense_117_bias_m_read_readvariableop2savev2_adam_dense_118_kernel_m_read_readvariableop0savev2_adam_dense_118_bias_m_read_readvariableop2savev2_adam_dense_112_kernel_v_read_readvariableop0savev2_adam_dense_112_bias_v_read_readvariableop2savev2_adam_dense_113_kernel_v_read_readvariableop0savev2_adam_dense_113_bias_v_read_readvariableop2savev2_adam_dense_114_kernel_v_read_readvariableop0savev2_adam_dense_114_bias_v_read_readvariableop2savev2_adam_dense_115_kernel_v_read_readvariableop0savev2_adam_dense_115_bias_v_read_readvariableop2savev2_adam_dense_116_kernel_v_read_readvariableop0savev2_adam_dense_116_bias_v_read_readvariableop2savev2_adam_dense_117_kernel_v_read_readvariableop0savev2_adam_dense_117_bias_v_read_readvariableop2savev2_adam_dense_118_kernel_v_read_readvariableop0savev2_adam_dense_118_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
??:?:	?d:d:dd:d:d::d:d:dd:d:	d?:?: : :
??:?:	?d:d:dd:d:d::d:d:dd:d:	d?:?:
??:?:	?d:d:dd:d:d::d:d:dd:d:	d?:?: 2(
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

:d: 

_output_shapes
::$ 

_output_shapes

:d: 
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

:d: 

_output_shapes
::$ 

_output_shapes

:d: 
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

:d: +

_output_shapes
::$, 

_output_shapes

:d: -
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
?	
?
F__inference_dense_114_layer_call_and_return_conditional_losses_1967605

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
?*
?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1968486

inputs,
(dense_112_matmul_readvariableop_resource-
)dense_112_biasadd_readvariableop_resource,
(dense_113_matmul_readvariableop_resource-
)dense_113_biasadd_readvariableop_resource,
(dense_114_matmul_readvariableop_resource-
)dense_114_biasadd_readvariableop_resource,
(dense_115_matmul_readvariableop_resource-
)dense_115_biasadd_readvariableop_resource
identity?? dense_112/BiasAdd/ReadVariableOp?dense_112/MatMul/ReadVariableOp? dense_113/BiasAdd/ReadVariableOp?dense_113/MatMul/ReadVariableOp? dense_114/BiasAdd/ReadVariableOp?dense_114/MatMul/ReadVariableOp? dense_115/BiasAdd/ReadVariableOp?dense_115/MatMul/ReadVariableOpu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_16/Const?
flatten_16/ReshapeReshapeinputsflatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_16/Reshape?
dense_112/MatMul/ReadVariableOpReadVariableOp(dense_112_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_112/MatMul/ReadVariableOp?
dense_112/MatMulMatMulflatten_16/Reshape:output:0'dense_112/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_112/MatMul?
 dense_112/BiasAdd/ReadVariableOpReadVariableOp)dense_112_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_112/BiasAdd/ReadVariableOp?
dense_112/BiasAddBiasAdddense_112/MatMul:product:0(dense_112/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_112/BiasAddw
dense_112/ReluReludense_112/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_112/Relu?
dense_113/MatMul/ReadVariableOpReadVariableOp(dense_113_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02!
dense_113/MatMul/ReadVariableOp?
dense_113/MatMulMatMuldense_112/Relu:activations:0'dense_113/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_113/MatMul?
 dense_113/BiasAdd/ReadVariableOpReadVariableOp)dense_113_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_113/BiasAdd/ReadVariableOp?
dense_113/BiasAddBiasAdddense_113/MatMul:product:0(dense_113/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_113/BiasAddv
dense_113/ReluReludense_113/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_113/Relu?
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02!
dense_114/MatMul/ReadVariableOp?
dense_114/MatMulMatMuldense_113/Relu:activations:0'dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_114/MatMul?
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_114/BiasAdd/ReadVariableOp?
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_114/BiasAddv
dense_114/ReluReludense_114/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_114/Relu?
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_115/MatMul/ReadVariableOp?
dense_115/MatMulMatMuldense_114/Relu:activations:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_115/MatMul?
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_115/BiasAdd/ReadVariableOp?
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_115/BiasAdd?
dense_115/SoftsignSoftsigndense_115/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_115/Softsign?
IdentityIdentity dense_115/Softsign:activations:0!^dense_112/BiasAdd/ReadVariableOp ^dense_112/MatMul/ReadVariableOp!^dense_113/BiasAdd/ReadVariableOp ^dense_113/MatMul/ReadVariableOp!^dense_114/BiasAdd/ReadVariableOp ^dense_114/MatMul/ReadVariableOp!^dense_115/BiasAdd/ReadVariableOp ^dense_115/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2D
 dense_112/BiasAdd/ReadVariableOp dense_112/BiasAdd/ReadVariableOp2B
dense_112/MatMul/ReadVariableOpdense_112/MatMul/ReadVariableOp2D
 dense_113/BiasAdd/ReadVariableOp dense_113/BiasAdd/ReadVariableOp2B
dense_113/MatMul/ReadVariableOpdense_113/MatMul/ReadVariableOp2D
 dense_114/BiasAdd/ReadVariableOp dense_114/BiasAdd/ReadVariableOp2B
dense_114/MatMul/ReadVariableOpdense_114/MatMul/ReadVariableOp2D
 dense_115/BiasAdd/ReadVariableOp dense_115/BiasAdd/ReadVariableOp2B
dense_115/MatMul/ReadVariableOpdense_115/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1968630

inputs,
(dense_116_matmul_readvariableop_resource-
)dense_116_biasadd_readvariableop_resource,
(dense_117_matmul_readvariableop_resource-
)dense_117_biasadd_readvariableop_resource,
(dense_118_matmul_readvariableop_resource-
)dense_118_biasadd_readvariableop_resource
identity?? dense_116/BiasAdd/ReadVariableOp?dense_116/MatMul/ReadVariableOp? dense_117/BiasAdd/ReadVariableOp?dense_117/MatMul/ReadVariableOp? dense_118/BiasAdd/ReadVariableOp?dense_118/MatMul/ReadVariableOp?
dense_116/MatMul/ReadVariableOpReadVariableOp(dense_116_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_116/MatMul/ReadVariableOp?
dense_116/MatMulMatMulinputs'dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_116/MatMul?
 dense_116/BiasAdd/ReadVariableOpReadVariableOp)dense_116_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_116/BiasAdd/ReadVariableOp?
dense_116/BiasAddBiasAdddense_116/MatMul:product:0(dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_116/BiasAddv
dense_116/ReluReludense_116/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_116/Relu?
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02!
dense_117/MatMul/ReadVariableOp?
dense_117/MatMulMatMuldense_116/Relu:activations:0'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_117/MatMul?
 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_117/BiasAdd/ReadVariableOp?
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_117/BiasAddv
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_117/Relu?
dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02!
dense_118/MatMul/ReadVariableOp?
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_118/MatMul?
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_118/BiasAdd/ReadVariableOp?
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_118/BiasAdd?
dense_118/SigmoidSigmoiddense_118/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_118/Sigmoidi
reshape_16/ShapeShapedense_118/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapedense_118/Sigmoid:y:0!reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_16/Reshape?
IdentityIdentityreshape_16/Reshape:output:0!^dense_116/BiasAdd/ReadVariableOp ^dense_116/MatMul/ReadVariableOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_116/BiasAdd/ReadVariableOp dense_116/BiasAdd/ReadVariableOp2B
dense_116/MatMul/ReadVariableOpdense_116/MatMul/ReadVariableOp2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_32_layer_call_fn_1968562

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
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_16_layer_call_fn_1968675

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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_19675322
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
F__inference_dense_112_layer_call_and_return_conditional_losses_1967551

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
F__inference_dense_116_layer_call_and_return_conditional_losses_1967782

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_32_layer_call_fn_1967721
flatten_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameflatten_16_input
?
?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968080
input_1
sequential_32_1968015
sequential_32_1968017
sequential_32_1968019
sequential_32_1968021
sequential_32_1968023
sequential_32_1968025
sequential_32_1968027
sequential_32_1968029
sequential_33_1968066
sequential_33_1968068
sequential_33_1968070
sequential_33_1968072
sequential_33_1968074
sequential_33_1968076
identity??%sequential_32/StatefulPartitionedCall?%sequential_33/StatefulPartitionedCall?
%sequential_32/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_32_1968015sequential_32_1968017sequential_32_1968019sequential_32_1968021sequential_32_1968023sequential_32_1968025sequential_32_1968027sequential_32_1968029*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_19677022'
%sequential_32/StatefulPartitionedCall?
%sequential_33/StatefulPartitionedCallStatefulPartitionedCall.sequential_32/StatefulPartitionedCall:output:0sequential_33_1968066sequential_33_1968068sequential_33_1968070sequential_33_1968072sequential_33_1968074sequential_33_1968076*
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
J__inference_sequential_33_layer_call_and_return_conditional_losses_19679172'
%sequential_33/StatefulPartitionedCall?
IdentityIdentity.sequential_33/StatefulPartitionedCall:output:0&^sequential_32/StatefulPartitionedCall&^sequential_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????::::::::::::::2N
%sequential_32/StatefulPartitionedCall%sequential_32/StatefulPartitionedCall2N
%sequential_33/StatefulPartitionedCall%sequential_33/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
+__inference_dense_112_layer_call_fn_1968695

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
F__inference_dense_112_layer_call_and_return_conditional_losses_19675512
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
?	
?
F__inference_dense_112_layer_call_and_return_conditional_losses_1968686

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
F__inference_dense_113_layer_call_and_return_conditional_losses_1967578

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
?
?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967917

inputs
dense_116_1967900
dense_116_1967902
dense_117_1967905
dense_117_1967907
dense_118_1967910
dense_118_1967912
identity??!dense_116/StatefulPartitionedCall?!dense_117/StatefulPartitionedCall?!dense_118/StatefulPartitionedCall?
!dense_116/StatefulPartitionedCallStatefulPartitionedCallinputsdense_116_1967900dense_116_1967902*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_116_layer_call_and_return_conditional_losses_19677822#
!dense_116/StatefulPartitionedCall?
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*dense_116/StatefulPartitionedCall:output:0dense_117_1967905dense_117_1967907*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_19678092#
!dense_117/StatefulPartitionedCall?
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_1967910dense_118_1967912*
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
F__inference_dense_118_layer_call_and_return_conditional_losses_19678362#
!dense_118/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCall*dense_118/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *P
fKRI
G__inference_reshape_16_layer_call_and_return_conditional_losses_19678652
reshape_16/PartitionedCall?
IdentityIdentity#reshape_16/PartitionedCall:output:0"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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
_tf_keras_model?{"class_name": "Autoencoder", "name": "autoencoder_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
+?&call_and_return_all_conditional_losses"?%
_tf_keras_sequential?${"class_name": "Sequential", "name": "sequential_32", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_32", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_16_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_112", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_113", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_114", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_115", "trainable": true, "dtype": "float32", "units": 17, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_32", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_16_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_112", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_113", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_114", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_115", "trainable": true, "dtype": "float32", "units": 17, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_33", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_33", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 17]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_116_input"}}, {"class_name": "Dense", "config": {"name": "dense_116", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_117", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_118", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_33", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 17]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_116_input"}}, {"class_name": "Dense", "config": {"name": "dense_116", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_117", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_118", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}}}
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
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
 bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_112", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_112", "trainable": true, "dtype": "float32", "units": 784, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

!kernel
"bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_113", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_113", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

#kernel
$bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_114", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_114", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

%kernel
&bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_115", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_115", "trainable": true, "dtype": "float32", "units": 17, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_116", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_116", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
?

)kernel
*bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_117", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_117", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

+kernel
,bias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_118", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_118", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}
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
$:"
??2dense_112/kernel
:?2dense_112/bias
#:!	?d2dense_113/kernel
:d2dense_113/bias
": dd2dense_114/kernel
:d2dense_114/bias
": d2dense_115/kernel
:2dense_115/bias
": d2dense_116/kernel
:d2dense_116/bias
": dd2dense_117/kernel
:d2dense_117/bias
#:!	d?2dense_118/kernel
:?2dense_118/bias
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
):'
??2Adam/dense_112/kernel/m
": ?2Adam/dense_112/bias/m
(:&	?d2Adam/dense_113/kernel/m
!:d2Adam/dense_113/bias/m
':%dd2Adam/dense_114/kernel/m
!:d2Adam/dense_114/bias/m
':%d2Adam/dense_115/kernel/m
!:2Adam/dense_115/bias/m
':%d2Adam/dense_116/kernel/m
!:d2Adam/dense_116/bias/m
':%dd2Adam/dense_117/kernel/m
!:d2Adam/dense_117/bias/m
(:&	d?2Adam/dense_118/kernel/m
": ?2Adam/dense_118/bias/m
):'
??2Adam/dense_112/kernel/v
": ?2Adam/dense_112/bias/v
(:&	?d2Adam/dense_113/kernel/v
!:d2Adam/dense_113/bias/v
':%dd2Adam/dense_114/kernel/v
!:d2Adam/dense_114/bias/v
':%d2Adam/dense_115/kernel/v
!:2Adam/dense_115/bias/v
':%d2Adam/dense_116/kernel/v
!:d2Adam/dense_116/bias/v
':%dd2Adam/dense_117/kernel/v
!:d2Adam/dense_117/bias/v
(:&	d?2Adam/dense_118/kernel/v
": ?2Adam/dense_118/bias/v
?2?
0__inference_autoencoder_16_layer_call_fn_1968419
0__inference_autoencoder_16_layer_call_fn_1968452
0__inference_autoencoder_16_layer_call_fn_1968182
0__inference_autoencoder_16_layer_call_fn_1968215?
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
"__inference__wrapped_model_1967522?
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
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968114
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968322
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968386
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968080?
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
/__inference_sequential_32_layer_call_fn_1967721
/__inference_sequential_32_layer_call_fn_1968541
/__inference_sequential_32_layer_call_fn_1968562
/__inference_sequential_32_layer_call_fn_1967767?
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
J__inference_sequential_32_layer_call_and_return_conditional_losses_1968486
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967649
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967674
J__inference_sequential_32_layer_call_and_return_conditional_losses_1968520?
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
/__inference_sequential_33_layer_call_fn_1968647
/__inference_sequential_33_layer_call_fn_1968664
/__inference_sequential_33_layer_call_fn_1967932
/__inference_sequential_33_layer_call_fn_1967969?
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
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967874
J__inference_sequential_33_layer_call_and_return_conditional_losses_1968630
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967894
J__inference_sequential_33_layer_call_and_return_conditional_losses_1968596?
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
%__inference_signature_wrapper_1968258input_1"?
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
,__inference_flatten_16_layer_call_fn_1968675?
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
G__inference_flatten_16_layer_call_and_return_conditional_losses_1968670?
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
+__inference_dense_112_layer_call_fn_1968695?
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
F__inference_dense_112_layer_call_and_return_conditional_losses_1968686?
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
+__inference_dense_113_layer_call_fn_1968715?
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
F__inference_dense_113_layer_call_and_return_conditional_losses_1968706?
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
+__inference_dense_114_layer_call_fn_1968735?
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
F__inference_dense_114_layer_call_and_return_conditional_losses_1968726?
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
+__inference_dense_115_layer_call_fn_1968755?
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
F__inference_dense_115_layer_call_and_return_conditional_losses_1968746?
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
+__inference_dense_116_layer_call_fn_1968775?
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
F__inference_dense_116_layer_call_and_return_conditional_losses_1968766?
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
+__inference_dense_117_layer_call_fn_1968795?
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
F__inference_dense_117_layer_call_and_return_conditional_losses_1968786?
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
+__inference_dense_118_layer_call_fn_1968815?
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
F__inference_dense_118_layer_call_and_return_conditional_losses_1968806?
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
,__inference_reshape_16_layer_call_fn_1968833?
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
G__inference_reshape_16_layer_call_and_return_conditional_losses_1968828?
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
"__inference__wrapped_model_1967522 !"#$%&'()*+,4?1
*?'
%?"
input_1?????????
? "7?4
2
output_1&?#
output_1??????????
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968080u !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p
? ")?&
?
0?????????
? ?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968114u !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p 
? ")?&
?
0?????????
? ?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968322o !"#$%&'()*+,2?/
(?%
?
x?????????
p
? ")?&
?
0?????????
? ?
K__inference_autoencoder_16_layer_call_and_return_conditional_losses_1968386o !"#$%&'()*+,2?/
(?%
?
x?????????
p 
? ")?&
?
0?????????
? ?
0__inference_autoencoder_16_layer_call_fn_1968182h !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p
? "???????????
0__inference_autoencoder_16_layer_call_fn_1968215h !"#$%&'()*+,8?5
.?+
%?"
input_1?????????
p 
? "???????????
0__inference_autoencoder_16_layer_call_fn_1968419b !"#$%&'()*+,2?/
(?%
?
x?????????
p
? "???????????
0__inference_autoencoder_16_layer_call_fn_1968452b !"#$%&'()*+,2?/
(?%
?
x?????????
p 
? "???????????
F__inference_dense_112_layer_call_and_return_conditional_losses_1968686^ 0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_112_layer_call_fn_1968695Q 0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_113_layer_call_and_return_conditional_losses_1968706]!"0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? 
+__inference_dense_113_layer_call_fn_1968715P!"0?-
&?#
!?
inputs??????????
? "??????????d?
F__inference_dense_114_layer_call_and_return_conditional_losses_1968726\#$/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ~
+__inference_dense_114_layer_call_fn_1968735O#$/?,
%?"
 ?
inputs?????????d
? "??????????d?
F__inference_dense_115_layer_call_and_return_conditional_losses_1968746\%&/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? ~
+__inference_dense_115_layer_call_fn_1968755O%&/?,
%?"
 ?
inputs?????????d
? "???????????
F__inference_dense_116_layer_call_and_return_conditional_losses_1968766\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????d
? ~
+__inference_dense_116_layer_call_fn_1968775O'(/?,
%?"
 ?
inputs?????????
? "??????????d?
F__inference_dense_117_layer_call_and_return_conditional_losses_1968786\)*/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ~
+__inference_dense_117_layer_call_fn_1968795O)*/?,
%?"
 ?
inputs?????????d
? "??????????d?
F__inference_dense_118_layer_call_and_return_conditional_losses_1968806]+,/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? 
+__inference_dense_118_layer_call_fn_1968815P+,/?,
%?"
 ?
inputs?????????d
? "????????????
G__inference_flatten_16_layer_call_and_return_conditional_losses_1968670]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_flatten_16_layer_call_fn_1968675P3?0
)?&
$?!
inputs?????????
? "????????????
G__inference_reshape_16_layer_call_and_return_conditional_losses_1968828]0?-
&?#
!?
inputs??????????
? ")?&
?
0?????????
? ?
,__inference_reshape_16_layer_call_fn_1968833P0?-
&?#
!?
inputs??????????
? "???????????
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967649x !"#$%&E?B
;?8
.?+
flatten_16_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1967674x !"#$%&E?B
;?8
.?+
flatten_16_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1968486n !"#$%&;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_32_layer_call_and_return_conditional_losses_1968520n !"#$%&;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_32_layer_call_fn_1967721k !"#$%&E?B
;?8
.?+
flatten_16_input?????????
p

 
? "???????????
/__inference_sequential_32_layer_call_fn_1967767k !"#$%&E?B
;?8
.?+
flatten_16_input?????????
p 

 
? "???????????
/__inference_sequential_32_layer_call_fn_1968541a !"#$%&;?8
1?.
$?!
inputs?????????
p

 
? "???????????
/__inference_sequential_32_layer_call_fn_1968562a !"#$%&;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967874u'()*+,@?=
6?3
)?&
dense_116_input?????????
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1967894u'()*+,@?=
6?3
)?&
dense_116_input?????????
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1968596l'()*+,7?4
-?*
 ?
inputs?????????
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_33_layer_call_and_return_conditional_losses_1968630l'()*+,7?4
-?*
 ?
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
/__inference_sequential_33_layer_call_fn_1967932h'()*+,@?=
6?3
)?&
dense_116_input?????????
p

 
? "???????????
/__inference_sequential_33_layer_call_fn_1967969h'()*+,@?=
6?3
)?&
dense_116_input?????????
p 

 
? "???????????
/__inference_sequential_33_layer_call_fn_1968647_'()*+,7?4
-?*
 ?
inputs?????????
p

 
? "???????????
/__inference_sequential_33_layer_call_fn_1968664_'()*+,7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
%__inference_signature_wrapper_1968258? !"#$%&'()*+,??<
? 
5?2
0
input_1%?"
input_1?????????"7?4
2
output_1&?#
output_1?????????