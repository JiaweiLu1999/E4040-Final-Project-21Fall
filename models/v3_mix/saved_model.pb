Ê©<
É
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8,

input_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameinput_conv/kernel

%input_conv/kernel/Read/ReadVariableOpReadVariableOpinput_conv/kernel*&
_output_shapes
:@*
dtype0
v
input_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameinput_conv/bias
o
#input_conv/bias/Read/ReadVariableOpReadVariableOpinput_conv/bias*
_output_shapes
:@*
dtype0

downsampler_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_namedownsampler_1/kernel

(downsampler_1/kernel/Read/ReadVariableOpReadVariableOpdownsampler_1/kernel*&
_output_shapes
:@@*
dtype0
|
downsampler_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedownsampler_1/bias
u
&downsampler_1/bias/Read/ReadVariableOpReadVariableOpdownsampler_1/bias*
_output_shapes
:@*
dtype0

resblock_part1_1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_1_conv1/kernel

1resblock_part1_1_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_1_conv1/bias

/resblock_part1_1_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_1_conv2/kernel

1resblock_part1_1_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_1_conv2/bias

/resblock_part1_1_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part1_2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_2_conv1/kernel

1resblock_part1_2_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_2_conv1/bias

/resblock_part1_2_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_2_conv2/kernel

1resblock_part1_2_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_2_conv2/bias

/resblock_part1_2_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part1_3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_3_conv1/kernel

1resblock_part1_3_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_3_conv1/bias

/resblock_part1_3_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_3_conv2/kernel

1resblock_part1_3_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_3_conv2/bias

/resblock_part1_3_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part1_4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_4_conv1/kernel

1resblock_part1_4_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_4_conv1/bias

/resblock_part1_4_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_4_conv2/kernel

1resblock_part1_4_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_4_conv2/bias

/resblock_part1_4_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv2/bias*
_output_shapes
:@*
dtype0

downsampler_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_namedownsampler_2/kernel

(downsampler_2/kernel/Read/ReadVariableOpReadVariableOpdownsampler_2/kernel*&
_output_shapes
:@@*
dtype0
|
downsampler_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedownsampler_2/bias
u
&downsampler_2/bias/Read/ReadVariableOpReadVariableOpdownsampler_2/bias*
_output_shapes
:@*
dtype0

resblock_part2_1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_1_conv1/kernel

1resblock_part2_1_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_1_conv1/bias

/resblock_part2_1_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_1_conv2/kernel

1resblock_part2_1_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_1_conv2/bias

/resblock_part2_1_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_2_conv1/kernel

1resblock_part2_2_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_2_conv1/bias

/resblock_part2_2_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_2_conv2/kernel

1resblock_part2_2_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_2_conv2/bias

/resblock_part2_2_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_3_conv1/kernel

1resblock_part2_3_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_3_conv1/bias

/resblock_part2_3_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_3_conv2/kernel

1resblock_part2_3_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_3_conv2/bias

/resblock_part2_3_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_4_conv1/kernel

1resblock_part2_4_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_4_conv1/bias

/resblock_part2_4_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_4_conv2/kernel

1resblock_part2_4_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_4_conv2/bias

/resblock_part2_4_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_5_conv1/kernel

1resblock_part2_5_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_5_conv1/bias

/resblock_part2_5_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_5_conv2/kernel

1resblock_part2_5_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_5_conv2/bias

/resblock_part2_5_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_6_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_6_conv1/kernel

1resblock_part2_6_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_6_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_6_conv1/bias

/resblock_part2_6_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_6_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_6_conv2/kernel

1resblock_part2_6_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_6_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_6_conv2/bias

/resblock_part2_6_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_7_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_7_conv1/kernel

1resblock_part2_7_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_7_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_7_conv1/bias

/resblock_part2_7_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_7_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_7_conv2/kernel

1resblock_part2_7_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_7_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_7_conv2/bias

/resblock_part2_7_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_8_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_8_conv1/kernel

1resblock_part2_8_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_8_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_8_conv1/bias

/resblock_part2_8_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_8_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_8_conv2/kernel

1resblock_part2_8_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_8_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_8_conv2/bias

/resblock_part2_8_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv2/bias*
_output_shapes
:@*
dtype0

upsampler_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameupsampler_1/kernel

&upsampler_1/kernel/Read/ReadVariableOpReadVariableOpupsampler_1/kernel*'
_output_shapes
:@*
dtype0
y
upsampler_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameupsampler_1/bias
r
$upsampler_1/bias/Read/ReadVariableOpReadVariableOpupsampler_1/bias*
_output_shapes	
:*
dtype0

resblock_part3_1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_1_conv1/kernel

1resblock_part3_1_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_1_conv1/bias

/resblock_part3_1_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_1_conv2/kernel

1resblock_part3_1_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_1_conv2/bias

/resblock_part3_1_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part3_2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_2_conv1/kernel

1resblock_part3_2_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_2_conv1/bias

/resblock_part3_2_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_2_conv2/kernel

1resblock_part3_2_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_2_conv2/bias

/resblock_part3_2_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part3_3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_3_conv1/kernel

1resblock_part3_3_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_3_conv1/bias

/resblock_part3_3_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_3_conv2/kernel

1resblock_part3_3_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_3_conv2/bias

/resblock_part3_3_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part3_4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_4_conv1/kernel

1resblock_part3_4_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_4_conv1/bias

/resblock_part3_4_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_4_conv2/kernel

1resblock_part3_4_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_4_conv2/bias

/resblock_part3_4_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv2/bias*
_output_shapes
:@*
dtype0

extra_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameextra_conv/kernel

%extra_conv/kernel/Read/ReadVariableOpReadVariableOpextra_conv/kernel*&
_output_shapes
:@@*
dtype0
v
extra_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameextra_conv/bias
o
#extra_conv/bias/Read/ReadVariableOpReadVariableOpextra_conv/bias*
_output_shapes
:@*
dtype0

upsampler_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameupsampler_2/kernel

&upsampler_2/kernel/Read/ReadVariableOpReadVariableOpupsampler_2/kernel*'
_output_shapes
:@*
dtype0
y
upsampler_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameupsampler_2/bias
r
$upsampler_2/bias/Read/ReadVariableOpReadVariableOpupsampler_2/bias*
_output_shapes	
:*
dtype0

output_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameoutput_conv/kernel

&output_conv/kernel/Read/ReadVariableOpReadVariableOpoutput_conv/kernel*&
_output_shapes
:@*
dtype0
x
output_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameoutput_conv/bias
q
$output_conv/bias/Read/ReadVariableOpReadVariableOpoutput_conv/bias*
_output_shapes
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *  ?

NoOpNoOp
¯û
Const_16Const"/device:CPU:0*
_output_shapes
: *
dtype0*æú
valueÛúB×ú BÏú

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer-24
layer_with_weights-10
layer-25
layer_with_weights-11
layer-26
layer-27
layer_with_weights-12
layer-28
layer-29
layer-30
 layer_with_weights-13
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$layer-35
%layer_with_weights-15
%layer-36
&layer-37
'layer_with_weights-16
'layer-38
(layer-39
)layer-40
*layer_with_weights-17
*layer-41
+layer-42
,layer_with_weights-18
,layer-43
-layer-44
.layer-45
/layer_with_weights-19
/layer-46
0layer-47
1layer_with_weights-20
1layer-48
2layer-49
3layer-50
4layer_with_weights-21
4layer-51
5layer-52
6layer_with_weights-22
6layer-53
7layer-54
8layer-55
9layer_with_weights-23
9layer-56
:layer-57
;layer_with_weights-24
;layer-58
<layer-59
=layer-60
>layer_with_weights-25
>layer-61
?layer-62
@layer_with_weights-26
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-27
Clayer-66
Dlayer-67
Elayer_with_weights-28
Elayer-68
Flayer-69
Glayer_with_weights-29
Glayer-70
Hlayer-71
Ilayer-72
Jlayer_with_weights-30
Jlayer-73
Klayer-74
Llayer_with_weights-31
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-32
Olayer-78
Player-79
Qlayer_with_weights-33
Qlayer-80
Rlayer-81
Slayer-82
Tlayer_with_weights-34
Tlayer-83
Ulayer-84
Vlayer_with_weights-35
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-36
Ylayer-88
Zlayer-89
[layer_with_weights-37
[layer-90
\layer-91
]layer_with_weights-38
]layer-92
^trainable_variables
_regularization_losses
`	variables
a	keras_api
b
signatures
 
h

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
R
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
h

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
R
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
k

}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
 	keras_api
n
¡kernel
	¢bias
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api

§	keras_api

¨	keras_api
n
©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
V
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
n
³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api

¹	keras_api

º	keras_api
V
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
n
¿kernel
	Àbias
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
n
Åkernel
	Æbias
Ç	variables
Ètrainable_variables
Éregularization_losses
Ê	keras_api
V
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
n
Ïkernel
	Ðbias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api

Õ	keras_api

Ö	keras_api
n
×kernel
	Øbias
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
V
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
n
ákernel
	âbias
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api

ç	keras_api

è	keras_api
n
ékernel
	êbias
ë	variables
ìtrainable_variables
íregularization_losses
î	keras_api
V
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
n
ókernel
	ôbias
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api

ù	keras_api

ú	keras_api
n
ûkernel
	übias
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	 bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
V
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
n
©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api

¯	keras_api

°	keras_api
n
±kernel
	²bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
V
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
n
»kernel
	¼bias
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api

Á	keras_api

Â	keras_api
n
Ãkernel
	Äbias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
V
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
n
Íkernel
	Îbias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api

Ó	keras_api

Ô	keras_api
n
Õkernel
	Öbias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api

Û	keras_api
n
Ükernel
	Ýbias
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
V
â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
n
ækernel
	çbias
è	variables
étrainable_variables
êregularization_losses
ë	keras_api

ì	keras_api

í	keras_api
n
îkernel
	ïbias
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
V
ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
n
økernel
	ùbias
ú	variables
ûtrainable_variables
üregularization_losses
ý	keras_api

þ	keras_api

ÿ	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
 regularization_losses
¡	keras_api

¢	keras_api

£	keras_api
n
¤kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api

ª	keras_api
n
«kernel
	¬bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api

±	keras_api
n
²kernel
	³bias
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¬
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77
 
¬
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77
²
¸metrics
¹layers
^trainable_variables
ºlayer_metrics
_regularization_losses
`	variables
 »layer_regularization_losses
¼non_trainable_variables
 
][
VARIABLE_VALUEinput_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEinput_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
²
e	variables
½layers
ftrainable_variables
¾layer_metrics
gregularization_losses
¿metrics
 Àlayer_regularization_losses
Ánon_trainable_variables
 
 
 
²
i	variables
Âlayers
jtrainable_variables
Ãlayer_metrics
kregularization_losses
Ämetrics
 Ålayer_regularization_losses
Ænon_trainable_variables
`^
VARIABLE_VALUEdownsampler_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdownsampler_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
 
²
o	variables
Çlayers
ptrainable_variables
Èlayer_metrics
qregularization_losses
Émetrics
 Êlayer_regularization_losses
Ënon_trainable_variables
ig
VARIABLE_VALUEresblock_part1_1_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_1_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
²
u	variables
Ìlayers
vtrainable_variables
Ílayer_metrics
wregularization_losses
Îmetrics
 Ïlayer_regularization_losses
Ðnon_trainable_variables
 
 
 
²
y	variables
Ñlayers
ztrainable_variables
Òlayer_metrics
{regularization_losses
Ómetrics
 Ôlayer_regularization_losses
Õnon_trainable_variables
ig
VARIABLE_VALUEresblock_part1_1_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_1_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1

}0
~1
 
´
	variables
Ölayers
trainable_variables
×layer_metrics
regularization_losses
Ømetrics
 Ùlayer_regularization_losses
Únon_trainable_variables
 
 
ig
VARIABLE_VALUEresblock_part1_2_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_2_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
Ûlayers
trainable_variables
Ülayer_metrics
regularization_losses
Ýmetrics
 Þlayer_regularization_losses
ßnon_trainable_variables
 
 
 
µ
	variables
àlayers
trainable_variables
álayer_metrics
regularization_losses
âmetrics
 ãlayer_regularization_losses
änon_trainable_variables
ig
VARIABLE_VALUEresblock_part1_2_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_2_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
ålayers
trainable_variables
ælayer_metrics
regularization_losses
çmetrics
 èlayer_regularization_losses
énon_trainable_variables
 
 
ig
VARIABLE_VALUEresblock_part1_3_conv1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_3_conv1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
êlayers
trainable_variables
ëlayer_metrics
regularization_losses
ìmetrics
 ílayer_regularization_losses
înon_trainable_variables
 
 
 
µ
	variables
ïlayers
trainable_variables
ðlayer_metrics
regularization_losses
ñmetrics
 òlayer_regularization_losses
ónon_trainable_variables
ig
VARIABLE_VALUEresblock_part1_3_conv2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_3_conv2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

¡0
¢1

¡0
¢1
 
µ
£	variables
ôlayers
¤trainable_variables
õlayer_metrics
¥regularization_losses
ömetrics
 ÷layer_regularization_losses
ønon_trainable_variables
 
 
ig
VARIABLE_VALUEresblock_part1_4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

©0
ª1

©0
ª1
 
µ
«	variables
ùlayers
¬trainable_variables
úlayer_metrics
­regularization_losses
ûmetrics
 ülayer_regularization_losses
ýnon_trainable_variables
 
 
 
µ
¯	variables
þlayers
°trainable_variables
ÿlayer_metrics
±regularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
ig
VARIABLE_VALUEresblock_part1_4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

³0
´1

³0
´1
 
µ
µ	variables
layers
¶trainable_variables
layer_metrics
·regularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
 
 
 
 
 
µ
»	variables
layers
¼trainable_variables
layer_metrics
½regularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
a_
VARIABLE_VALUEdownsampler_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdownsampler_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

¿0
À1

¿0
À1
 
µ
Á	variables
layers
Âtrainable_variables
layer_metrics
Ãregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
jh
VARIABLE_VALUEresblock_part2_1_conv1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_1_conv1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Å0
Æ1

Å0
Æ1
 
µ
Ç	variables
layers
Ètrainable_variables
layer_metrics
Éregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
 
 
 
µ
Ë	variables
layers
Ìtrainable_variables
layer_metrics
Íregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
jh
VARIABLE_VALUEresblock_part2_1_conv2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_1_conv2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Ï0
Ð1

Ï0
Ð1
 
µ
Ñ	variables
layers
Òtrainable_variables
layer_metrics
Óregularization_losses
metrics
 layer_regularization_losses
 non_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_2_conv1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_2_conv1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

×0
Ø1

×0
Ø1
 
µ
Ù	variables
¡layers
Útrainable_variables
¢layer_metrics
Ûregularization_losses
£metrics
 ¤layer_regularization_losses
¥non_trainable_variables
 
 
 
µ
Ý	variables
¦layers
Þtrainable_variables
§layer_metrics
ßregularization_losses
¨metrics
 ©layer_regularization_losses
ªnon_trainable_variables
jh
VARIABLE_VALUEresblock_part2_2_conv2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_2_conv2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

á0
â1

á0
â1
 
µ
ã	variables
«layers
ätrainable_variables
¬layer_metrics
åregularization_losses
­metrics
 ®layer_regularization_losses
¯non_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_3_conv1/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_3_conv1/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

é0
ê1

é0
ê1
 
µ
ë	variables
°layers
ìtrainable_variables
±layer_metrics
íregularization_losses
²metrics
 ³layer_regularization_losses
´non_trainable_variables
 
 
 
µ
ï	variables
µlayers
ðtrainable_variables
¶layer_metrics
ñregularization_losses
·metrics
 ¸layer_regularization_losses
¹non_trainable_variables
jh
VARIABLE_VALUEresblock_part2_3_conv2/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_3_conv2/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

ó0
ô1

ó0
ô1
 
µ
õ	variables
ºlayers
ötrainable_variables
»layer_metrics
÷regularization_losses
¼metrics
 ½layer_regularization_losses
¾non_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_4_conv1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_4_conv1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

û0
ü1

û0
ü1
 
µ
ý	variables
¿layers
þtrainable_variables
Àlayer_metrics
ÿregularization_losses
Ámetrics
 Âlayer_regularization_losses
Ãnon_trainable_variables
 
 
 
µ
	variables
Älayers
trainable_variables
Ålayer_metrics
regularization_losses
Æmetrics
 Çlayer_regularization_losses
Ènon_trainable_variables
jh
VARIABLE_VALUEresblock_part2_4_conv2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_4_conv2/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
Élayers
trainable_variables
Êlayer_metrics
regularization_losses
Ëmetrics
 Ìlayer_regularization_losses
Ínon_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_5_conv1/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_5_conv1/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
Îlayers
trainable_variables
Ïlayer_metrics
regularization_losses
Ðmetrics
 Ñlayer_regularization_losses
Ònon_trainable_variables
 
 
 
µ
	variables
Ólayers
trainable_variables
Ôlayer_metrics
regularization_losses
Õmetrics
 Ölayer_regularization_losses
×non_trainable_variables
jh
VARIABLE_VALUEresblock_part2_5_conv2/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_5_conv2/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
Ølayers
trainable_variables
Ùlayer_metrics
regularization_losses
Úmetrics
 Ûlayer_regularization_losses
Ünon_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_6_conv1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_6_conv1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
µ
¡	variables
Ýlayers
¢trainable_variables
Þlayer_metrics
£regularization_losses
ßmetrics
 àlayer_regularization_losses
ánon_trainable_variables
 
 
 
µ
¥	variables
âlayers
¦trainable_variables
ãlayer_metrics
§regularization_losses
ämetrics
 ålayer_regularization_losses
ænon_trainable_variables
jh
VARIABLE_VALUEresblock_part2_6_conv2/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_6_conv2/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

©0
ª1

©0
ª1
 
µ
«	variables
çlayers
¬trainable_variables
èlayer_metrics
­regularization_losses
émetrics
 êlayer_regularization_losses
ënon_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_7_conv1/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_7_conv1/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE

±0
²1

±0
²1
 
µ
³	variables
ìlayers
´trainable_variables
ílayer_metrics
µregularization_losses
îmetrics
 ïlayer_regularization_losses
ðnon_trainable_variables
 
 
 
µ
·	variables
ñlayers
¸trainable_variables
òlayer_metrics
¹regularization_losses
ómetrics
 ôlayer_regularization_losses
õnon_trainable_variables
jh
VARIABLE_VALUEresblock_part2_7_conv2/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_7_conv2/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE

»0
¼1

»0
¼1
 
µ
½	variables
ölayers
¾trainable_variables
÷layer_metrics
¿regularization_losses
ømetrics
 ùlayer_regularization_losses
únon_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part2_8_conv1/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_8_conv1/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE

Ã0
Ä1

Ã0
Ä1
 
µ
Å	variables
ûlayers
Ætrainable_variables
ülayer_metrics
Çregularization_losses
ýmetrics
 þlayer_regularization_losses
ÿnon_trainable_variables
 
 
 
µ
É	variables
layers
Êtrainable_variables
layer_metrics
Ëregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
jh
VARIABLE_VALUEresblock_part2_8_conv2/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_8_conv2/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE

Í0
Î1

Í0
Î1
 
µ
Ï	variables
layers
Ðtrainable_variables
layer_metrics
Ñregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
 
 
_]
VARIABLE_VALUEupsampler_1/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEupsampler_1/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE

Õ0
Ö1

Õ0
Ö1
 
µ
×	variables
layers
Øtrainable_variables
layer_metrics
Ùregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
 
jh
VARIABLE_VALUEresblock_part3_1_conv1/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_1_conv1/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE

Ü0
Ý1

Ü0
Ý1
 
µ
Þ	variables
layers
ßtrainable_variables
layer_metrics
àregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
 
 
 
µ
â	variables
layers
ãtrainable_variables
layer_metrics
äregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
jh
VARIABLE_VALUEresblock_part3_1_conv2/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_1_conv2/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE

æ0
ç1

æ0
ç1
 
µ
è	variables
layers
étrainable_variables
layer_metrics
êregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part3_2_conv1/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_2_conv1/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE

î0
ï1

î0
ï1
 
µ
ð	variables
layers
ñtrainable_variables
layer_metrics
òregularization_losses
 metrics
 ¡layer_regularization_losses
¢non_trainable_variables
 
 
 
µ
ô	variables
£layers
õtrainable_variables
¤layer_metrics
öregularization_losses
¥metrics
 ¦layer_regularization_losses
§non_trainable_variables
jh
VARIABLE_VALUEresblock_part3_2_conv2/kernel7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_2_conv2/bias5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUE

ø0
ù1

ø0
ù1
 
µ
ú	variables
¨layers
ûtrainable_variables
©layer_metrics
üregularization_losses
ªmetrics
 «layer_regularization_losses
¬non_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part3_3_conv1/kernel7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_3_conv1/bias5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
­layers
trainable_variables
®layer_metrics
regularization_losses
¯metrics
 °layer_regularization_losses
±non_trainable_variables
 
 
 
µ
	variables
²layers
trainable_variables
³layer_metrics
regularization_losses
´metrics
 µlayer_regularization_losses
¶non_trainable_variables
jh
VARIABLE_VALUEresblock_part3_3_conv2/kernel7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_3_conv2/bias5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
·layers
trainable_variables
¸layer_metrics
regularization_losses
¹metrics
 ºlayer_regularization_losses
»non_trainable_variables
 
 
jh
VARIABLE_VALUEresblock_part3_4_conv1/kernel7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_4_conv1/bias5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
¼layers
trainable_variables
½layer_metrics
regularization_losses
¾metrics
 ¿layer_regularization_losses
Ànon_trainable_variables
 
 
 
µ
	variables
Álayers
trainable_variables
Âlayer_metrics
regularization_losses
Ãmetrics
 Älayer_regularization_losses
Ånon_trainable_variables
jh
VARIABLE_VALUEresblock_part3_4_conv2/kernel7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_4_conv2/bias5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
	variables
Ælayers
trainable_variables
Çlayer_metrics
 regularization_losses
Èmetrics
 Élayer_regularization_losses
Ênon_trainable_variables
 
 
^\
VARIABLE_VALUEextra_conv/kernel7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEextra_conv/bias5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUE

¤0
¥1

¤0
¥1
 
µ
¦	variables
Ëlayers
§trainable_variables
Ìlayer_metrics
¨regularization_losses
Ímetrics
 Îlayer_regularization_losses
Ïnon_trainable_variables
 
_]
VARIABLE_VALUEupsampler_2/kernel7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEupsampler_2/bias5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUE

«0
¬1

«0
¬1
 
µ
­	variables
Ðlayers
®trainable_variables
Ñlayer_metrics
¯regularization_losses
Òmetrics
 Ólayer_regularization_losses
Ônon_trainable_variables
 
_]
VARIABLE_VALUEoutput_conv/kernel7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_conv/bias5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE

²0
³1

²0
³1
 
µ
´	variables
Õlayers
µtrainable_variables
Ölayer_metrics
¶regularization_losses
×metrics
 Ølayer_regularization_losses
Ùnon_trainable_variables
 
Þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89
[90
\91
]92
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_layerPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerinput_conv/kernelinput_conv/biasdownsampler_1/kerneldownsampler_1/biasresblock_part1_1_conv1/kernelresblock_part1_1_conv1/biasresblock_part1_1_conv2/kernelresblock_part1_1_conv2/biasConstresblock_part1_2_conv1/kernelresblock_part1_2_conv1/biasresblock_part1_2_conv2/kernelresblock_part1_2_conv2/biasConst_1resblock_part1_3_conv1/kernelresblock_part1_3_conv1/biasresblock_part1_3_conv2/kernelresblock_part1_3_conv2/biasConst_2resblock_part1_4_conv1/kernelresblock_part1_4_conv1/biasresblock_part1_4_conv2/kernelresblock_part1_4_conv2/biasConst_3downsampler_2/kerneldownsampler_2/biasresblock_part2_1_conv1/kernelresblock_part2_1_conv1/biasresblock_part2_1_conv2/kernelresblock_part2_1_conv2/biasConst_4resblock_part2_2_conv1/kernelresblock_part2_2_conv1/biasresblock_part2_2_conv2/kernelresblock_part2_2_conv2/biasConst_5resblock_part2_3_conv1/kernelresblock_part2_3_conv1/biasresblock_part2_3_conv2/kernelresblock_part2_3_conv2/biasConst_6resblock_part2_4_conv1/kernelresblock_part2_4_conv1/biasresblock_part2_4_conv2/kernelresblock_part2_4_conv2/biasConst_7resblock_part2_5_conv1/kernelresblock_part2_5_conv1/biasresblock_part2_5_conv2/kernelresblock_part2_5_conv2/biasConst_8resblock_part2_6_conv1/kernelresblock_part2_6_conv1/biasresblock_part2_6_conv2/kernelresblock_part2_6_conv2/biasConst_9resblock_part2_7_conv1/kernelresblock_part2_7_conv1/biasresblock_part2_7_conv2/kernelresblock_part2_7_conv2/biasConst_10resblock_part2_8_conv1/kernelresblock_part2_8_conv1/biasresblock_part2_8_conv2/kernelresblock_part2_8_conv2/biasConst_11upsampler_1/kernelupsampler_1/biasresblock_part3_1_conv1/kernelresblock_part3_1_conv1/biasresblock_part3_1_conv2/kernelresblock_part3_1_conv2/biasConst_12resblock_part3_2_conv1/kernelresblock_part3_2_conv1/biasresblock_part3_2_conv2/kernelresblock_part3_2_conv2/biasConst_13resblock_part3_3_conv1/kernelresblock_part3_3_conv1/biasresblock_part3_3_conv2/kernelresblock_part3_3_conv2/biasConst_14resblock_part3_4_conv1/kernelresblock_part3_4_conv1/biasresblock_part3_4_conv2/kernelresblock_part3_4_conv2/biasConst_15extra_conv/kernelextra_conv/biasupsampler_2/kernelupsampler_2/biasoutput_conv/kerneloutput_conv/bias*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_5642
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%input_conv/kernel/Read/ReadVariableOp#input_conv/bias/Read/ReadVariableOp(downsampler_1/kernel/Read/ReadVariableOp&downsampler_1/bias/Read/ReadVariableOp1resblock_part1_1_conv1/kernel/Read/ReadVariableOp/resblock_part1_1_conv1/bias/Read/ReadVariableOp1resblock_part1_1_conv2/kernel/Read/ReadVariableOp/resblock_part1_1_conv2/bias/Read/ReadVariableOp1resblock_part1_2_conv1/kernel/Read/ReadVariableOp/resblock_part1_2_conv1/bias/Read/ReadVariableOp1resblock_part1_2_conv2/kernel/Read/ReadVariableOp/resblock_part1_2_conv2/bias/Read/ReadVariableOp1resblock_part1_3_conv1/kernel/Read/ReadVariableOp/resblock_part1_3_conv1/bias/Read/ReadVariableOp1resblock_part1_3_conv2/kernel/Read/ReadVariableOp/resblock_part1_3_conv2/bias/Read/ReadVariableOp1resblock_part1_4_conv1/kernel/Read/ReadVariableOp/resblock_part1_4_conv1/bias/Read/ReadVariableOp1resblock_part1_4_conv2/kernel/Read/ReadVariableOp/resblock_part1_4_conv2/bias/Read/ReadVariableOp(downsampler_2/kernel/Read/ReadVariableOp&downsampler_2/bias/Read/ReadVariableOp1resblock_part2_1_conv1/kernel/Read/ReadVariableOp/resblock_part2_1_conv1/bias/Read/ReadVariableOp1resblock_part2_1_conv2/kernel/Read/ReadVariableOp/resblock_part2_1_conv2/bias/Read/ReadVariableOp1resblock_part2_2_conv1/kernel/Read/ReadVariableOp/resblock_part2_2_conv1/bias/Read/ReadVariableOp1resblock_part2_2_conv2/kernel/Read/ReadVariableOp/resblock_part2_2_conv2/bias/Read/ReadVariableOp1resblock_part2_3_conv1/kernel/Read/ReadVariableOp/resblock_part2_3_conv1/bias/Read/ReadVariableOp1resblock_part2_3_conv2/kernel/Read/ReadVariableOp/resblock_part2_3_conv2/bias/Read/ReadVariableOp1resblock_part2_4_conv1/kernel/Read/ReadVariableOp/resblock_part2_4_conv1/bias/Read/ReadVariableOp1resblock_part2_4_conv2/kernel/Read/ReadVariableOp/resblock_part2_4_conv2/bias/Read/ReadVariableOp1resblock_part2_5_conv1/kernel/Read/ReadVariableOp/resblock_part2_5_conv1/bias/Read/ReadVariableOp1resblock_part2_5_conv2/kernel/Read/ReadVariableOp/resblock_part2_5_conv2/bias/Read/ReadVariableOp1resblock_part2_6_conv1/kernel/Read/ReadVariableOp/resblock_part2_6_conv1/bias/Read/ReadVariableOp1resblock_part2_6_conv2/kernel/Read/ReadVariableOp/resblock_part2_6_conv2/bias/Read/ReadVariableOp1resblock_part2_7_conv1/kernel/Read/ReadVariableOp/resblock_part2_7_conv1/bias/Read/ReadVariableOp1resblock_part2_7_conv2/kernel/Read/ReadVariableOp/resblock_part2_7_conv2/bias/Read/ReadVariableOp1resblock_part2_8_conv1/kernel/Read/ReadVariableOp/resblock_part2_8_conv1/bias/Read/ReadVariableOp1resblock_part2_8_conv2/kernel/Read/ReadVariableOp/resblock_part2_8_conv2/bias/Read/ReadVariableOp&upsampler_1/kernel/Read/ReadVariableOp$upsampler_1/bias/Read/ReadVariableOp1resblock_part3_1_conv1/kernel/Read/ReadVariableOp/resblock_part3_1_conv1/bias/Read/ReadVariableOp1resblock_part3_1_conv2/kernel/Read/ReadVariableOp/resblock_part3_1_conv2/bias/Read/ReadVariableOp1resblock_part3_2_conv1/kernel/Read/ReadVariableOp/resblock_part3_2_conv1/bias/Read/ReadVariableOp1resblock_part3_2_conv2/kernel/Read/ReadVariableOp/resblock_part3_2_conv2/bias/Read/ReadVariableOp1resblock_part3_3_conv1/kernel/Read/ReadVariableOp/resblock_part3_3_conv1/bias/Read/ReadVariableOp1resblock_part3_3_conv2/kernel/Read/ReadVariableOp/resblock_part3_3_conv2/bias/Read/ReadVariableOp1resblock_part3_4_conv1/kernel/Read/ReadVariableOp/resblock_part3_4_conv1/bias/Read/ReadVariableOp1resblock_part3_4_conv2/kernel/Read/ReadVariableOp/resblock_part3_4_conv2/bias/Read/ReadVariableOp%extra_conv/kernel/Read/ReadVariableOp#extra_conv/bias/Read/ReadVariableOp&upsampler_2/kernel/Read/ReadVariableOp$upsampler_2/bias/Read/ReadVariableOp&output_conv/kernel/Read/ReadVariableOp$output_conv/bias/Read/ReadVariableOpConst_16*[
TinT
R2P*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_7820
ó
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_conv/kernelinput_conv/biasdownsampler_1/kerneldownsampler_1/biasresblock_part1_1_conv1/kernelresblock_part1_1_conv1/biasresblock_part1_1_conv2/kernelresblock_part1_1_conv2/biasresblock_part1_2_conv1/kernelresblock_part1_2_conv1/biasresblock_part1_2_conv2/kernelresblock_part1_2_conv2/biasresblock_part1_3_conv1/kernelresblock_part1_3_conv1/biasresblock_part1_3_conv2/kernelresblock_part1_3_conv2/biasresblock_part1_4_conv1/kernelresblock_part1_4_conv1/biasresblock_part1_4_conv2/kernelresblock_part1_4_conv2/biasdownsampler_2/kerneldownsampler_2/biasresblock_part2_1_conv1/kernelresblock_part2_1_conv1/biasresblock_part2_1_conv2/kernelresblock_part2_1_conv2/biasresblock_part2_2_conv1/kernelresblock_part2_2_conv1/biasresblock_part2_2_conv2/kernelresblock_part2_2_conv2/biasresblock_part2_3_conv1/kernelresblock_part2_3_conv1/biasresblock_part2_3_conv2/kernelresblock_part2_3_conv2/biasresblock_part2_4_conv1/kernelresblock_part2_4_conv1/biasresblock_part2_4_conv2/kernelresblock_part2_4_conv2/biasresblock_part2_5_conv1/kernelresblock_part2_5_conv1/biasresblock_part2_5_conv2/kernelresblock_part2_5_conv2/biasresblock_part2_6_conv1/kernelresblock_part2_6_conv1/biasresblock_part2_6_conv2/kernelresblock_part2_6_conv2/biasresblock_part2_7_conv1/kernelresblock_part2_7_conv1/biasresblock_part2_7_conv2/kernelresblock_part2_7_conv2/biasresblock_part2_8_conv1/kernelresblock_part2_8_conv1/biasresblock_part2_8_conv2/kernelresblock_part2_8_conv2/biasupsampler_1/kernelupsampler_1/biasresblock_part3_1_conv1/kernelresblock_part3_1_conv1/biasresblock_part3_1_conv2/kernelresblock_part3_1_conv2/biasresblock_part3_2_conv1/kernelresblock_part3_2_conv1/biasresblock_part3_2_conv2/kernelresblock_part3_2_conv2/biasresblock_part3_3_conv1/kernelresblock_part3_3_conv1/biasresblock_part3_3_conv2/kernelresblock_part3_3_conv2/biasresblock_part3_4_conv1/kernelresblock_part3_4_conv1/biasresblock_part3_4_conv2/kernelresblock_part3_4_conv2/biasextra_conv/kernelextra_conv/biasupsampler_2/kernelupsampler_2/biasoutput_conv/kerneloutput_conv/bias*Z
TinS
Q2O*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_8064¤(
¤

é
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_3818

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_3_conv2_layer_call_fn_7039

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_35172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_7241

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¢

Ý
D__inference_input_conv_layer_call_and_return_conditional_losses_2990

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

5__inference_resblock_part1_3_conv2_layer_call_fn_6828

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_32182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_3268

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6919

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_7_conv1_layer_call_fn_7202

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_37502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¢

Ý
D__inference_input_conv_layer_call_and_return_conditional_losses_6656

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
d
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2957

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_7270

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_8_conv2_layer_call_fn_7279

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_38572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_7063

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


Þ
E__inference_upsampler_1_layer_call_and_return_conditional_losses_7289

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp 
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_3635

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_3478

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Ä"
³
"__inference_signature_wrapper_5642
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_29502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
 

5__inference_resblock_part3_4_conv1_layer_call_fn_7461

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_41172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6905

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_4_conv2_layer_call_fn_7490

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_41562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_3585

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_7418

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_7466

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_7174

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¦

à
G__inference_downsampler_1_layer_call_and_return_conditional_losses_6675

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

~
)__inference_input_conv_layer_call_fn_6665

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_29902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_6723

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_7193

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_2_conv2_layer_call_fn_6991

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_34492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_7337

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_3200

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_1_conv1_layer_call_fn_7317

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_39132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_4_conv1_layer_call_fn_7058

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_35462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_4002

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_3082

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_7078

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_7097

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_7_relu1_layer_call_fn_7212

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37712
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_7404

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_3614

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
îÆ
Á-
 __inference__traced_restore_8064
file_prefix&
"assignvariableop_input_conv_kernel&
"assignvariableop_1_input_conv_bias+
'assignvariableop_2_downsampler_1_kernel)
%assignvariableop_3_downsampler_1_bias4
0assignvariableop_4_resblock_part1_1_conv1_kernel2
.assignvariableop_5_resblock_part1_1_conv1_bias4
0assignvariableop_6_resblock_part1_1_conv2_kernel2
.assignvariableop_7_resblock_part1_1_conv2_bias4
0assignvariableop_8_resblock_part1_2_conv1_kernel2
.assignvariableop_9_resblock_part1_2_conv1_bias5
1assignvariableop_10_resblock_part1_2_conv2_kernel3
/assignvariableop_11_resblock_part1_2_conv2_bias5
1assignvariableop_12_resblock_part1_3_conv1_kernel3
/assignvariableop_13_resblock_part1_3_conv1_bias5
1assignvariableop_14_resblock_part1_3_conv2_kernel3
/assignvariableop_15_resblock_part1_3_conv2_bias5
1assignvariableop_16_resblock_part1_4_conv1_kernel3
/assignvariableop_17_resblock_part1_4_conv1_bias5
1assignvariableop_18_resblock_part1_4_conv2_kernel3
/assignvariableop_19_resblock_part1_4_conv2_bias,
(assignvariableop_20_downsampler_2_kernel*
&assignvariableop_21_downsampler_2_bias5
1assignvariableop_22_resblock_part2_1_conv1_kernel3
/assignvariableop_23_resblock_part2_1_conv1_bias5
1assignvariableop_24_resblock_part2_1_conv2_kernel3
/assignvariableop_25_resblock_part2_1_conv2_bias5
1assignvariableop_26_resblock_part2_2_conv1_kernel3
/assignvariableop_27_resblock_part2_2_conv1_bias5
1assignvariableop_28_resblock_part2_2_conv2_kernel3
/assignvariableop_29_resblock_part2_2_conv2_bias5
1assignvariableop_30_resblock_part2_3_conv1_kernel3
/assignvariableop_31_resblock_part2_3_conv1_bias5
1assignvariableop_32_resblock_part2_3_conv2_kernel3
/assignvariableop_33_resblock_part2_3_conv2_bias5
1assignvariableop_34_resblock_part2_4_conv1_kernel3
/assignvariableop_35_resblock_part2_4_conv1_bias5
1assignvariableop_36_resblock_part2_4_conv2_kernel3
/assignvariableop_37_resblock_part2_4_conv2_bias5
1assignvariableop_38_resblock_part2_5_conv1_kernel3
/assignvariableop_39_resblock_part2_5_conv1_bias5
1assignvariableop_40_resblock_part2_5_conv2_kernel3
/assignvariableop_41_resblock_part2_5_conv2_bias5
1assignvariableop_42_resblock_part2_6_conv1_kernel3
/assignvariableop_43_resblock_part2_6_conv1_bias5
1assignvariableop_44_resblock_part2_6_conv2_kernel3
/assignvariableop_45_resblock_part2_6_conv2_bias5
1assignvariableop_46_resblock_part2_7_conv1_kernel3
/assignvariableop_47_resblock_part2_7_conv1_bias5
1assignvariableop_48_resblock_part2_7_conv2_kernel3
/assignvariableop_49_resblock_part2_7_conv2_bias5
1assignvariableop_50_resblock_part2_8_conv1_kernel3
/assignvariableop_51_resblock_part2_8_conv1_bias5
1assignvariableop_52_resblock_part2_8_conv2_kernel3
/assignvariableop_53_resblock_part2_8_conv2_bias*
&assignvariableop_54_upsampler_1_kernel(
$assignvariableop_55_upsampler_1_bias5
1assignvariableop_56_resblock_part3_1_conv1_kernel3
/assignvariableop_57_resblock_part3_1_conv1_bias5
1assignvariableop_58_resblock_part3_1_conv2_kernel3
/assignvariableop_59_resblock_part3_1_conv2_bias5
1assignvariableop_60_resblock_part3_2_conv1_kernel3
/assignvariableop_61_resblock_part3_2_conv1_bias5
1assignvariableop_62_resblock_part3_2_conv2_kernel3
/assignvariableop_63_resblock_part3_2_conv2_bias5
1assignvariableop_64_resblock_part3_3_conv1_kernel3
/assignvariableop_65_resblock_part3_3_conv1_bias5
1assignvariableop_66_resblock_part3_3_conv2_kernel3
/assignvariableop_67_resblock_part3_3_conv2_bias5
1assignvariableop_68_resblock_part3_4_conv1_kernel3
/assignvariableop_69_resblock_part3_4_conv1_bias5
1assignvariableop_70_resblock_part3_4_conv2_kernel3
/assignvariableop_71_resblock_part3_4_conv2_bias)
%assignvariableop_72_extra_conv_kernel'
#assignvariableop_73_extra_conv_bias*
&assignvariableop_74_upsampler_2_kernel(
$assignvariableop_75_upsampler_2_bias*
&assignvariableop_76_output_conv_kernel(
$assignvariableop_77_output_conv_bias
identity_79¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_8¢AssignVariableOp_9£#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*¯"
value¥"B¢"OB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¯
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*³
value©B¦OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¹
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*]
dtypesS
Q2O2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_input_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_input_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¬
AssignVariableOp_2AssignVariableOp'assignvariableop_2_downsampler_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ª
AssignVariableOp_3AssignVariableOp%assignvariableop_3_downsampler_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4µ
AssignVariableOp_4AssignVariableOp0assignvariableop_4_resblock_part1_1_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_resblock_part1_1_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6µ
AssignVariableOp_6AssignVariableOp0assignvariableop_6_resblock_part1_1_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_resblock_part1_1_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8µ
AssignVariableOp_8AssignVariableOp0assignvariableop_8_resblock_part1_2_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_resblock_part1_2_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¹
AssignVariableOp_10AssignVariableOp1assignvariableop_10_resblock_part1_2_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_resblock_part1_2_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¹
AssignVariableOp_12AssignVariableOp1assignvariableop_12_resblock_part1_3_conv1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13·
AssignVariableOp_13AssignVariableOp/assignvariableop_13_resblock_part1_3_conv1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¹
AssignVariableOp_14AssignVariableOp1assignvariableop_14_resblock_part1_3_conv2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_resblock_part1_3_conv2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¹
AssignVariableOp_16AssignVariableOp1assignvariableop_16_resblock_part1_4_conv1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_resblock_part1_4_conv1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¹
AssignVariableOp_18AssignVariableOp1assignvariableop_18_resblock_part1_4_conv2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19·
AssignVariableOp_19AssignVariableOp/assignvariableop_19_resblock_part1_4_conv2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_downsampler_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp&assignvariableop_21_downsampler_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¹
AssignVariableOp_22AssignVariableOp1assignvariableop_22_resblock_part2_1_conv1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23·
AssignVariableOp_23AssignVariableOp/assignvariableop_23_resblock_part2_1_conv1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¹
AssignVariableOp_24AssignVariableOp1assignvariableop_24_resblock_part2_1_conv2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25·
AssignVariableOp_25AssignVariableOp/assignvariableop_25_resblock_part2_1_conv2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¹
AssignVariableOp_26AssignVariableOp1assignvariableop_26_resblock_part2_2_conv1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27·
AssignVariableOp_27AssignVariableOp/assignvariableop_27_resblock_part2_2_conv1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¹
AssignVariableOp_28AssignVariableOp1assignvariableop_28_resblock_part2_2_conv2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29·
AssignVariableOp_29AssignVariableOp/assignvariableop_29_resblock_part2_2_conv2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¹
AssignVariableOp_30AssignVariableOp1assignvariableop_30_resblock_part2_3_conv1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31·
AssignVariableOp_31AssignVariableOp/assignvariableop_31_resblock_part2_3_conv1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¹
AssignVariableOp_32AssignVariableOp1assignvariableop_32_resblock_part2_3_conv2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33·
AssignVariableOp_33AssignVariableOp/assignvariableop_33_resblock_part2_3_conv2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¹
AssignVariableOp_34AssignVariableOp1assignvariableop_34_resblock_part2_4_conv1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35·
AssignVariableOp_35AssignVariableOp/assignvariableop_35_resblock_part2_4_conv1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¹
AssignVariableOp_36AssignVariableOp1assignvariableop_36_resblock_part2_4_conv2_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37·
AssignVariableOp_37AssignVariableOp/assignvariableop_37_resblock_part2_4_conv2_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¹
AssignVariableOp_38AssignVariableOp1assignvariableop_38_resblock_part2_5_conv1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39·
AssignVariableOp_39AssignVariableOp/assignvariableop_39_resblock_part2_5_conv1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¹
AssignVariableOp_40AssignVariableOp1assignvariableop_40_resblock_part2_5_conv2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41·
AssignVariableOp_41AssignVariableOp/assignvariableop_41_resblock_part2_5_conv2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¹
AssignVariableOp_42AssignVariableOp1assignvariableop_42_resblock_part2_6_conv1_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43·
AssignVariableOp_43AssignVariableOp/assignvariableop_43_resblock_part2_6_conv1_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¹
AssignVariableOp_44AssignVariableOp1assignvariableop_44_resblock_part2_6_conv2_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45·
AssignVariableOp_45AssignVariableOp/assignvariableop_45_resblock_part2_6_conv2_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¹
AssignVariableOp_46AssignVariableOp1assignvariableop_46_resblock_part2_7_conv1_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47·
AssignVariableOp_47AssignVariableOp/assignvariableop_47_resblock_part2_7_conv1_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¹
AssignVariableOp_48AssignVariableOp1assignvariableop_48_resblock_part2_7_conv2_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49·
AssignVariableOp_49AssignVariableOp/assignvariableop_49_resblock_part2_7_conv2_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¹
AssignVariableOp_50AssignVariableOp1assignvariableop_50_resblock_part2_8_conv1_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51·
AssignVariableOp_51AssignVariableOp/assignvariableop_51_resblock_part2_8_conv1_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¹
AssignVariableOp_52AssignVariableOp1assignvariableop_52_resblock_part2_8_conv2_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53·
AssignVariableOp_53AssignVariableOp/assignvariableop_53_resblock_part2_8_conv2_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54®
AssignVariableOp_54AssignVariableOp&assignvariableop_54_upsampler_1_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¬
AssignVariableOp_55AssignVariableOp$assignvariableop_55_upsampler_1_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¹
AssignVariableOp_56AssignVariableOp1assignvariableop_56_resblock_part3_1_conv1_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57·
AssignVariableOp_57AssignVariableOp/assignvariableop_57_resblock_part3_1_conv1_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¹
AssignVariableOp_58AssignVariableOp1assignvariableop_58_resblock_part3_1_conv2_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59·
AssignVariableOp_59AssignVariableOp/assignvariableop_59_resblock_part3_1_conv2_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¹
AssignVariableOp_60AssignVariableOp1assignvariableop_60_resblock_part3_2_conv1_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61·
AssignVariableOp_61AssignVariableOp/assignvariableop_61_resblock_part3_2_conv1_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62¹
AssignVariableOp_62AssignVariableOp1assignvariableop_62_resblock_part3_2_conv2_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63·
AssignVariableOp_63AssignVariableOp/assignvariableop_63_resblock_part3_2_conv2_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64¹
AssignVariableOp_64AssignVariableOp1assignvariableop_64_resblock_part3_3_conv1_kernelIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65·
AssignVariableOp_65AssignVariableOp/assignvariableop_65_resblock_part3_3_conv1_biasIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¹
AssignVariableOp_66AssignVariableOp1assignvariableop_66_resblock_part3_3_conv2_kernelIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67·
AssignVariableOp_67AssignVariableOp/assignvariableop_67_resblock_part3_3_conv2_biasIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¹
AssignVariableOp_68AssignVariableOp1assignvariableop_68_resblock_part3_4_conv1_kernelIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69·
AssignVariableOp_69AssignVariableOp/assignvariableop_69_resblock_part3_4_conv1_biasIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70¹
AssignVariableOp_70AssignVariableOp1assignvariableop_70_resblock_part3_4_conv2_kernelIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71·
AssignVariableOp_71AssignVariableOp/assignvariableop_71_resblock_part3_4_conv2_biasIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72­
AssignVariableOp_72AssignVariableOp%assignvariableop_72_extra_conv_kernelIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73«
AssignVariableOp_73AssignVariableOp#assignvariableop_73_extra_conv_biasIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74®
AssignVariableOp_74AssignVariableOp&assignvariableop_74_upsampler_2_kernelIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75¬
AssignVariableOp_75AssignVariableOp$assignvariableop_75_upsampler_2_biasIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76®
AssignVariableOp_76AssignVariableOp&assignvariableop_76_output_conv_kernelIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77¬
AssignVariableOp_77AssignVariableOp$assignvariableop_77_output_conv_biasIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_779
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_78Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_78
Identity_79IdentityIdentity_78:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_79"#
identity_79Identity_79:output:0*Ï
_input_shapes½
º: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_77AssignVariableOp_772(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤

é
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6982

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_3150

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


,__inference_downsampler_1_layer_call_fn_6684

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_3_conv2_layer_call_fn_7442

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_40882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_3218

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
·C
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5951

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource0
,downsampler_1_conv2d_readvariableop_resource1
-downsampler_1_biasadd_readvariableop_resource9
5resblock_part1_1_conv1_conv2d_readvariableop_resource:
6resblock_part1_1_conv1_biasadd_readvariableop_resource9
5resblock_part1_1_conv2_conv2d_readvariableop_resource:
6resblock_part1_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_mul_x9
5resblock_part1_2_conv1_conv2d_readvariableop_resource:
6resblock_part1_2_conv1_biasadd_readvariableop_resource9
5resblock_part1_2_conv2_conv2d_readvariableop_resource:
6resblock_part1_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_1_mul_x9
5resblock_part1_3_conv1_conv2d_readvariableop_resource:
6resblock_part1_3_conv1_biasadd_readvariableop_resource9
5resblock_part1_3_conv2_conv2d_readvariableop_resource:
6resblock_part1_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_2_mul_x9
5resblock_part1_4_conv1_conv2d_readvariableop_resource:
6resblock_part1_4_conv1_biasadd_readvariableop_resource9
5resblock_part1_4_conv2_conv2d_readvariableop_resource:
6resblock_part1_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_3_mul_x0
,downsampler_2_conv2d_readvariableop_resource1
-downsampler_2_biasadd_readvariableop_resource9
5resblock_part2_1_conv1_conv2d_readvariableop_resource:
6resblock_part2_1_conv1_biasadd_readvariableop_resource9
5resblock_part2_1_conv2_conv2d_readvariableop_resource:
6resblock_part2_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_4_mul_x9
5resblock_part2_2_conv1_conv2d_readvariableop_resource:
6resblock_part2_2_conv1_biasadd_readvariableop_resource9
5resblock_part2_2_conv2_conv2d_readvariableop_resource:
6resblock_part2_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_5_mul_x9
5resblock_part2_3_conv1_conv2d_readvariableop_resource:
6resblock_part2_3_conv1_biasadd_readvariableop_resource9
5resblock_part2_3_conv2_conv2d_readvariableop_resource:
6resblock_part2_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_6_mul_x9
5resblock_part2_4_conv1_conv2d_readvariableop_resource:
6resblock_part2_4_conv1_biasadd_readvariableop_resource9
5resblock_part2_4_conv2_conv2d_readvariableop_resource:
6resblock_part2_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_7_mul_x9
5resblock_part2_5_conv1_conv2d_readvariableop_resource:
6resblock_part2_5_conv1_biasadd_readvariableop_resource9
5resblock_part2_5_conv2_conv2d_readvariableop_resource:
6resblock_part2_5_conv2_biasadd_readvariableop_resource
tf_math_multiply_8_mul_x9
5resblock_part2_6_conv1_conv2d_readvariableop_resource:
6resblock_part2_6_conv1_biasadd_readvariableop_resource9
5resblock_part2_6_conv2_conv2d_readvariableop_resource:
6resblock_part2_6_conv2_biasadd_readvariableop_resource
tf_math_multiply_9_mul_x9
5resblock_part2_7_conv1_conv2d_readvariableop_resource:
6resblock_part2_7_conv1_biasadd_readvariableop_resource9
5resblock_part2_7_conv2_conv2d_readvariableop_resource:
6resblock_part2_7_conv2_biasadd_readvariableop_resource
tf_math_multiply_10_mul_x9
5resblock_part2_8_conv1_conv2d_readvariableop_resource:
6resblock_part2_8_conv1_biasadd_readvariableop_resource9
5resblock_part2_8_conv2_conv2d_readvariableop_resource:
6resblock_part2_8_conv2_biasadd_readvariableop_resource
tf_math_multiply_11_mul_x.
*upsampler_1_conv2d_readvariableop_resource/
+upsampler_1_biasadd_readvariableop_resource9
5resblock_part3_1_conv1_conv2d_readvariableop_resource:
6resblock_part3_1_conv1_biasadd_readvariableop_resource9
5resblock_part3_1_conv2_conv2d_readvariableop_resource:
6resblock_part3_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_12_mul_x9
5resblock_part3_2_conv1_conv2d_readvariableop_resource:
6resblock_part3_2_conv1_biasadd_readvariableop_resource9
5resblock_part3_2_conv2_conv2d_readvariableop_resource:
6resblock_part3_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_13_mul_x9
5resblock_part3_3_conv1_conv2d_readvariableop_resource:
6resblock_part3_3_conv1_biasadd_readvariableop_resource9
5resblock_part3_3_conv2_conv2d_readvariableop_resource:
6resblock_part3_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_14_mul_x9
5resblock_part3_4_conv1_conv2d_readvariableop_resource:
6resblock_part3_4_conv1_biasadd_readvariableop_resource9
5resblock_part3_4_conv2_conv2d_readvariableop_resource:
6resblock_part3_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_15_mul_x-
)extra_conv_conv2d_readvariableop_resource.
*extra_conv_biasadd_readvariableop_resource.
*upsampler_2_conv2d_readvariableop_resource/
+upsampler_2_biasadd_readvariableop_resource.
*output_conv_conv2d_readvariableop_resource/
+output_conv_biasadd_readvariableop_resource
identity¢$downsampler_1/BiasAdd/ReadVariableOp¢#downsampler_1/Conv2D/ReadVariableOp¢$downsampler_2/BiasAdd/ReadVariableOp¢#downsampler_2/Conv2D/ReadVariableOp¢!extra_conv/BiasAdd/ReadVariableOp¢ extra_conv/Conv2D/ReadVariableOp¢!input_conv/BiasAdd/ReadVariableOp¢ input_conv/Conv2D/ReadVariableOp¢"output_conv/BiasAdd/ReadVariableOp¢!output_conv/Conv2D/ReadVariableOp¢-resblock_part1_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv1/Conv2D/ReadVariableOp¢-resblock_part1_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv2/Conv2D/ReadVariableOp¢-resblock_part1_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv1/Conv2D/ReadVariableOp¢-resblock_part1_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv2/Conv2D/ReadVariableOp¢-resblock_part1_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv1/Conv2D/ReadVariableOp¢-resblock_part1_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv2/Conv2D/ReadVariableOp¢-resblock_part1_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv1/Conv2D/ReadVariableOp¢-resblock_part1_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv1/Conv2D/ReadVariableOp¢-resblock_part2_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢-resblock_part2_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv1/Conv2D/ReadVariableOp¢-resblock_part2_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢-resblock_part2_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv1/Conv2D/ReadVariableOp¢-resblock_part2_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢-resblock_part2_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv1/Conv2D/ReadVariableOp¢-resblock_part2_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_5_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv1/Conv2D/ReadVariableOp¢-resblock_part2_5_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢-resblock_part2_6_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv1/Conv2D/ReadVariableOp¢-resblock_part2_6_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢-resblock_part2_7_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv1/Conv2D/ReadVariableOp¢-resblock_part2_7_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢-resblock_part2_8_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv1/Conv2D/ReadVariableOp¢-resblock_part2_8_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢-resblock_part3_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv1/Conv2D/ReadVariableOp¢-resblock_part3_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv2/Conv2D/ReadVariableOp¢-resblock_part3_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv1/Conv2D/ReadVariableOp¢-resblock_part3_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv2/Conv2D/ReadVariableOp¢-resblock_part3_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv1/Conv2D/ReadVariableOp¢-resblock_part3_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv2/Conv2D/ReadVariableOp¢-resblock_part3_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv1/Conv2D/ReadVariableOp¢-resblock_part3_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv2/Conv2D/ReadVariableOp¢"upsampler_1/BiasAdd/ReadVariableOp¢!upsampler_1/Conv2D/ReadVariableOp¢"upsampler_2/BiasAdd/ReadVariableOp¢!upsampler_2/Conv2D/ReadVariableOp¶
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 input_conv/Conv2D/ReadVariableOpÝ
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
input_conv/Conv2D­
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!input_conv/BiasAdd/ReadVariableOpÍ
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
input_conv/BiasAdd«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings®
zero_padding2d/PadPadinput_conv/BiasAdd:output:0$zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d/Pad¿
#downsampler_1/Conv2D/ReadVariableOpReadVariableOp,downsampler_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_1/Conv2D/ReadVariableOpü
downsampler_1/Conv2DConv2Dzero_padding2d/Pad:output:0+downsampler_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_1/Conv2D¶
$downsampler_1/BiasAdd/ReadVariableOpReadVariableOp-downsampler_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_1/BiasAdd/ReadVariableOpÙ
downsampler_1/BiasAddBiasAdddownsampler_1/Conv2D:output:0,downsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
downsampler_1/BiasAddÚ
,resblock_part1_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv1/Conv2D/ReadVariableOp
resblock_part1_1_conv1/Conv2DConv2Ddownsampler_1/BiasAdd:output:04resblock_part1_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv1/Conv2DÑ
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpý
resblock_part1_1_conv1/BiasAddBiasAdd&resblock_part1_1_conv1/Conv2D:output:05resblock_part1_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv1/BiasAdd§
resblock_part1_1_relu1/ReluRelu'resblock_part1_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_1_relu1/ReluÚ
,resblock_part1_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv2/Conv2D/ReadVariableOp¤
resblock_part1_1_conv2/Conv2DConv2D)resblock_part1_1_relu1/Relu:activations:04resblock_part1_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv2/Conv2DÑ
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpý
resblock_part1_1_conv2/BiasAddBiasAdd&resblock_part1_1_conv2/Conv2D:output:05resblock_part1_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv2/BiasAdd°
tf.math.multiply/MulMultf_math_multiply_mul_x'resblock_part1_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/Mul·
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2Ú
,resblock_part1_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv1/Conv2D/ReadVariableOp
resblock_part1_2_conv1/Conv2DConv2Dtf.__operators__.add/AddV2:z:04resblock_part1_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv1/Conv2DÑ
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpý
resblock_part1_2_conv1/BiasAddBiasAdd&resblock_part1_2_conv1/Conv2D:output:05resblock_part1_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv1/BiasAdd§
resblock_part1_2_relu1/ReluRelu'resblock_part1_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_2_relu1/ReluÚ
,resblock_part1_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv2/Conv2D/ReadVariableOp¤
resblock_part1_2_conv2/Conv2DConv2D)resblock_part1_2_relu1/Relu:activations:04resblock_part1_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv2/Conv2DÑ
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpý
resblock_part1_2_conv2/BiasAddBiasAdd&resblock_part1_2_conv2/Conv2D:output:05resblock_part1_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv2/BiasAdd¶
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x'resblock_part1_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2Ú
,resblock_part1_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv1/Conv2D/ReadVariableOp
resblock_part1_3_conv1/Conv2DConv2D tf.__operators__.add_1/AddV2:z:04resblock_part1_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv1/Conv2DÑ
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpý
resblock_part1_3_conv1/BiasAddBiasAdd&resblock_part1_3_conv1/Conv2D:output:05resblock_part1_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv1/BiasAdd§
resblock_part1_3_relu1/ReluRelu'resblock_part1_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_3_relu1/ReluÚ
,resblock_part1_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv2/Conv2D/ReadVariableOp¤
resblock_part1_3_conv2/Conv2DConv2D)resblock_part1_3_relu1/Relu:activations:04resblock_part1_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv2/Conv2DÑ
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpý
resblock_part1_3_conv2/BiasAddBiasAdd&resblock_part1_3_conv2/Conv2D:output:05resblock_part1_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv2/BiasAdd¶
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x'resblock_part1_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2Ú
,resblock_part1_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv1/Conv2D/ReadVariableOp
resblock_part1_4_conv1/Conv2DConv2D tf.__operators__.add_2/AddV2:z:04resblock_part1_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv1/Conv2DÑ
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpý
resblock_part1_4_conv1/BiasAddBiasAdd&resblock_part1_4_conv1/Conv2D:output:05resblock_part1_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv1/BiasAdd§
resblock_part1_4_relu1/ReluRelu'resblock_part1_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_4_relu1/ReluÚ
,resblock_part1_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv2/Conv2D/ReadVariableOp¤
resblock_part1_4_conv2/Conv2DConv2D)resblock_part1_4_relu1/Relu:activations:04resblock_part1_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv2/Conv2DÑ
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpý
resblock_part1_4_conv2/BiasAddBiasAdd&resblock_part1_4_conv2/Conv2D:output:05resblock_part1_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv2/BiasAdd¶
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x'resblock_part1_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2¯
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddings¹
zero_padding2d_1/PadPad tf.__operators__.add_3/AddV2:z:0&zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d_1/Pad¿
#downsampler_2/Conv2D/ReadVariableOpReadVariableOp,downsampler_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_2/Conv2D/ReadVariableOpü
downsampler_2/Conv2DConv2Dzero_padding2d_1/Pad:output:0+downsampler_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_2/Conv2D¶
$downsampler_2/BiasAdd/ReadVariableOpReadVariableOp-downsampler_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_2/BiasAdd/ReadVariableOp×
downsampler_2/BiasAddBiasAdddownsampler_2/Conv2D:output:0,downsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2
downsampler_2/BiasAddÚ
,resblock_part2_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv1/Conv2D/ReadVariableOp
resblock_part2_1_conv1/Conv2DConv2Ddownsampler_2/BiasAdd:output:04resblock_part2_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv1/Conv2DÑ
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpû
resblock_part2_1_conv1/BiasAddBiasAdd&resblock_part2_1_conv1/Conv2D:output:05resblock_part2_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv1/BiasAdd¥
resblock_part2_1_relu1/ReluRelu'resblock_part2_1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_1_relu1/ReluÚ
,resblock_part2_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢
resblock_part2_1_conv2/Conv2DConv2D)resblock_part2_1_relu1/Relu:activations:04resblock_part2_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv2/Conv2DÑ
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpû
resblock_part2_1_conv2/BiasAddBiasAdd&resblock_part2_1_conv2/Conv2D:output:05resblock_part2_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv2/BiasAdd´
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x'resblock_part2_1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/Mul»
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0downsampler_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2Ú
,resblock_part2_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv1/Conv2D/ReadVariableOp
resblock_part2_2_conv1/Conv2DConv2D tf.__operators__.add_4/AddV2:z:04resblock_part2_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv1/Conv2DÑ
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpû
resblock_part2_2_conv1/BiasAddBiasAdd&resblock_part2_2_conv1/Conv2D:output:05resblock_part2_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv1/BiasAdd¥
resblock_part2_2_relu1/ReluRelu'resblock_part2_2_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_2_relu1/ReluÚ
,resblock_part2_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢
resblock_part2_2_conv2/Conv2DConv2D)resblock_part2_2_relu1/Relu:activations:04resblock_part2_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv2/Conv2DÑ
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpû
resblock_part2_2_conv2/BiasAddBiasAdd&resblock_part2_2_conv2/Conv2D:output:05resblock_part2_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv2/BiasAdd´
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x'resblock_part2_2_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2Ú
,resblock_part2_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv1/Conv2D/ReadVariableOp
resblock_part2_3_conv1/Conv2DConv2D tf.__operators__.add_5/AddV2:z:04resblock_part2_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv1/Conv2DÑ
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpû
resblock_part2_3_conv1/BiasAddBiasAdd&resblock_part2_3_conv1/Conv2D:output:05resblock_part2_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv1/BiasAdd¥
resblock_part2_3_relu1/ReluRelu'resblock_part2_3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_3_relu1/ReluÚ
,resblock_part2_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢
resblock_part2_3_conv2/Conv2DConv2D)resblock_part2_3_relu1/Relu:activations:04resblock_part2_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv2/Conv2DÑ
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpû
resblock_part2_3_conv2/BiasAddBiasAdd&resblock_part2_3_conv2/Conv2D:output:05resblock_part2_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv2/BiasAdd´
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x'resblock_part2_3_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2Ú
,resblock_part2_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv1/Conv2D/ReadVariableOp
resblock_part2_4_conv1/Conv2DConv2D tf.__operators__.add_6/AddV2:z:04resblock_part2_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv1/Conv2DÑ
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpû
resblock_part2_4_conv1/BiasAddBiasAdd&resblock_part2_4_conv1/Conv2D:output:05resblock_part2_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv1/BiasAdd¥
resblock_part2_4_relu1/ReluRelu'resblock_part2_4_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_4_relu1/ReluÚ
,resblock_part2_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢
resblock_part2_4_conv2/Conv2DConv2D)resblock_part2_4_relu1/Relu:activations:04resblock_part2_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv2/Conv2DÑ
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpû
resblock_part2_4_conv2/BiasAddBiasAdd&resblock_part2_4_conv2/Conv2D:output:05resblock_part2_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv2/BiasAdd´
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x'resblock_part2_4_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2Ú
,resblock_part2_5_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv1/Conv2D/ReadVariableOp
resblock_part2_5_conv1/Conv2DConv2D tf.__operators__.add_7/AddV2:z:04resblock_part2_5_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv1/Conv2DÑ
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpû
resblock_part2_5_conv1/BiasAddBiasAdd&resblock_part2_5_conv1/Conv2D:output:05resblock_part2_5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv1/BiasAdd¥
resblock_part2_5_relu1/ReluRelu'resblock_part2_5_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_5_relu1/ReluÚ
,resblock_part2_5_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢
resblock_part2_5_conv2/Conv2DConv2D)resblock_part2_5_relu1/Relu:activations:04resblock_part2_5_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv2/Conv2DÑ
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpû
resblock_part2_5_conv2/BiasAddBiasAdd&resblock_part2_5_conv2/Conv2D:output:05resblock_part2_5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv2/BiasAdd´
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x'resblock_part2_5_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2Ú
,resblock_part2_6_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv1/Conv2D/ReadVariableOp
resblock_part2_6_conv1/Conv2DConv2D tf.__operators__.add_8/AddV2:z:04resblock_part2_6_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv1/Conv2DÑ
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpû
resblock_part2_6_conv1/BiasAddBiasAdd&resblock_part2_6_conv1/Conv2D:output:05resblock_part2_6_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv1/BiasAdd¥
resblock_part2_6_relu1/ReluRelu'resblock_part2_6_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_6_relu1/ReluÚ
,resblock_part2_6_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢
resblock_part2_6_conv2/Conv2DConv2D)resblock_part2_6_relu1/Relu:activations:04resblock_part2_6_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv2/Conv2DÑ
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpû
resblock_part2_6_conv2/BiasAddBiasAdd&resblock_part2_6_conv2/Conv2D:output:05resblock_part2_6_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv2/BiasAdd´
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x'resblock_part2_6_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2Ú
,resblock_part2_7_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv1/Conv2D/ReadVariableOp
resblock_part2_7_conv1/Conv2DConv2D tf.__operators__.add_9/AddV2:z:04resblock_part2_7_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv1/Conv2DÑ
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpû
resblock_part2_7_conv1/BiasAddBiasAdd&resblock_part2_7_conv1/Conv2D:output:05resblock_part2_7_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv1/BiasAdd¥
resblock_part2_7_relu1/ReluRelu'resblock_part2_7_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_7_relu1/ReluÚ
,resblock_part2_7_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢
resblock_part2_7_conv2/Conv2DConv2D)resblock_part2_7_relu1/Relu:activations:04resblock_part2_7_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv2/Conv2DÑ
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpû
resblock_part2_7_conv2/BiasAddBiasAdd&resblock_part2_7_conv2/Conv2D:output:05resblock_part2_7_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv2/BiasAdd·
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x'resblock_part2_7_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2Ú
,resblock_part2_8_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv1/Conv2D/ReadVariableOp
resblock_part2_8_conv1/Conv2DConv2D!tf.__operators__.add_10/AddV2:z:04resblock_part2_8_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv1/Conv2DÑ
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpû
resblock_part2_8_conv1/BiasAddBiasAdd&resblock_part2_8_conv1/Conv2D:output:05resblock_part2_8_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv1/BiasAdd¥
resblock_part2_8_relu1/ReluRelu'resblock_part2_8_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_8_relu1/ReluÚ
,resblock_part2_8_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢
resblock_part2_8_conv2/Conv2DConv2D)resblock_part2_8_relu1/Relu:activations:04resblock_part2_8_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv2/Conv2DÑ
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpû
resblock_part2_8_conv2/BiasAddBiasAdd&resblock_part2_8_conv2/Conv2D:output:05resblock_part2_8_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv2/BiasAdd·
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x'resblock_part2_8_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2º
!upsampler_1/Conv2D/ReadVariableOpReadVariableOp*upsampler_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_1/Conv2D/ReadVariableOpú
upsampler_1/Conv2DConv2D!tf.__operators__.add_11/AddV2:z:0)upsampler_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_1/Conv2D±
"upsampler_1/BiasAdd/ReadVariableOpReadVariableOp+upsampler_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_1/BiasAdd/ReadVariableOpÐ
upsampler_1/BiasAddBiasAddupsampler_1/Conv2D:output:0*upsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2
upsampler_1/BiasAddÙ
!tf.nn.depth_to_space/DepthToSpaceDepthToSpaceupsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpaceÚ
,resblock_part3_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv1/Conv2D/ReadVariableOp¥
resblock_part3_1_conv1/Conv2DConv2D*tf.nn.depth_to_space/DepthToSpace:output:04resblock_part3_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv1/Conv2DÑ
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpý
resblock_part3_1_conv1/BiasAddBiasAdd&resblock_part3_1_conv1/Conv2D:output:05resblock_part3_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv1/BiasAdd§
resblock_part3_1_relu1/ReluRelu'resblock_part3_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_1_relu1/ReluÚ
,resblock_part3_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv2/Conv2D/ReadVariableOp¤
resblock_part3_1_conv2/Conv2DConv2D)resblock_part3_1_relu1/Relu:activations:04resblock_part3_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv2/Conv2DÑ
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpý
resblock_part3_1_conv2/BiasAddBiasAdd&resblock_part3_1_conv2/Conv2D:output:05resblock_part3_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv2/BiasAdd¹
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x'resblock_part3_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2Ú
,resblock_part3_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv1/Conv2D/ReadVariableOp
resblock_part3_2_conv1/Conv2DConv2D!tf.__operators__.add_12/AddV2:z:04resblock_part3_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv1/Conv2DÑ
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpý
resblock_part3_2_conv1/BiasAddBiasAdd&resblock_part3_2_conv1/Conv2D:output:05resblock_part3_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv1/BiasAdd§
resblock_part3_2_relu1/ReluRelu'resblock_part3_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_2_relu1/ReluÚ
,resblock_part3_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv2/Conv2D/ReadVariableOp¤
resblock_part3_2_conv2/Conv2DConv2D)resblock_part3_2_relu1/Relu:activations:04resblock_part3_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv2/Conv2DÑ
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpý
resblock_part3_2_conv2/BiasAddBiasAdd&resblock_part3_2_conv2/Conv2D:output:05resblock_part3_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv2/BiasAdd¹
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x'resblock_part3_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2Ú
,resblock_part3_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv1/Conv2D/ReadVariableOp
resblock_part3_3_conv1/Conv2DConv2D!tf.__operators__.add_13/AddV2:z:04resblock_part3_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv1/Conv2DÑ
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpý
resblock_part3_3_conv1/BiasAddBiasAdd&resblock_part3_3_conv1/Conv2D:output:05resblock_part3_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv1/BiasAdd§
resblock_part3_3_relu1/ReluRelu'resblock_part3_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_3_relu1/ReluÚ
,resblock_part3_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv2/Conv2D/ReadVariableOp¤
resblock_part3_3_conv2/Conv2DConv2D)resblock_part3_3_relu1/Relu:activations:04resblock_part3_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv2/Conv2DÑ
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpý
resblock_part3_3_conv2/BiasAddBiasAdd&resblock_part3_3_conv2/Conv2D:output:05resblock_part3_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv2/BiasAdd¹
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x'resblock_part3_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2Ú
,resblock_part3_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv1/Conv2D/ReadVariableOp
resblock_part3_4_conv1/Conv2DConv2D!tf.__operators__.add_14/AddV2:z:04resblock_part3_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv1/Conv2DÑ
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpý
resblock_part3_4_conv1/BiasAddBiasAdd&resblock_part3_4_conv1/Conv2D:output:05resblock_part3_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv1/BiasAdd§
resblock_part3_4_relu1/ReluRelu'resblock_part3_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_4_relu1/ReluÚ
,resblock_part3_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv2/Conv2D/ReadVariableOp¤
resblock_part3_4_conv2/Conv2DConv2D)resblock_part3_4_relu1/Relu:activations:04resblock_part3_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv2/Conv2DÑ
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpý
resblock_part3_4_conv2/BiasAddBiasAdd&resblock_part3_4_conv2/Conv2D:output:05resblock_part3_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv2/BiasAdd¹
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x'resblock_part3_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2¶
 extra_conv/Conv2D/ReadVariableOpReadVariableOp)extra_conv_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 extra_conv/Conv2D/ReadVariableOpø
extra_conv/Conv2DConv2D!tf.__operators__.add_15/AddV2:z:0(extra_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
extra_conv/Conv2D­
!extra_conv/BiasAdd/ReadVariableOpReadVariableOp*extra_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!extra_conv/BiasAdd/ReadVariableOpÍ
extra_conv/BiasAddBiasAddextra_conv/Conv2D:output:0)extra_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
extra_conv/BiasAddÀ
tf.__operators__.add_16/AddV2AddV2extra_conv/BiasAdd:output:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2º
!upsampler_2/Conv2D/ReadVariableOpReadVariableOp*upsampler_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_2/Conv2D/ReadVariableOpü
upsampler_2/Conv2DConv2D!tf.__operators__.add_16/AddV2:z:0)upsampler_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_2/Conv2D±
"upsampler_2/BiasAdd/ReadVariableOpReadVariableOp+upsampler_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_2/BiasAdd/ReadVariableOpÒ
upsampler_2/BiasAddBiasAddupsampler_2/Conv2D:output:0*upsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
upsampler_2/BiasAddÝ
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpaceupsampler_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpace¹
!output_conv/Conv2D/ReadVariableOpReadVariableOp*output_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!output_conv/Conv2D/ReadVariableOp
output_conv/Conv2DConv2D,tf.nn.depth_to_space_1/DepthToSpace:output:0)output_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
output_conv/Conv2D°
"output_conv/BiasAdd/ReadVariableOpReadVariableOp+output_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"output_conv/BiasAdd/ReadVariableOpÑ
output_conv/BiasAddBiasAddoutput_conv/Conv2D:output:0*output_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
output_conv/BiasAddÝ
IdentityIdentityoutput_conv/BiasAdd:output:0%^downsampler_1/BiasAdd/ReadVariableOp$^downsampler_1/Conv2D/ReadVariableOp%^downsampler_2/BiasAdd/ReadVariableOp$^downsampler_2/Conv2D/ReadVariableOp"^extra_conv/BiasAdd/ReadVariableOp!^extra_conv/Conv2D/ReadVariableOp"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp#^output_conv/BiasAdd/ReadVariableOp"^output_conv/Conv2D/ReadVariableOp.^resblock_part1_1_conv1/BiasAdd/ReadVariableOp-^resblock_part1_1_conv1/Conv2D/ReadVariableOp.^resblock_part1_1_conv2/BiasAdd/ReadVariableOp-^resblock_part1_1_conv2/Conv2D/ReadVariableOp.^resblock_part1_2_conv1/BiasAdd/ReadVariableOp-^resblock_part1_2_conv1/Conv2D/ReadVariableOp.^resblock_part1_2_conv2/BiasAdd/ReadVariableOp-^resblock_part1_2_conv2/Conv2D/ReadVariableOp.^resblock_part1_3_conv1/BiasAdd/ReadVariableOp-^resblock_part1_3_conv1/Conv2D/ReadVariableOp.^resblock_part1_3_conv2/BiasAdd/ReadVariableOp-^resblock_part1_3_conv2/Conv2D/ReadVariableOp.^resblock_part1_4_conv1/BiasAdd/ReadVariableOp-^resblock_part1_4_conv1/Conv2D/ReadVariableOp.^resblock_part1_4_conv2/BiasAdd/ReadVariableOp-^resblock_part1_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_1_conv1/BiasAdd/ReadVariableOp-^resblock_part2_1_conv1/Conv2D/ReadVariableOp.^resblock_part2_1_conv2/BiasAdd/ReadVariableOp-^resblock_part2_1_conv2/Conv2D/ReadVariableOp.^resblock_part2_2_conv1/BiasAdd/ReadVariableOp-^resblock_part2_2_conv1/Conv2D/ReadVariableOp.^resblock_part2_2_conv2/BiasAdd/ReadVariableOp-^resblock_part2_2_conv2/Conv2D/ReadVariableOp.^resblock_part2_3_conv1/BiasAdd/ReadVariableOp-^resblock_part2_3_conv1/Conv2D/ReadVariableOp.^resblock_part2_3_conv2/BiasAdd/ReadVariableOp-^resblock_part2_3_conv2/Conv2D/ReadVariableOp.^resblock_part2_4_conv1/BiasAdd/ReadVariableOp-^resblock_part2_4_conv1/Conv2D/ReadVariableOp.^resblock_part2_4_conv2/BiasAdd/ReadVariableOp-^resblock_part2_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_5_conv1/BiasAdd/ReadVariableOp-^resblock_part2_5_conv1/Conv2D/ReadVariableOp.^resblock_part2_5_conv2/BiasAdd/ReadVariableOp-^resblock_part2_5_conv2/Conv2D/ReadVariableOp.^resblock_part2_6_conv1/BiasAdd/ReadVariableOp-^resblock_part2_6_conv1/Conv2D/ReadVariableOp.^resblock_part2_6_conv2/BiasAdd/ReadVariableOp-^resblock_part2_6_conv2/Conv2D/ReadVariableOp.^resblock_part2_7_conv1/BiasAdd/ReadVariableOp-^resblock_part2_7_conv1/Conv2D/ReadVariableOp.^resblock_part2_7_conv2/BiasAdd/ReadVariableOp-^resblock_part2_7_conv2/Conv2D/ReadVariableOp.^resblock_part2_8_conv1/BiasAdd/ReadVariableOp-^resblock_part2_8_conv1/Conv2D/ReadVariableOp.^resblock_part2_8_conv2/BiasAdd/ReadVariableOp-^resblock_part2_8_conv2/Conv2D/ReadVariableOp.^resblock_part3_1_conv1/BiasAdd/ReadVariableOp-^resblock_part3_1_conv1/Conv2D/ReadVariableOp.^resblock_part3_1_conv2/BiasAdd/ReadVariableOp-^resblock_part3_1_conv2/Conv2D/ReadVariableOp.^resblock_part3_2_conv1/BiasAdd/ReadVariableOp-^resblock_part3_2_conv1/Conv2D/ReadVariableOp.^resblock_part3_2_conv2/BiasAdd/ReadVariableOp-^resblock_part3_2_conv2/Conv2D/ReadVariableOp.^resblock_part3_3_conv1/BiasAdd/ReadVariableOp-^resblock_part3_3_conv1/Conv2D/ReadVariableOp.^resblock_part3_3_conv2/BiasAdd/ReadVariableOp-^resblock_part3_3_conv2/Conv2D/ReadVariableOp.^resblock_part3_4_conv1/BiasAdd/ReadVariableOp-^resblock_part3_4_conv1/Conv2D/ReadVariableOp.^resblock_part3_4_conv2/BiasAdd/ReadVariableOp-^resblock_part3_4_conv2/Conv2D/ReadVariableOp#^upsampler_1/BiasAdd/ReadVariableOp"^upsampler_1/Conv2D/ReadVariableOp#^upsampler_2/BiasAdd/ReadVariableOp"^upsampler_2/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2L
$downsampler_1/BiasAdd/ReadVariableOp$downsampler_1/BiasAdd/ReadVariableOp2J
#downsampler_1/Conv2D/ReadVariableOp#downsampler_1/Conv2D/ReadVariableOp2L
$downsampler_2/BiasAdd/ReadVariableOp$downsampler_2/BiasAdd/ReadVariableOp2J
#downsampler_2/Conv2D/ReadVariableOp#downsampler_2/Conv2D/ReadVariableOp2F
!extra_conv/BiasAdd/ReadVariableOp!extra_conv/BiasAdd/ReadVariableOp2D
 extra_conv/Conv2D/ReadVariableOp extra_conv/Conv2D/ReadVariableOp2F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2H
"output_conv/BiasAdd/ReadVariableOp"output_conv/BiasAdd/ReadVariableOp2F
!output_conv/Conv2D/ReadVariableOp!output_conv/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv1/BiasAdd/ReadVariableOp-resblock_part1_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv1/Conv2D/ReadVariableOp,resblock_part1_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv2/BiasAdd/ReadVariableOp-resblock_part1_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv2/Conv2D/ReadVariableOp,resblock_part1_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv1/BiasAdd/ReadVariableOp-resblock_part1_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv1/Conv2D/ReadVariableOp,resblock_part1_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv2/BiasAdd/ReadVariableOp-resblock_part1_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv2/Conv2D/ReadVariableOp,resblock_part1_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv1/BiasAdd/ReadVariableOp-resblock_part1_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv1/Conv2D/ReadVariableOp,resblock_part1_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv2/BiasAdd/ReadVariableOp-resblock_part1_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv2/Conv2D/ReadVariableOp,resblock_part1_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv1/BiasAdd/ReadVariableOp-resblock_part1_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv1/Conv2D/ReadVariableOp,resblock_part1_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv2/BiasAdd/ReadVariableOp-resblock_part1_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv2/Conv2D/ReadVariableOp,resblock_part1_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv1/BiasAdd/ReadVariableOp-resblock_part2_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv1/Conv2D/ReadVariableOp,resblock_part2_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv2/BiasAdd/ReadVariableOp-resblock_part2_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv2/Conv2D/ReadVariableOp,resblock_part2_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv1/BiasAdd/ReadVariableOp-resblock_part2_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv1/Conv2D/ReadVariableOp,resblock_part2_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv2/BiasAdd/ReadVariableOp-resblock_part2_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv2/Conv2D/ReadVariableOp,resblock_part2_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv1/BiasAdd/ReadVariableOp-resblock_part2_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv1/Conv2D/ReadVariableOp,resblock_part2_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv2/BiasAdd/ReadVariableOp-resblock_part2_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv2/Conv2D/ReadVariableOp,resblock_part2_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv1/BiasAdd/ReadVariableOp-resblock_part2_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv1/Conv2D/ReadVariableOp,resblock_part2_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv2/BiasAdd/ReadVariableOp-resblock_part2_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv2/Conv2D/ReadVariableOp,resblock_part2_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv1/BiasAdd/ReadVariableOp-resblock_part2_5_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv1/Conv2D/ReadVariableOp,resblock_part2_5_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv2/BiasAdd/ReadVariableOp-resblock_part2_5_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv2/Conv2D/ReadVariableOp,resblock_part2_5_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv1/BiasAdd/ReadVariableOp-resblock_part2_6_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv1/Conv2D/ReadVariableOp,resblock_part2_6_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv2/BiasAdd/ReadVariableOp-resblock_part2_6_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv2/Conv2D/ReadVariableOp,resblock_part2_6_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv1/BiasAdd/ReadVariableOp-resblock_part2_7_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv1/Conv2D/ReadVariableOp,resblock_part2_7_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv2/BiasAdd/ReadVariableOp-resblock_part2_7_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv2/Conv2D/ReadVariableOp,resblock_part2_7_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv1/BiasAdd/ReadVariableOp-resblock_part2_8_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv1/Conv2D/ReadVariableOp,resblock_part2_8_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv2/BiasAdd/ReadVariableOp-resblock_part2_8_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv2/Conv2D/ReadVariableOp,resblock_part2_8_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv1/BiasAdd/ReadVariableOp-resblock_part3_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv1/Conv2D/ReadVariableOp,resblock_part3_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv2/BiasAdd/ReadVariableOp-resblock_part3_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv2/Conv2D/ReadVariableOp,resblock_part3_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv1/BiasAdd/ReadVariableOp-resblock_part3_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv1/Conv2D/ReadVariableOp,resblock_part3_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv2/BiasAdd/ReadVariableOp-resblock_part3_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv2/Conv2D/ReadVariableOp,resblock_part3_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv1/BiasAdd/ReadVariableOp-resblock_part3_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv1/Conv2D/ReadVariableOp,resblock_part3_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv2/BiasAdd/ReadVariableOp-resblock_part3_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv2/Conv2D/ReadVariableOp,resblock_part3_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv1/BiasAdd/ReadVariableOp-resblock_part3_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv1/Conv2D/ReadVariableOp,resblock_part3_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv2/BiasAdd/ReadVariableOp-resblock_part3_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv2/Conv2D/ReadVariableOp,resblock_part3_4_conv2/Conv2D/ReadVariableOp2H
"upsampler_1/BiasAdd/ReadVariableOp"upsampler_1/BiasAdd/ReadVariableOp2F
!upsampler_1/Conv2D/ReadVariableOp!upsampler_1/Conv2D/ReadVariableOp2H
"upsampler_2/BiasAdd/ReadVariableOp"upsampler_2/BiasAdd/ReadVariableOp2F
!upsampler_2/Conv2D/ReadVariableOp!upsampler_2/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
¤

é
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6934

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¨

Þ
E__inference_upsampler_2_layer_call_and_return_conditional_losses_4212

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¢
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd 
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_4_relu1_layer_call_fn_6857

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32682
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦

à
G__inference_downsampler_1_layer_call_and_return_conditional_losses_3017

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_4049

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢

Ý
D__inference_extra_conv_layer_call_and_return_conditional_losses_4185

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_7255

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_3431

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_4138

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_3_relu1_layer_call_fn_6809

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32002
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_2_conv1_layer_call_fn_7365

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_39812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_3_relu1_layer_call_fn_7020

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_34992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_7015

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_7126

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_4070

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_2_conv1_layer_call_fn_6751

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_31112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_6804

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_1_conv2_layer_call_fn_6943

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_33812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_3934

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_3410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_3682

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


Þ
E__inference_upsampler_1_layer_call_and_return_conditional_losses_3886

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp 
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_1_conv2_layer_call_fn_6732

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_30822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_4088

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_7145

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_6_relu1_layer_call_fn_7164

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_7207

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_5_relu1_layer_call_fn_7116

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


*__inference_upsampler_1_layer_call_fn_7298

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_7370

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

~
)__inference_extra_conv_layer_call_fn_7509

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_41852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_3653

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_2_relu1_layer_call_fn_6761

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31322
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_3043

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_3517

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_3342

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_3381

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_6756

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_6819

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

à
G__inference_downsampler_2_layer_call_and_return_conditional_losses_3316

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_3913

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_3952

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_3_conv1_layer_call_fn_6799

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_31792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_7001

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_7030

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_4_relu1_layer_call_fn_7068

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35672
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_5_conv1_layer_call_fn_7106

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_36142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_2_relu1_layer_call_fn_7375

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40022
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_6790

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_4_conv1_layer_call_fn_6847

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_32472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_3839

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
£

Þ
E__inference_output_conv_layer_call_and_return_conditional_losses_4239

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
é
í%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4795

inputs
input_conv_4530
input_conv_4532
downsampler_1_4536
downsampler_1_4538
resblock_part1_1_conv1_4541
resblock_part1_1_conv1_4543
resblock_part1_1_conv2_4547
resblock_part1_1_conv2_4549
tf_math_multiply_mul_x
resblock_part1_2_conv1_4555
resblock_part1_2_conv1_4557
resblock_part1_2_conv2_4561
resblock_part1_2_conv2_4563
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_4569
resblock_part1_3_conv1_4571
resblock_part1_3_conv2_4575
resblock_part1_3_conv2_4577
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_4583
resblock_part1_4_conv1_4585
resblock_part1_4_conv2_4589
resblock_part1_4_conv2_4591
tf_math_multiply_3_mul_x
downsampler_2_4598
downsampler_2_4600
resblock_part2_1_conv1_4603
resblock_part2_1_conv1_4605
resblock_part2_1_conv2_4609
resblock_part2_1_conv2_4611
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_4617
resblock_part2_2_conv1_4619
resblock_part2_2_conv2_4623
resblock_part2_2_conv2_4625
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_4631
resblock_part2_3_conv1_4633
resblock_part2_3_conv2_4637
resblock_part2_3_conv2_4639
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_4645
resblock_part2_4_conv1_4647
resblock_part2_4_conv2_4651
resblock_part2_4_conv2_4653
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_4659
resblock_part2_5_conv1_4661
resblock_part2_5_conv2_4665
resblock_part2_5_conv2_4667
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_4673
resblock_part2_6_conv1_4675
resblock_part2_6_conv2_4679
resblock_part2_6_conv2_4681
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_4687
resblock_part2_7_conv1_4689
resblock_part2_7_conv2_4693
resblock_part2_7_conv2_4695
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_4701
resblock_part2_8_conv1_4703
resblock_part2_8_conv2_4707
resblock_part2_8_conv2_4709
tf_math_multiply_11_mul_x
upsampler_1_4715
upsampler_1_4717
resblock_part3_1_conv1_4721
resblock_part3_1_conv1_4723
resblock_part3_1_conv2_4727
resblock_part3_1_conv2_4729
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_4735
resblock_part3_2_conv1_4737
resblock_part3_2_conv2_4741
resblock_part3_2_conv2_4743
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4749
resblock_part3_3_conv1_4751
resblock_part3_3_conv2_4755
resblock_part3_3_conv2_4757
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4763
resblock_part3_4_conv1_4765
resblock_part3_4_conv2_4769
resblock_part3_4_conv2_4771
tf_math_multiply_15_mul_x
extra_conv_4777
extra_conv_4779
upsampler_2_4783
upsampler_2_4785
output_conv_4789
output_conv_4791
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCall¥
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4530input_conv_4532*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_29902$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29572 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_4536downsampler_1_4538*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30172'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_4541resblock_part1_1_conv1_4543*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304320
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30642(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_4547resblock_part1_1_conv2_4549*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308220
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_4555resblock_part1_2_conv1_4557*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311120
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31322(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_4561resblock_part1_2_conv2_4563*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315020
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_4569resblock_part1_3_conv1_4571*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_317920
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32002(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_4575resblock_part1_3_conv2_4577*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_321820
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_4583resblock_part1_4_conv1_4585*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_324720
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32682(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_4589resblock_part1_4_conv2_4591*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_328620
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29702"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_4598downsampler_2_4600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33162'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_4603resblock_part2_1_conv1_4605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334220
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33632(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_4609resblock_part2_1_conv2_4611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338120
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_4617resblock_part2_2_conv1_4619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341020
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34312(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_4623resblock_part2_2_conv2_4625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_344920
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_4631resblock_part2_3_conv1_4633*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_347820
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_34992(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_4637resblock_part2_3_conv2_4639*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_351720
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_4645resblock_part2_4_conv1_4647*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_354620
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35672(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_4651resblock_part2_4_conv2_4653*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358520
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_4659resblock_part2_5_conv1_4661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361420
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36352(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_4665resblock_part2_5_conv2_4667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365320
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_4673resblock_part2_6_conv1_4675*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368220
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37032(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_4679resblock_part2_6_conv2_4681*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372120
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_4687resblock_part2_7_conv1_4689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375020
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37712(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_4693resblock_part2_7_conv2_4695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_378920
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_4701resblock_part2_8_conv1_4703*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_381820
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38392(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_4707resblock_part2_8_conv2_4709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_385720
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_4715upsampler_1_4717*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38862%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_4721resblock_part3_1_conv1_4723*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391320
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39342(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_4727resblock_part3_1_conv2_4729*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395220
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_4735resblock_part3_2_conv1_4737*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398120
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40022(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4741resblock_part3_2_conv2_4743*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402020
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4749resblock_part3_3_conv1_4751*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_404920
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40702(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4755resblock_part3_3_conv2_4757*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_408820
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4763resblock_part3_4_conv1_4765*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_411720
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41382(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4769resblock_part3_4_conv2_4771*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_415620
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4777extra_conv_4779*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_41852$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4783upsampler_2_4785*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42122%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4789output_conv_4791*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_42392%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
¤

é
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_3721

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_3064

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_3771

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_7433

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_8_relu1_layer_call_fn_7260

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_6867

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6967

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_3981

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_7111

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_5_conv2_layer_call_fn_7135

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_36532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_1_relu1_layer_call_fn_7327

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39342
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_6742

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

à
G__inference_downsampler_2_layer_call_and_return_conditional_losses_6886

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_7385

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_4_conv2_layer_call_fn_6876

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_32862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ
í$
__inference__traced_save_7820
file_prefix0
,savev2_input_conv_kernel_read_readvariableop.
*savev2_input_conv_bias_read_readvariableop3
/savev2_downsampler_1_kernel_read_readvariableop1
-savev2_downsampler_1_bias_read_readvariableop<
8savev2_resblock_part1_1_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_1_conv1_bias_read_readvariableop<
8savev2_resblock_part1_1_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_1_conv2_bias_read_readvariableop<
8savev2_resblock_part1_2_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_2_conv1_bias_read_readvariableop<
8savev2_resblock_part1_2_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_2_conv2_bias_read_readvariableop<
8savev2_resblock_part1_3_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_3_conv1_bias_read_readvariableop<
8savev2_resblock_part1_3_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_3_conv2_bias_read_readvariableop<
8savev2_resblock_part1_4_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_4_conv1_bias_read_readvariableop<
8savev2_resblock_part1_4_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_4_conv2_bias_read_readvariableop3
/savev2_downsampler_2_kernel_read_readvariableop1
-savev2_downsampler_2_bias_read_readvariableop<
8savev2_resblock_part2_1_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_1_conv1_bias_read_readvariableop<
8savev2_resblock_part2_1_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_1_conv2_bias_read_readvariableop<
8savev2_resblock_part2_2_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_2_conv1_bias_read_readvariableop<
8savev2_resblock_part2_2_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_2_conv2_bias_read_readvariableop<
8savev2_resblock_part2_3_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_3_conv1_bias_read_readvariableop<
8savev2_resblock_part2_3_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_3_conv2_bias_read_readvariableop<
8savev2_resblock_part2_4_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_4_conv1_bias_read_readvariableop<
8savev2_resblock_part2_4_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_4_conv2_bias_read_readvariableop<
8savev2_resblock_part2_5_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_5_conv1_bias_read_readvariableop<
8savev2_resblock_part2_5_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_5_conv2_bias_read_readvariableop<
8savev2_resblock_part2_6_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_6_conv1_bias_read_readvariableop<
8savev2_resblock_part2_6_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_6_conv2_bias_read_readvariableop<
8savev2_resblock_part2_7_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_7_conv1_bias_read_readvariableop<
8savev2_resblock_part2_7_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_7_conv2_bias_read_readvariableop<
8savev2_resblock_part2_8_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_8_conv1_bias_read_readvariableop<
8savev2_resblock_part2_8_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_8_conv2_bias_read_readvariableop1
-savev2_upsampler_1_kernel_read_readvariableop/
+savev2_upsampler_1_bias_read_readvariableop<
8savev2_resblock_part3_1_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_1_conv1_bias_read_readvariableop<
8savev2_resblock_part3_1_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_1_conv2_bias_read_readvariableop<
8savev2_resblock_part3_2_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_2_conv1_bias_read_readvariableop<
8savev2_resblock_part3_2_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_2_conv2_bias_read_readvariableop<
8savev2_resblock_part3_3_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_3_conv1_bias_read_readvariableop<
8savev2_resblock_part3_3_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_3_conv2_bias_read_readvariableop<
8savev2_resblock_part3_4_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_4_conv1_bias_read_readvariableop<
8savev2_resblock_part3_4_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_4_conv2_bias_read_readvariableop0
,savev2_extra_conv_kernel_read_readvariableop.
*savev2_extra_conv_bias_read_readvariableop1
-savev2_upsampler_2_kernel_read_readvariableop/
+savev2_upsampler_2_bias_read_readvariableop1
-savev2_output_conv_kernel_read_readvariableop/
+savev2_output_conv_bias_read_readvariableop
savev2_const_16

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*¯"
value¥"B¢"OB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*³
value©B¦OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÓ#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_input_conv_kernel_read_readvariableop*savev2_input_conv_bias_read_readvariableop/savev2_downsampler_1_kernel_read_readvariableop-savev2_downsampler_1_bias_read_readvariableop8savev2_resblock_part1_1_conv1_kernel_read_readvariableop6savev2_resblock_part1_1_conv1_bias_read_readvariableop8savev2_resblock_part1_1_conv2_kernel_read_readvariableop6savev2_resblock_part1_1_conv2_bias_read_readvariableop8savev2_resblock_part1_2_conv1_kernel_read_readvariableop6savev2_resblock_part1_2_conv1_bias_read_readvariableop8savev2_resblock_part1_2_conv2_kernel_read_readvariableop6savev2_resblock_part1_2_conv2_bias_read_readvariableop8savev2_resblock_part1_3_conv1_kernel_read_readvariableop6savev2_resblock_part1_3_conv1_bias_read_readvariableop8savev2_resblock_part1_3_conv2_kernel_read_readvariableop6savev2_resblock_part1_3_conv2_bias_read_readvariableop8savev2_resblock_part1_4_conv1_kernel_read_readvariableop6savev2_resblock_part1_4_conv1_bias_read_readvariableop8savev2_resblock_part1_4_conv2_kernel_read_readvariableop6savev2_resblock_part1_4_conv2_bias_read_readvariableop/savev2_downsampler_2_kernel_read_readvariableop-savev2_downsampler_2_bias_read_readvariableop8savev2_resblock_part2_1_conv1_kernel_read_readvariableop6savev2_resblock_part2_1_conv1_bias_read_readvariableop8savev2_resblock_part2_1_conv2_kernel_read_readvariableop6savev2_resblock_part2_1_conv2_bias_read_readvariableop8savev2_resblock_part2_2_conv1_kernel_read_readvariableop6savev2_resblock_part2_2_conv1_bias_read_readvariableop8savev2_resblock_part2_2_conv2_kernel_read_readvariableop6savev2_resblock_part2_2_conv2_bias_read_readvariableop8savev2_resblock_part2_3_conv1_kernel_read_readvariableop6savev2_resblock_part2_3_conv1_bias_read_readvariableop8savev2_resblock_part2_3_conv2_kernel_read_readvariableop6savev2_resblock_part2_3_conv2_bias_read_readvariableop8savev2_resblock_part2_4_conv1_kernel_read_readvariableop6savev2_resblock_part2_4_conv1_bias_read_readvariableop8savev2_resblock_part2_4_conv2_kernel_read_readvariableop6savev2_resblock_part2_4_conv2_bias_read_readvariableop8savev2_resblock_part2_5_conv1_kernel_read_readvariableop6savev2_resblock_part2_5_conv1_bias_read_readvariableop8savev2_resblock_part2_5_conv2_kernel_read_readvariableop6savev2_resblock_part2_5_conv2_bias_read_readvariableop8savev2_resblock_part2_6_conv1_kernel_read_readvariableop6savev2_resblock_part2_6_conv1_bias_read_readvariableop8savev2_resblock_part2_6_conv2_kernel_read_readvariableop6savev2_resblock_part2_6_conv2_bias_read_readvariableop8savev2_resblock_part2_7_conv1_kernel_read_readvariableop6savev2_resblock_part2_7_conv1_bias_read_readvariableop8savev2_resblock_part2_7_conv2_kernel_read_readvariableop6savev2_resblock_part2_7_conv2_bias_read_readvariableop8savev2_resblock_part2_8_conv1_kernel_read_readvariableop6savev2_resblock_part2_8_conv1_bias_read_readvariableop8savev2_resblock_part2_8_conv2_kernel_read_readvariableop6savev2_resblock_part2_8_conv2_bias_read_readvariableop-savev2_upsampler_1_kernel_read_readvariableop+savev2_upsampler_1_bias_read_readvariableop8savev2_resblock_part3_1_conv1_kernel_read_readvariableop6savev2_resblock_part3_1_conv1_bias_read_readvariableop8savev2_resblock_part3_1_conv2_kernel_read_readvariableop6savev2_resblock_part3_1_conv2_bias_read_readvariableop8savev2_resblock_part3_2_conv1_kernel_read_readvariableop6savev2_resblock_part3_2_conv1_bias_read_readvariableop8savev2_resblock_part3_2_conv2_kernel_read_readvariableop6savev2_resblock_part3_2_conv2_bias_read_readvariableop8savev2_resblock_part3_3_conv1_kernel_read_readvariableop6savev2_resblock_part3_3_conv1_bias_read_readvariableop8savev2_resblock_part3_3_conv2_kernel_read_readvariableop6savev2_resblock_part3_3_conv2_bias_read_readvariableop8savev2_resblock_part3_4_conv1_kernel_read_readvariableop6savev2_resblock_part3_4_conv1_bias_read_readvariableop8savev2_resblock_part3_4_conv2_kernel_read_readvariableop6savev2_resblock_part3_4_conv2_bias_read_readvariableop,savev2_extra_conv_kernel_read_readvariableop*savev2_extra_conv_bias_read_readvariableop-savev2_upsampler_2_kernel_read_readvariableop+savev2_upsampler_2_bias_read_readvariableop-savev2_output_conv_kernel_read_readvariableop+savev2_output_conv_bias_read_readvariableopsavev2_const_16"/device:CPU:0*
_output_shapes
 *]
dtypesS
Q2O2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Å
_input_shapes³
°: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:,!(
&
_output_shapes
:@@: "

_output_shapes
:@:,#(
&
_output_shapes
:@@: $

_output_shapes
:@:,%(
&
_output_shapes
:@@: &

_output_shapes
:@:,'(
&
_output_shapes
:@@: (

_output_shapes
:@:,)(
&
_output_shapes
:@@: *

_output_shapes
:@:,+(
&
_output_shapes
:@@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@:,/(
&
_output_shapes
:@@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@@: 2

_output_shapes
:@:,3(
&
_output_shapes
:@@: 4

_output_shapes
:@:,5(
&
_output_shapes
:@@: 6

_output_shapes
:@:-7)
'
_output_shapes
:@:!8

_output_shapes	
::,9(
&
_output_shapes
:@@: :

_output_shapes
:@:,;(
&
_output_shapes
:@@: <

_output_shapes
:@:,=(
&
_output_shapes
:@@: >

_output_shapes
:@:,?(
&
_output_shapes
:@@: @

_output_shapes
:@:,A(
&
_output_shapes
:@@: B

_output_shapes
:@:,C(
&
_output_shapes
:@@: D

_output_shapes
:@:,E(
&
_output_shapes
:@@: F

_output_shapes
:@:,G(
&
_output_shapes
:@@: H

_output_shapes
:@:,I(
&
_output_shapes
:@@: J

_output_shapes
:@:-K)
'
_output_shapes
:@:!L

_output_shapes	
::,M(
&
_output_shapes
:@: N

_output_shapes
::O

_output_shapes
: 
æ
l
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_6852

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_6694

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_2_conv2_layer_call_fn_6780

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_31502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©é
ò%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4256
input_layer
input_conv_3001
input_conv_3003
downsampler_1_3028
downsampler_1_3030
resblock_part1_1_conv1_3054
resblock_part1_1_conv1_3056
resblock_part1_1_conv2_3093
resblock_part1_1_conv2_3095
tf_math_multiply_mul_x
resblock_part1_2_conv1_3122
resblock_part1_2_conv1_3124
resblock_part1_2_conv2_3161
resblock_part1_2_conv2_3163
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_3190
resblock_part1_3_conv1_3192
resblock_part1_3_conv2_3229
resblock_part1_3_conv2_3231
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_3258
resblock_part1_4_conv1_3260
resblock_part1_4_conv2_3297
resblock_part1_4_conv2_3299
tf_math_multiply_3_mul_x
downsampler_2_3327
downsampler_2_3329
resblock_part2_1_conv1_3353
resblock_part2_1_conv1_3355
resblock_part2_1_conv2_3392
resblock_part2_1_conv2_3394
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_3421
resblock_part2_2_conv1_3423
resblock_part2_2_conv2_3460
resblock_part2_2_conv2_3462
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_3489
resblock_part2_3_conv1_3491
resblock_part2_3_conv2_3528
resblock_part2_3_conv2_3530
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_3557
resblock_part2_4_conv1_3559
resblock_part2_4_conv2_3596
resblock_part2_4_conv2_3598
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_3625
resblock_part2_5_conv1_3627
resblock_part2_5_conv2_3664
resblock_part2_5_conv2_3666
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_3693
resblock_part2_6_conv1_3695
resblock_part2_6_conv2_3732
resblock_part2_6_conv2_3734
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_3761
resblock_part2_7_conv1_3763
resblock_part2_7_conv2_3800
resblock_part2_7_conv2_3802
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_3829
resblock_part2_8_conv1_3831
resblock_part2_8_conv2_3868
resblock_part2_8_conv2_3870
tf_math_multiply_11_mul_x
upsampler_1_3897
upsampler_1_3899
resblock_part3_1_conv1_3924
resblock_part3_1_conv1_3926
resblock_part3_1_conv2_3963
resblock_part3_1_conv2_3965
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_3992
resblock_part3_2_conv1_3994
resblock_part3_2_conv2_4031
resblock_part3_2_conv2_4033
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4060
resblock_part3_3_conv1_4062
resblock_part3_3_conv2_4099
resblock_part3_3_conv2_4101
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4128
resblock_part3_4_conv1_4130
resblock_part3_4_conv2_4167
resblock_part3_4_conv2_4169
tf_math_multiply_15_mul_x
extra_conv_4196
extra_conv_4198
upsampler_2_4223
upsampler_2_4225
output_conv_4250
output_conv_4252
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCallª
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_layerinput_conv_3001input_conv_3003*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_29902$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29572 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_3028downsampler_1_3030*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30172'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_3054resblock_part1_1_conv1_3056*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304320
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30642(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_3093resblock_part1_1_conv2_3095*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308220
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_3122resblock_part1_2_conv1_3124*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311120
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31322(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_3161resblock_part1_2_conv2_3163*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315020
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_3190resblock_part1_3_conv1_3192*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_317920
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32002(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_3229resblock_part1_3_conv2_3231*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_321820
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_3258resblock_part1_4_conv1_3260*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_324720
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32682(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_3297resblock_part1_4_conv2_3299*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_328620
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29702"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_3327downsampler_2_3329*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33162'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_3353resblock_part2_1_conv1_3355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334220
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33632(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_3392resblock_part2_1_conv2_3394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338120
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_3421resblock_part2_2_conv1_3423*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341020
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34312(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_3460resblock_part2_2_conv2_3462*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_344920
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_3489resblock_part2_3_conv1_3491*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_347820
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_34992(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_3528resblock_part2_3_conv2_3530*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_351720
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_3557resblock_part2_4_conv1_3559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_354620
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35672(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_3596resblock_part2_4_conv2_3598*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358520
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_3625resblock_part2_5_conv1_3627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361420
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36352(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_3664resblock_part2_5_conv2_3666*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365320
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_3693resblock_part2_6_conv1_3695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368220
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37032(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_3732resblock_part2_6_conv2_3734*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372120
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_3761resblock_part2_7_conv1_3763*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375020
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37712(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_3800resblock_part2_7_conv2_3802*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_378920
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_3829resblock_part2_8_conv1_3831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_381820
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38392(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_3868resblock_part2_8_conv2_3870*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_385720
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_3897upsampler_1_3899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38862%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_3924resblock_part3_1_conv1_3926*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391320
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39342(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_3963resblock_part3_1_conv2_3965*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395220
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_3992resblock_part3_2_conv1_3994*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398120
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40022(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4031resblock_part3_2_conv2_4033*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402020
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4060resblock_part3_3_conv1_4062*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_404920
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40702(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4099resblock_part3_3_conv2_4101*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_408820
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4128resblock_part3_4_conv1_4130*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_411720
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41382(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4167resblock_part3_4_conv2_4169*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_415620
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4196extra_conv_4198*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_41852$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4223upsampler_2_4225*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42122%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4250output_conv_4252*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_42392%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
®
K
/__inference_zero_padding2d_1_layer_call_fn_2976

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29702
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_3789

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_6771

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_7159

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_1_relu1_layer_call_fn_6924

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_7322

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_3286

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_4020

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_8_conv1_layer_call_fn_7250

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_38182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
é
í%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5256

inputs
input_conv_4991
input_conv_4993
downsampler_1_4997
downsampler_1_4999
resblock_part1_1_conv1_5002
resblock_part1_1_conv1_5004
resblock_part1_1_conv2_5008
resblock_part1_1_conv2_5010
tf_math_multiply_mul_x
resblock_part1_2_conv1_5016
resblock_part1_2_conv1_5018
resblock_part1_2_conv2_5022
resblock_part1_2_conv2_5024
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_5030
resblock_part1_3_conv1_5032
resblock_part1_3_conv2_5036
resblock_part1_3_conv2_5038
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_5044
resblock_part1_4_conv1_5046
resblock_part1_4_conv2_5050
resblock_part1_4_conv2_5052
tf_math_multiply_3_mul_x
downsampler_2_5059
downsampler_2_5061
resblock_part2_1_conv1_5064
resblock_part2_1_conv1_5066
resblock_part2_1_conv2_5070
resblock_part2_1_conv2_5072
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_5078
resblock_part2_2_conv1_5080
resblock_part2_2_conv2_5084
resblock_part2_2_conv2_5086
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_5092
resblock_part2_3_conv1_5094
resblock_part2_3_conv2_5098
resblock_part2_3_conv2_5100
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_5106
resblock_part2_4_conv1_5108
resblock_part2_4_conv2_5112
resblock_part2_4_conv2_5114
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_5120
resblock_part2_5_conv1_5122
resblock_part2_5_conv2_5126
resblock_part2_5_conv2_5128
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_5134
resblock_part2_6_conv1_5136
resblock_part2_6_conv2_5140
resblock_part2_6_conv2_5142
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_5148
resblock_part2_7_conv1_5150
resblock_part2_7_conv2_5154
resblock_part2_7_conv2_5156
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_5162
resblock_part2_8_conv1_5164
resblock_part2_8_conv2_5168
resblock_part2_8_conv2_5170
tf_math_multiply_11_mul_x
upsampler_1_5176
upsampler_1_5178
resblock_part3_1_conv1_5182
resblock_part3_1_conv1_5184
resblock_part3_1_conv2_5188
resblock_part3_1_conv2_5190
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_5196
resblock_part3_2_conv1_5198
resblock_part3_2_conv2_5202
resblock_part3_2_conv2_5204
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_5210
resblock_part3_3_conv1_5212
resblock_part3_3_conv2_5216
resblock_part3_3_conv2_5218
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_5224
resblock_part3_4_conv1_5226
resblock_part3_4_conv2_5230
resblock_part3_4_conv2_5232
tf_math_multiply_15_mul_x
extra_conv_5238
extra_conv_5240
upsampler_2_5244
upsampler_2_5246
output_conv_5250
output_conv_5252
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCall¥
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4991input_conv_4993*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_29902$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29572 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_4997downsampler_1_4999*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30172'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_5002resblock_part1_1_conv1_5004*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304320
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30642(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_5008resblock_part1_1_conv2_5010*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308220
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_5016resblock_part1_2_conv1_5018*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311120
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31322(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_5022resblock_part1_2_conv2_5024*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315020
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_5030resblock_part1_3_conv1_5032*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_317920
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32002(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_5036resblock_part1_3_conv2_5038*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_321820
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_5044resblock_part1_4_conv1_5046*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_324720
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32682(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_5050resblock_part1_4_conv2_5052*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_328620
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29702"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_5059downsampler_2_5061*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33162'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_5064resblock_part2_1_conv1_5066*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334220
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33632(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_5070resblock_part2_1_conv2_5072*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338120
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_5078resblock_part2_2_conv1_5080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341020
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34312(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_5084resblock_part2_2_conv2_5086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_344920
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_5092resblock_part2_3_conv1_5094*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_347820
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_34992(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_5098resblock_part2_3_conv2_5100*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_351720
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_5106resblock_part2_4_conv1_5108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_354620
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35672(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_5112resblock_part2_4_conv2_5114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358520
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_5120resblock_part2_5_conv1_5122*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361420
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36352(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_5126resblock_part2_5_conv2_5128*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365320
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_5134resblock_part2_6_conv1_5136*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368220
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37032(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_5140resblock_part2_6_conv2_5142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372120
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_5148resblock_part2_7_conv1_5150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375020
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37712(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_5154resblock_part2_7_conv2_5156*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_378920
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_5162resblock_part2_8_conv1_5164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_381820
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38392(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_5168resblock_part2_8_conv2_5170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_385720
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_5176upsampler_1_5178*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38862%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_5182resblock_part3_1_conv1_5184*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391320
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39342(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_5188resblock_part3_1_conv2_5190*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395220
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_5196resblock_part3_2_conv1_5198*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398120
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40022(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_5202resblock_part3_2_conv2_5204*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402020
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_5210resblock_part3_3_conv1_5212*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_404920
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40702(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_5216resblock_part3_3_conv2_5218*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_408820
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_5224resblock_part3_4_conv1_5226*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_411720
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41382(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_5230resblock_part3_4_conv2_5232*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_415620
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_5238extra_conv_5240*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_41852$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_5244upsampler_2_5246*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42122%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_5250output_conv_5252*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_42392%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
Þ
l
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_3499

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_7308

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_7481

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å"
·
+__inference_ssi_res_unet_layer_call_fn_6646

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCallò
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_52562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
æ
l
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_3132

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£

Þ
E__inference_output_conv_layer_call_and_return_conditional_losses_7538

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_3449

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
ô"
¼
+__inference_ssi_res_unet_layer_call_fn_5447
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_52562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 


5__inference_resblock_part2_2_conv1_layer_call_fn_6962

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_34102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
å"
·
+__inference_ssi_res_unet_layer_call_fn_6453

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCallò
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_47952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 


,__inference_downsampler_2_layer_call_fn_6895

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_4_conv2_layer_call_fn_7087

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_35852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
ª
I
-__inference_zero_padding2d_layer_call_fn_2963

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29572
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
f
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2970

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_4156

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_3111

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_7356

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô"
¼
+__inference_ssi_res_unet_layer_call_fn_4986
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_47952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
¤

é
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_3750

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_4117

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_3_conv1_layer_call_fn_7413

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_40492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_2_conv2_layer_call_fn_7394

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_40202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
·C
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_6260

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource0
,downsampler_1_conv2d_readvariableop_resource1
-downsampler_1_biasadd_readvariableop_resource9
5resblock_part1_1_conv1_conv2d_readvariableop_resource:
6resblock_part1_1_conv1_biasadd_readvariableop_resource9
5resblock_part1_1_conv2_conv2d_readvariableop_resource:
6resblock_part1_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_mul_x9
5resblock_part1_2_conv1_conv2d_readvariableop_resource:
6resblock_part1_2_conv1_biasadd_readvariableop_resource9
5resblock_part1_2_conv2_conv2d_readvariableop_resource:
6resblock_part1_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_1_mul_x9
5resblock_part1_3_conv1_conv2d_readvariableop_resource:
6resblock_part1_3_conv1_biasadd_readvariableop_resource9
5resblock_part1_3_conv2_conv2d_readvariableop_resource:
6resblock_part1_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_2_mul_x9
5resblock_part1_4_conv1_conv2d_readvariableop_resource:
6resblock_part1_4_conv1_biasadd_readvariableop_resource9
5resblock_part1_4_conv2_conv2d_readvariableop_resource:
6resblock_part1_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_3_mul_x0
,downsampler_2_conv2d_readvariableop_resource1
-downsampler_2_biasadd_readvariableop_resource9
5resblock_part2_1_conv1_conv2d_readvariableop_resource:
6resblock_part2_1_conv1_biasadd_readvariableop_resource9
5resblock_part2_1_conv2_conv2d_readvariableop_resource:
6resblock_part2_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_4_mul_x9
5resblock_part2_2_conv1_conv2d_readvariableop_resource:
6resblock_part2_2_conv1_biasadd_readvariableop_resource9
5resblock_part2_2_conv2_conv2d_readvariableop_resource:
6resblock_part2_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_5_mul_x9
5resblock_part2_3_conv1_conv2d_readvariableop_resource:
6resblock_part2_3_conv1_biasadd_readvariableop_resource9
5resblock_part2_3_conv2_conv2d_readvariableop_resource:
6resblock_part2_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_6_mul_x9
5resblock_part2_4_conv1_conv2d_readvariableop_resource:
6resblock_part2_4_conv1_biasadd_readvariableop_resource9
5resblock_part2_4_conv2_conv2d_readvariableop_resource:
6resblock_part2_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_7_mul_x9
5resblock_part2_5_conv1_conv2d_readvariableop_resource:
6resblock_part2_5_conv1_biasadd_readvariableop_resource9
5resblock_part2_5_conv2_conv2d_readvariableop_resource:
6resblock_part2_5_conv2_biasadd_readvariableop_resource
tf_math_multiply_8_mul_x9
5resblock_part2_6_conv1_conv2d_readvariableop_resource:
6resblock_part2_6_conv1_biasadd_readvariableop_resource9
5resblock_part2_6_conv2_conv2d_readvariableop_resource:
6resblock_part2_6_conv2_biasadd_readvariableop_resource
tf_math_multiply_9_mul_x9
5resblock_part2_7_conv1_conv2d_readvariableop_resource:
6resblock_part2_7_conv1_biasadd_readvariableop_resource9
5resblock_part2_7_conv2_conv2d_readvariableop_resource:
6resblock_part2_7_conv2_biasadd_readvariableop_resource
tf_math_multiply_10_mul_x9
5resblock_part2_8_conv1_conv2d_readvariableop_resource:
6resblock_part2_8_conv1_biasadd_readvariableop_resource9
5resblock_part2_8_conv2_conv2d_readvariableop_resource:
6resblock_part2_8_conv2_biasadd_readvariableop_resource
tf_math_multiply_11_mul_x.
*upsampler_1_conv2d_readvariableop_resource/
+upsampler_1_biasadd_readvariableop_resource9
5resblock_part3_1_conv1_conv2d_readvariableop_resource:
6resblock_part3_1_conv1_biasadd_readvariableop_resource9
5resblock_part3_1_conv2_conv2d_readvariableop_resource:
6resblock_part3_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_12_mul_x9
5resblock_part3_2_conv1_conv2d_readvariableop_resource:
6resblock_part3_2_conv1_biasadd_readvariableop_resource9
5resblock_part3_2_conv2_conv2d_readvariableop_resource:
6resblock_part3_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_13_mul_x9
5resblock_part3_3_conv1_conv2d_readvariableop_resource:
6resblock_part3_3_conv1_biasadd_readvariableop_resource9
5resblock_part3_3_conv2_conv2d_readvariableop_resource:
6resblock_part3_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_14_mul_x9
5resblock_part3_4_conv1_conv2d_readvariableop_resource:
6resblock_part3_4_conv1_biasadd_readvariableop_resource9
5resblock_part3_4_conv2_conv2d_readvariableop_resource:
6resblock_part3_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_15_mul_x-
)extra_conv_conv2d_readvariableop_resource.
*extra_conv_biasadd_readvariableop_resource.
*upsampler_2_conv2d_readvariableop_resource/
+upsampler_2_biasadd_readvariableop_resource.
*output_conv_conv2d_readvariableop_resource/
+output_conv_biasadd_readvariableop_resource
identity¢$downsampler_1/BiasAdd/ReadVariableOp¢#downsampler_1/Conv2D/ReadVariableOp¢$downsampler_2/BiasAdd/ReadVariableOp¢#downsampler_2/Conv2D/ReadVariableOp¢!extra_conv/BiasAdd/ReadVariableOp¢ extra_conv/Conv2D/ReadVariableOp¢!input_conv/BiasAdd/ReadVariableOp¢ input_conv/Conv2D/ReadVariableOp¢"output_conv/BiasAdd/ReadVariableOp¢!output_conv/Conv2D/ReadVariableOp¢-resblock_part1_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv1/Conv2D/ReadVariableOp¢-resblock_part1_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv2/Conv2D/ReadVariableOp¢-resblock_part1_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv1/Conv2D/ReadVariableOp¢-resblock_part1_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv2/Conv2D/ReadVariableOp¢-resblock_part1_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv1/Conv2D/ReadVariableOp¢-resblock_part1_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv2/Conv2D/ReadVariableOp¢-resblock_part1_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv1/Conv2D/ReadVariableOp¢-resblock_part1_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv1/Conv2D/ReadVariableOp¢-resblock_part2_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢-resblock_part2_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv1/Conv2D/ReadVariableOp¢-resblock_part2_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢-resblock_part2_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv1/Conv2D/ReadVariableOp¢-resblock_part2_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢-resblock_part2_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv1/Conv2D/ReadVariableOp¢-resblock_part2_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_5_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv1/Conv2D/ReadVariableOp¢-resblock_part2_5_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢-resblock_part2_6_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv1/Conv2D/ReadVariableOp¢-resblock_part2_6_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢-resblock_part2_7_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv1/Conv2D/ReadVariableOp¢-resblock_part2_7_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢-resblock_part2_8_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv1/Conv2D/ReadVariableOp¢-resblock_part2_8_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢-resblock_part3_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv1/Conv2D/ReadVariableOp¢-resblock_part3_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv2/Conv2D/ReadVariableOp¢-resblock_part3_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv1/Conv2D/ReadVariableOp¢-resblock_part3_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv2/Conv2D/ReadVariableOp¢-resblock_part3_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv1/Conv2D/ReadVariableOp¢-resblock_part3_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv2/Conv2D/ReadVariableOp¢-resblock_part3_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv1/Conv2D/ReadVariableOp¢-resblock_part3_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv2/Conv2D/ReadVariableOp¢"upsampler_1/BiasAdd/ReadVariableOp¢!upsampler_1/Conv2D/ReadVariableOp¢"upsampler_2/BiasAdd/ReadVariableOp¢!upsampler_2/Conv2D/ReadVariableOp¶
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 input_conv/Conv2D/ReadVariableOpÝ
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
input_conv/Conv2D­
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!input_conv/BiasAdd/ReadVariableOpÍ
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
input_conv/BiasAdd«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings®
zero_padding2d/PadPadinput_conv/BiasAdd:output:0$zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d/Pad¿
#downsampler_1/Conv2D/ReadVariableOpReadVariableOp,downsampler_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_1/Conv2D/ReadVariableOpü
downsampler_1/Conv2DConv2Dzero_padding2d/Pad:output:0+downsampler_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_1/Conv2D¶
$downsampler_1/BiasAdd/ReadVariableOpReadVariableOp-downsampler_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_1/BiasAdd/ReadVariableOpÙ
downsampler_1/BiasAddBiasAdddownsampler_1/Conv2D:output:0,downsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
downsampler_1/BiasAddÚ
,resblock_part1_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv1/Conv2D/ReadVariableOp
resblock_part1_1_conv1/Conv2DConv2Ddownsampler_1/BiasAdd:output:04resblock_part1_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv1/Conv2DÑ
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpý
resblock_part1_1_conv1/BiasAddBiasAdd&resblock_part1_1_conv1/Conv2D:output:05resblock_part1_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv1/BiasAdd§
resblock_part1_1_relu1/ReluRelu'resblock_part1_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_1_relu1/ReluÚ
,resblock_part1_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv2/Conv2D/ReadVariableOp¤
resblock_part1_1_conv2/Conv2DConv2D)resblock_part1_1_relu1/Relu:activations:04resblock_part1_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv2/Conv2DÑ
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpý
resblock_part1_1_conv2/BiasAddBiasAdd&resblock_part1_1_conv2/Conv2D:output:05resblock_part1_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv2/BiasAdd°
tf.math.multiply/MulMultf_math_multiply_mul_x'resblock_part1_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/Mul·
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2Ú
,resblock_part1_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv1/Conv2D/ReadVariableOp
resblock_part1_2_conv1/Conv2DConv2Dtf.__operators__.add/AddV2:z:04resblock_part1_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv1/Conv2DÑ
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpý
resblock_part1_2_conv1/BiasAddBiasAdd&resblock_part1_2_conv1/Conv2D:output:05resblock_part1_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv1/BiasAdd§
resblock_part1_2_relu1/ReluRelu'resblock_part1_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_2_relu1/ReluÚ
,resblock_part1_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv2/Conv2D/ReadVariableOp¤
resblock_part1_2_conv2/Conv2DConv2D)resblock_part1_2_relu1/Relu:activations:04resblock_part1_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv2/Conv2DÑ
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpý
resblock_part1_2_conv2/BiasAddBiasAdd&resblock_part1_2_conv2/Conv2D:output:05resblock_part1_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv2/BiasAdd¶
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x'resblock_part1_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2Ú
,resblock_part1_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv1/Conv2D/ReadVariableOp
resblock_part1_3_conv1/Conv2DConv2D tf.__operators__.add_1/AddV2:z:04resblock_part1_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv1/Conv2DÑ
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpý
resblock_part1_3_conv1/BiasAddBiasAdd&resblock_part1_3_conv1/Conv2D:output:05resblock_part1_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv1/BiasAdd§
resblock_part1_3_relu1/ReluRelu'resblock_part1_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_3_relu1/ReluÚ
,resblock_part1_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv2/Conv2D/ReadVariableOp¤
resblock_part1_3_conv2/Conv2DConv2D)resblock_part1_3_relu1/Relu:activations:04resblock_part1_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv2/Conv2DÑ
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpý
resblock_part1_3_conv2/BiasAddBiasAdd&resblock_part1_3_conv2/Conv2D:output:05resblock_part1_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv2/BiasAdd¶
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x'resblock_part1_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2Ú
,resblock_part1_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv1/Conv2D/ReadVariableOp
resblock_part1_4_conv1/Conv2DConv2D tf.__operators__.add_2/AddV2:z:04resblock_part1_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv1/Conv2DÑ
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpý
resblock_part1_4_conv1/BiasAddBiasAdd&resblock_part1_4_conv1/Conv2D:output:05resblock_part1_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv1/BiasAdd§
resblock_part1_4_relu1/ReluRelu'resblock_part1_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_4_relu1/ReluÚ
,resblock_part1_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv2/Conv2D/ReadVariableOp¤
resblock_part1_4_conv2/Conv2DConv2D)resblock_part1_4_relu1/Relu:activations:04resblock_part1_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv2/Conv2DÑ
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpý
resblock_part1_4_conv2/BiasAddBiasAdd&resblock_part1_4_conv2/Conv2D:output:05resblock_part1_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv2/BiasAdd¶
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x'resblock_part1_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2¯
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddings¹
zero_padding2d_1/PadPad tf.__operators__.add_3/AddV2:z:0&zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d_1/Pad¿
#downsampler_2/Conv2D/ReadVariableOpReadVariableOp,downsampler_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_2/Conv2D/ReadVariableOpü
downsampler_2/Conv2DConv2Dzero_padding2d_1/Pad:output:0+downsampler_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_2/Conv2D¶
$downsampler_2/BiasAdd/ReadVariableOpReadVariableOp-downsampler_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_2/BiasAdd/ReadVariableOp×
downsampler_2/BiasAddBiasAdddownsampler_2/Conv2D:output:0,downsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2
downsampler_2/BiasAddÚ
,resblock_part2_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv1/Conv2D/ReadVariableOp
resblock_part2_1_conv1/Conv2DConv2Ddownsampler_2/BiasAdd:output:04resblock_part2_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv1/Conv2DÑ
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpû
resblock_part2_1_conv1/BiasAddBiasAdd&resblock_part2_1_conv1/Conv2D:output:05resblock_part2_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv1/BiasAdd¥
resblock_part2_1_relu1/ReluRelu'resblock_part2_1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_1_relu1/ReluÚ
,resblock_part2_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢
resblock_part2_1_conv2/Conv2DConv2D)resblock_part2_1_relu1/Relu:activations:04resblock_part2_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv2/Conv2DÑ
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpû
resblock_part2_1_conv2/BiasAddBiasAdd&resblock_part2_1_conv2/Conv2D:output:05resblock_part2_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv2/BiasAdd´
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x'resblock_part2_1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/Mul»
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0downsampler_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2Ú
,resblock_part2_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv1/Conv2D/ReadVariableOp
resblock_part2_2_conv1/Conv2DConv2D tf.__operators__.add_4/AddV2:z:04resblock_part2_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv1/Conv2DÑ
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpû
resblock_part2_2_conv1/BiasAddBiasAdd&resblock_part2_2_conv1/Conv2D:output:05resblock_part2_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv1/BiasAdd¥
resblock_part2_2_relu1/ReluRelu'resblock_part2_2_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_2_relu1/ReluÚ
,resblock_part2_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢
resblock_part2_2_conv2/Conv2DConv2D)resblock_part2_2_relu1/Relu:activations:04resblock_part2_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv2/Conv2DÑ
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpû
resblock_part2_2_conv2/BiasAddBiasAdd&resblock_part2_2_conv2/Conv2D:output:05resblock_part2_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv2/BiasAdd´
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x'resblock_part2_2_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2Ú
,resblock_part2_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv1/Conv2D/ReadVariableOp
resblock_part2_3_conv1/Conv2DConv2D tf.__operators__.add_5/AddV2:z:04resblock_part2_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv1/Conv2DÑ
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpû
resblock_part2_3_conv1/BiasAddBiasAdd&resblock_part2_3_conv1/Conv2D:output:05resblock_part2_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv1/BiasAdd¥
resblock_part2_3_relu1/ReluRelu'resblock_part2_3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_3_relu1/ReluÚ
,resblock_part2_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢
resblock_part2_3_conv2/Conv2DConv2D)resblock_part2_3_relu1/Relu:activations:04resblock_part2_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv2/Conv2DÑ
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpû
resblock_part2_3_conv2/BiasAddBiasAdd&resblock_part2_3_conv2/Conv2D:output:05resblock_part2_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv2/BiasAdd´
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x'resblock_part2_3_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2Ú
,resblock_part2_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv1/Conv2D/ReadVariableOp
resblock_part2_4_conv1/Conv2DConv2D tf.__operators__.add_6/AddV2:z:04resblock_part2_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv1/Conv2DÑ
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpû
resblock_part2_4_conv1/BiasAddBiasAdd&resblock_part2_4_conv1/Conv2D:output:05resblock_part2_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv1/BiasAdd¥
resblock_part2_4_relu1/ReluRelu'resblock_part2_4_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_4_relu1/ReluÚ
,resblock_part2_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢
resblock_part2_4_conv2/Conv2DConv2D)resblock_part2_4_relu1/Relu:activations:04resblock_part2_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv2/Conv2DÑ
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpû
resblock_part2_4_conv2/BiasAddBiasAdd&resblock_part2_4_conv2/Conv2D:output:05resblock_part2_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv2/BiasAdd´
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x'resblock_part2_4_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2Ú
,resblock_part2_5_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv1/Conv2D/ReadVariableOp
resblock_part2_5_conv1/Conv2DConv2D tf.__operators__.add_7/AddV2:z:04resblock_part2_5_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv1/Conv2DÑ
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpû
resblock_part2_5_conv1/BiasAddBiasAdd&resblock_part2_5_conv1/Conv2D:output:05resblock_part2_5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv1/BiasAdd¥
resblock_part2_5_relu1/ReluRelu'resblock_part2_5_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_5_relu1/ReluÚ
,resblock_part2_5_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢
resblock_part2_5_conv2/Conv2DConv2D)resblock_part2_5_relu1/Relu:activations:04resblock_part2_5_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv2/Conv2DÑ
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpû
resblock_part2_5_conv2/BiasAddBiasAdd&resblock_part2_5_conv2/Conv2D:output:05resblock_part2_5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv2/BiasAdd´
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x'resblock_part2_5_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2Ú
,resblock_part2_6_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv1/Conv2D/ReadVariableOp
resblock_part2_6_conv1/Conv2DConv2D tf.__operators__.add_8/AddV2:z:04resblock_part2_6_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv1/Conv2DÑ
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpû
resblock_part2_6_conv1/BiasAddBiasAdd&resblock_part2_6_conv1/Conv2D:output:05resblock_part2_6_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv1/BiasAdd¥
resblock_part2_6_relu1/ReluRelu'resblock_part2_6_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_6_relu1/ReluÚ
,resblock_part2_6_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢
resblock_part2_6_conv2/Conv2DConv2D)resblock_part2_6_relu1/Relu:activations:04resblock_part2_6_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv2/Conv2DÑ
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpû
resblock_part2_6_conv2/BiasAddBiasAdd&resblock_part2_6_conv2/Conv2D:output:05resblock_part2_6_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv2/BiasAdd´
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x'resblock_part2_6_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2Ú
,resblock_part2_7_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv1/Conv2D/ReadVariableOp
resblock_part2_7_conv1/Conv2DConv2D tf.__operators__.add_9/AddV2:z:04resblock_part2_7_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv1/Conv2DÑ
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpû
resblock_part2_7_conv1/BiasAddBiasAdd&resblock_part2_7_conv1/Conv2D:output:05resblock_part2_7_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv1/BiasAdd¥
resblock_part2_7_relu1/ReluRelu'resblock_part2_7_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_7_relu1/ReluÚ
,resblock_part2_7_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢
resblock_part2_7_conv2/Conv2DConv2D)resblock_part2_7_relu1/Relu:activations:04resblock_part2_7_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv2/Conv2DÑ
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpû
resblock_part2_7_conv2/BiasAddBiasAdd&resblock_part2_7_conv2/Conv2D:output:05resblock_part2_7_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv2/BiasAdd·
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x'resblock_part2_7_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2Ú
,resblock_part2_8_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv1/Conv2D/ReadVariableOp
resblock_part2_8_conv1/Conv2DConv2D!tf.__operators__.add_10/AddV2:z:04resblock_part2_8_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv1/Conv2DÑ
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpû
resblock_part2_8_conv1/BiasAddBiasAdd&resblock_part2_8_conv1/Conv2D:output:05resblock_part2_8_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv1/BiasAdd¥
resblock_part2_8_relu1/ReluRelu'resblock_part2_8_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_8_relu1/ReluÚ
,resblock_part2_8_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢
resblock_part2_8_conv2/Conv2DConv2D)resblock_part2_8_relu1/Relu:activations:04resblock_part2_8_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv2/Conv2DÑ
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpû
resblock_part2_8_conv2/BiasAddBiasAdd&resblock_part2_8_conv2/Conv2D:output:05resblock_part2_8_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv2/BiasAdd·
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x'resblock_part2_8_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2º
!upsampler_1/Conv2D/ReadVariableOpReadVariableOp*upsampler_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_1/Conv2D/ReadVariableOpú
upsampler_1/Conv2DConv2D!tf.__operators__.add_11/AddV2:z:0)upsampler_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_1/Conv2D±
"upsampler_1/BiasAdd/ReadVariableOpReadVariableOp+upsampler_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_1/BiasAdd/ReadVariableOpÐ
upsampler_1/BiasAddBiasAddupsampler_1/Conv2D:output:0*upsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2
upsampler_1/BiasAddÙ
!tf.nn.depth_to_space/DepthToSpaceDepthToSpaceupsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpaceÚ
,resblock_part3_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv1/Conv2D/ReadVariableOp¥
resblock_part3_1_conv1/Conv2DConv2D*tf.nn.depth_to_space/DepthToSpace:output:04resblock_part3_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv1/Conv2DÑ
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpý
resblock_part3_1_conv1/BiasAddBiasAdd&resblock_part3_1_conv1/Conv2D:output:05resblock_part3_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv1/BiasAdd§
resblock_part3_1_relu1/ReluRelu'resblock_part3_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_1_relu1/ReluÚ
,resblock_part3_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv2/Conv2D/ReadVariableOp¤
resblock_part3_1_conv2/Conv2DConv2D)resblock_part3_1_relu1/Relu:activations:04resblock_part3_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv2/Conv2DÑ
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpý
resblock_part3_1_conv2/BiasAddBiasAdd&resblock_part3_1_conv2/Conv2D:output:05resblock_part3_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv2/BiasAdd¹
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x'resblock_part3_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2Ú
,resblock_part3_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv1/Conv2D/ReadVariableOp
resblock_part3_2_conv1/Conv2DConv2D!tf.__operators__.add_12/AddV2:z:04resblock_part3_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv1/Conv2DÑ
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpý
resblock_part3_2_conv1/BiasAddBiasAdd&resblock_part3_2_conv1/Conv2D:output:05resblock_part3_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv1/BiasAdd§
resblock_part3_2_relu1/ReluRelu'resblock_part3_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_2_relu1/ReluÚ
,resblock_part3_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv2/Conv2D/ReadVariableOp¤
resblock_part3_2_conv2/Conv2DConv2D)resblock_part3_2_relu1/Relu:activations:04resblock_part3_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv2/Conv2DÑ
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpý
resblock_part3_2_conv2/BiasAddBiasAdd&resblock_part3_2_conv2/Conv2D:output:05resblock_part3_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv2/BiasAdd¹
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x'resblock_part3_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2Ú
,resblock_part3_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv1/Conv2D/ReadVariableOp
resblock_part3_3_conv1/Conv2DConv2D!tf.__operators__.add_13/AddV2:z:04resblock_part3_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv1/Conv2DÑ
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpý
resblock_part3_3_conv1/BiasAddBiasAdd&resblock_part3_3_conv1/Conv2D:output:05resblock_part3_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv1/BiasAdd§
resblock_part3_3_relu1/ReluRelu'resblock_part3_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_3_relu1/ReluÚ
,resblock_part3_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv2/Conv2D/ReadVariableOp¤
resblock_part3_3_conv2/Conv2DConv2D)resblock_part3_3_relu1/Relu:activations:04resblock_part3_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv2/Conv2DÑ
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpý
resblock_part3_3_conv2/BiasAddBiasAdd&resblock_part3_3_conv2/Conv2D:output:05resblock_part3_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv2/BiasAdd¹
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x'resblock_part3_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2Ú
,resblock_part3_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv1/Conv2D/ReadVariableOp
resblock_part3_4_conv1/Conv2DConv2D!tf.__operators__.add_14/AddV2:z:04resblock_part3_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv1/Conv2DÑ
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpý
resblock_part3_4_conv1/BiasAddBiasAdd&resblock_part3_4_conv1/Conv2D:output:05resblock_part3_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv1/BiasAdd§
resblock_part3_4_relu1/ReluRelu'resblock_part3_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_4_relu1/ReluÚ
,resblock_part3_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv2/Conv2D/ReadVariableOp¤
resblock_part3_4_conv2/Conv2DConv2D)resblock_part3_4_relu1/Relu:activations:04resblock_part3_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv2/Conv2DÑ
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpý
resblock_part3_4_conv2/BiasAddBiasAdd&resblock_part3_4_conv2/Conv2D:output:05resblock_part3_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv2/BiasAdd¹
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x'resblock_part3_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2¶
 extra_conv/Conv2D/ReadVariableOpReadVariableOp)extra_conv_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 extra_conv/Conv2D/ReadVariableOpø
extra_conv/Conv2DConv2D!tf.__operators__.add_15/AddV2:z:0(extra_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
extra_conv/Conv2D­
!extra_conv/BiasAdd/ReadVariableOpReadVariableOp*extra_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!extra_conv/BiasAdd/ReadVariableOpÍ
extra_conv/BiasAddBiasAddextra_conv/Conv2D:output:0)extra_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
extra_conv/BiasAddÀ
tf.__operators__.add_16/AddV2AddV2extra_conv/BiasAdd:output:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2º
!upsampler_2/Conv2D/ReadVariableOpReadVariableOp*upsampler_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_2/Conv2D/ReadVariableOpü
upsampler_2/Conv2DConv2D!tf.__operators__.add_16/AddV2:z:0)upsampler_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_2/Conv2D±
"upsampler_2/BiasAdd/ReadVariableOpReadVariableOp+upsampler_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_2/BiasAdd/ReadVariableOpÒ
upsampler_2/BiasAddBiasAddupsampler_2/Conv2D:output:0*upsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
upsampler_2/BiasAddÝ
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpaceupsampler_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpace¹
!output_conv/Conv2D/ReadVariableOpReadVariableOp*output_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!output_conv/Conv2D/ReadVariableOp
output_conv/Conv2DConv2D,tf.nn.depth_to_space_1/DepthToSpace:output:0)output_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
output_conv/Conv2D°
"output_conv/BiasAdd/ReadVariableOpReadVariableOp+output_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"output_conv/BiasAdd/ReadVariableOpÑ
output_conv/BiasAddBiasAddoutput_conv/Conv2D:output:0*output_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
output_conv/BiasAddÝ
IdentityIdentityoutput_conv/BiasAdd:output:0%^downsampler_1/BiasAdd/ReadVariableOp$^downsampler_1/Conv2D/ReadVariableOp%^downsampler_2/BiasAdd/ReadVariableOp$^downsampler_2/Conv2D/ReadVariableOp"^extra_conv/BiasAdd/ReadVariableOp!^extra_conv/Conv2D/ReadVariableOp"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp#^output_conv/BiasAdd/ReadVariableOp"^output_conv/Conv2D/ReadVariableOp.^resblock_part1_1_conv1/BiasAdd/ReadVariableOp-^resblock_part1_1_conv1/Conv2D/ReadVariableOp.^resblock_part1_1_conv2/BiasAdd/ReadVariableOp-^resblock_part1_1_conv2/Conv2D/ReadVariableOp.^resblock_part1_2_conv1/BiasAdd/ReadVariableOp-^resblock_part1_2_conv1/Conv2D/ReadVariableOp.^resblock_part1_2_conv2/BiasAdd/ReadVariableOp-^resblock_part1_2_conv2/Conv2D/ReadVariableOp.^resblock_part1_3_conv1/BiasAdd/ReadVariableOp-^resblock_part1_3_conv1/Conv2D/ReadVariableOp.^resblock_part1_3_conv2/BiasAdd/ReadVariableOp-^resblock_part1_3_conv2/Conv2D/ReadVariableOp.^resblock_part1_4_conv1/BiasAdd/ReadVariableOp-^resblock_part1_4_conv1/Conv2D/ReadVariableOp.^resblock_part1_4_conv2/BiasAdd/ReadVariableOp-^resblock_part1_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_1_conv1/BiasAdd/ReadVariableOp-^resblock_part2_1_conv1/Conv2D/ReadVariableOp.^resblock_part2_1_conv2/BiasAdd/ReadVariableOp-^resblock_part2_1_conv2/Conv2D/ReadVariableOp.^resblock_part2_2_conv1/BiasAdd/ReadVariableOp-^resblock_part2_2_conv1/Conv2D/ReadVariableOp.^resblock_part2_2_conv2/BiasAdd/ReadVariableOp-^resblock_part2_2_conv2/Conv2D/ReadVariableOp.^resblock_part2_3_conv1/BiasAdd/ReadVariableOp-^resblock_part2_3_conv1/Conv2D/ReadVariableOp.^resblock_part2_3_conv2/BiasAdd/ReadVariableOp-^resblock_part2_3_conv2/Conv2D/ReadVariableOp.^resblock_part2_4_conv1/BiasAdd/ReadVariableOp-^resblock_part2_4_conv1/Conv2D/ReadVariableOp.^resblock_part2_4_conv2/BiasAdd/ReadVariableOp-^resblock_part2_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_5_conv1/BiasAdd/ReadVariableOp-^resblock_part2_5_conv1/Conv2D/ReadVariableOp.^resblock_part2_5_conv2/BiasAdd/ReadVariableOp-^resblock_part2_5_conv2/Conv2D/ReadVariableOp.^resblock_part2_6_conv1/BiasAdd/ReadVariableOp-^resblock_part2_6_conv1/Conv2D/ReadVariableOp.^resblock_part2_6_conv2/BiasAdd/ReadVariableOp-^resblock_part2_6_conv2/Conv2D/ReadVariableOp.^resblock_part2_7_conv1/BiasAdd/ReadVariableOp-^resblock_part2_7_conv1/Conv2D/ReadVariableOp.^resblock_part2_7_conv2/BiasAdd/ReadVariableOp-^resblock_part2_7_conv2/Conv2D/ReadVariableOp.^resblock_part2_8_conv1/BiasAdd/ReadVariableOp-^resblock_part2_8_conv1/Conv2D/ReadVariableOp.^resblock_part2_8_conv2/BiasAdd/ReadVariableOp-^resblock_part2_8_conv2/Conv2D/ReadVariableOp.^resblock_part3_1_conv1/BiasAdd/ReadVariableOp-^resblock_part3_1_conv1/Conv2D/ReadVariableOp.^resblock_part3_1_conv2/BiasAdd/ReadVariableOp-^resblock_part3_1_conv2/Conv2D/ReadVariableOp.^resblock_part3_2_conv1/BiasAdd/ReadVariableOp-^resblock_part3_2_conv1/Conv2D/ReadVariableOp.^resblock_part3_2_conv2/BiasAdd/ReadVariableOp-^resblock_part3_2_conv2/Conv2D/ReadVariableOp.^resblock_part3_3_conv1/BiasAdd/ReadVariableOp-^resblock_part3_3_conv1/Conv2D/ReadVariableOp.^resblock_part3_3_conv2/BiasAdd/ReadVariableOp-^resblock_part3_3_conv2/Conv2D/ReadVariableOp.^resblock_part3_4_conv1/BiasAdd/ReadVariableOp-^resblock_part3_4_conv1/Conv2D/ReadVariableOp.^resblock_part3_4_conv2/BiasAdd/ReadVariableOp-^resblock_part3_4_conv2/Conv2D/ReadVariableOp#^upsampler_1/BiasAdd/ReadVariableOp"^upsampler_1/Conv2D/ReadVariableOp#^upsampler_2/BiasAdd/ReadVariableOp"^upsampler_2/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2L
$downsampler_1/BiasAdd/ReadVariableOp$downsampler_1/BiasAdd/ReadVariableOp2J
#downsampler_1/Conv2D/ReadVariableOp#downsampler_1/Conv2D/ReadVariableOp2L
$downsampler_2/BiasAdd/ReadVariableOp$downsampler_2/BiasAdd/ReadVariableOp2J
#downsampler_2/Conv2D/ReadVariableOp#downsampler_2/Conv2D/ReadVariableOp2F
!extra_conv/BiasAdd/ReadVariableOp!extra_conv/BiasAdd/ReadVariableOp2D
 extra_conv/Conv2D/ReadVariableOp extra_conv/Conv2D/ReadVariableOp2F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2H
"output_conv/BiasAdd/ReadVariableOp"output_conv/BiasAdd/ReadVariableOp2F
!output_conv/Conv2D/ReadVariableOp!output_conv/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv1/BiasAdd/ReadVariableOp-resblock_part1_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv1/Conv2D/ReadVariableOp,resblock_part1_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv2/BiasAdd/ReadVariableOp-resblock_part1_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv2/Conv2D/ReadVariableOp,resblock_part1_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv1/BiasAdd/ReadVariableOp-resblock_part1_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv1/Conv2D/ReadVariableOp,resblock_part1_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv2/BiasAdd/ReadVariableOp-resblock_part1_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv2/Conv2D/ReadVariableOp,resblock_part1_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv1/BiasAdd/ReadVariableOp-resblock_part1_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv1/Conv2D/ReadVariableOp,resblock_part1_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv2/BiasAdd/ReadVariableOp-resblock_part1_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv2/Conv2D/ReadVariableOp,resblock_part1_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv1/BiasAdd/ReadVariableOp-resblock_part1_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv1/Conv2D/ReadVariableOp,resblock_part1_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv2/BiasAdd/ReadVariableOp-resblock_part1_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv2/Conv2D/ReadVariableOp,resblock_part1_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv1/BiasAdd/ReadVariableOp-resblock_part2_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv1/Conv2D/ReadVariableOp,resblock_part2_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv2/BiasAdd/ReadVariableOp-resblock_part2_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv2/Conv2D/ReadVariableOp,resblock_part2_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv1/BiasAdd/ReadVariableOp-resblock_part2_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv1/Conv2D/ReadVariableOp,resblock_part2_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv2/BiasAdd/ReadVariableOp-resblock_part2_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv2/Conv2D/ReadVariableOp,resblock_part2_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv1/BiasAdd/ReadVariableOp-resblock_part2_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv1/Conv2D/ReadVariableOp,resblock_part2_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv2/BiasAdd/ReadVariableOp-resblock_part2_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv2/Conv2D/ReadVariableOp,resblock_part2_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv1/BiasAdd/ReadVariableOp-resblock_part2_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv1/Conv2D/ReadVariableOp,resblock_part2_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv2/BiasAdd/ReadVariableOp-resblock_part2_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv2/Conv2D/ReadVariableOp,resblock_part2_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv1/BiasAdd/ReadVariableOp-resblock_part2_5_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv1/Conv2D/ReadVariableOp,resblock_part2_5_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv2/BiasAdd/ReadVariableOp-resblock_part2_5_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv2/Conv2D/ReadVariableOp,resblock_part2_5_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv1/BiasAdd/ReadVariableOp-resblock_part2_6_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv1/Conv2D/ReadVariableOp,resblock_part2_6_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv2/BiasAdd/ReadVariableOp-resblock_part2_6_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv2/Conv2D/ReadVariableOp,resblock_part2_6_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv1/BiasAdd/ReadVariableOp-resblock_part2_7_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv1/Conv2D/ReadVariableOp,resblock_part2_7_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv2/BiasAdd/ReadVariableOp-resblock_part2_7_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv2/Conv2D/ReadVariableOp,resblock_part2_7_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv1/BiasAdd/ReadVariableOp-resblock_part2_8_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv1/Conv2D/ReadVariableOp,resblock_part2_8_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv2/BiasAdd/ReadVariableOp-resblock_part2_8_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv2/Conv2D/ReadVariableOp,resblock_part2_8_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv1/BiasAdd/ReadVariableOp-resblock_part3_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv1/Conv2D/ReadVariableOp,resblock_part3_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv2/BiasAdd/ReadVariableOp-resblock_part3_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv2/Conv2D/ReadVariableOp,resblock_part3_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv1/BiasAdd/ReadVariableOp-resblock_part3_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv1/Conv2D/ReadVariableOp,resblock_part3_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv2/BiasAdd/ReadVariableOp-resblock_part3_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv2/Conv2D/ReadVariableOp,resblock_part3_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv1/BiasAdd/ReadVariableOp-resblock_part3_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv1/Conv2D/ReadVariableOp,resblock_part3_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv2/BiasAdd/ReadVariableOp-resblock_part3_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv2/Conv2D/ReadVariableOp,resblock_part3_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv1/BiasAdd/ReadVariableOp-resblock_part3_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv1/Conv2D/ReadVariableOp,resblock_part3_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv2/BiasAdd/ReadVariableOp-resblock_part3_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv2/Conv2D/ReadVariableOp,resblock_part3_4_conv2/Conv2D/ReadVariableOp2H
"upsampler_1/BiasAdd/ReadVariableOp"upsampler_1/BiasAdd/ReadVariableOp2F
!upsampler_1/Conv2D/ReadVariableOp!upsampler_1/Conv2D/ReadVariableOp2H
"upsampler_2/BiasAdd/ReadVariableOp"upsampler_2/BiasAdd/ReadVariableOp2F
!upsampler_2/Conv2D/ReadVariableOp!upsampler_2/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
®

é
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_7452

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨

Þ
E__inference_upsampler_2_layer_call_and_return_conditional_losses_7519

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¢
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd 
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_3363

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_7049

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_3179

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6953

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_3703

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
©é
ò%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4524
input_layer
input_conv_4259
input_conv_4261
downsampler_1_4265
downsampler_1_4267
resblock_part1_1_conv1_4270
resblock_part1_1_conv1_4272
resblock_part1_1_conv2_4276
resblock_part1_1_conv2_4278
tf_math_multiply_mul_x
resblock_part1_2_conv1_4284
resblock_part1_2_conv1_4286
resblock_part1_2_conv2_4290
resblock_part1_2_conv2_4292
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_4298
resblock_part1_3_conv1_4300
resblock_part1_3_conv2_4304
resblock_part1_3_conv2_4306
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_4312
resblock_part1_4_conv1_4314
resblock_part1_4_conv2_4318
resblock_part1_4_conv2_4320
tf_math_multiply_3_mul_x
downsampler_2_4327
downsampler_2_4329
resblock_part2_1_conv1_4332
resblock_part2_1_conv1_4334
resblock_part2_1_conv2_4338
resblock_part2_1_conv2_4340
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_4346
resblock_part2_2_conv1_4348
resblock_part2_2_conv2_4352
resblock_part2_2_conv2_4354
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_4360
resblock_part2_3_conv1_4362
resblock_part2_3_conv2_4366
resblock_part2_3_conv2_4368
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_4374
resblock_part2_4_conv1_4376
resblock_part2_4_conv2_4380
resblock_part2_4_conv2_4382
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_4388
resblock_part2_5_conv1_4390
resblock_part2_5_conv2_4394
resblock_part2_5_conv2_4396
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_4402
resblock_part2_6_conv1_4404
resblock_part2_6_conv2_4408
resblock_part2_6_conv2_4410
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_4416
resblock_part2_7_conv1_4418
resblock_part2_7_conv2_4422
resblock_part2_7_conv2_4424
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_4430
resblock_part2_8_conv1_4432
resblock_part2_8_conv2_4436
resblock_part2_8_conv2_4438
tf_math_multiply_11_mul_x
upsampler_1_4444
upsampler_1_4446
resblock_part3_1_conv1_4450
resblock_part3_1_conv1_4452
resblock_part3_1_conv2_4456
resblock_part3_1_conv2_4458
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_4464
resblock_part3_2_conv1_4466
resblock_part3_2_conv2_4470
resblock_part3_2_conv2_4472
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4478
resblock_part3_3_conv1_4480
resblock_part3_3_conv2_4484
resblock_part3_3_conv2_4486
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4492
resblock_part3_4_conv1_4494
resblock_part3_4_conv2_4498
resblock_part3_4_conv2_4500
tf_math_multiply_15_mul_x
extra_conv_4506
extra_conv_4508
upsampler_2_4512
upsampler_2_4514
output_conv_4518
output_conv_4520
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCallª
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_layerinput_conv_4259input_conv_4261*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_29902$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29572 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_4265downsampler_1_4267*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30172'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_4270resblock_part1_1_conv1_4272*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304320
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30642(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_4276resblock_part1_1_conv2_4278*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308220
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_4284resblock_part1_2_conv1_4286*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311120
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31322(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_4290resblock_part1_2_conv2_4292*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315020
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_4298resblock_part1_3_conv1_4300*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_317920
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32002(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_4304resblock_part1_3_conv2_4306*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_321820
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_4312resblock_part1_4_conv1_4314*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_324720
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32682(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_4318resblock_part1_4_conv2_4320*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_328620
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29702"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_4327downsampler_2_4329*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33162'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_4332resblock_part2_1_conv1_4334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334220
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33632(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_4338resblock_part2_1_conv2_4340*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338120
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_4346resblock_part2_2_conv1_4348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341020
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34312(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_4352resblock_part2_2_conv2_4354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_344920
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_4360resblock_part2_3_conv1_4362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_347820
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_34992(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_4366resblock_part2_3_conv2_4368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_351720
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_4374resblock_part2_4_conv1_4376*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_354620
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35672(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_4380resblock_part2_4_conv2_4382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358520
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_4388resblock_part2_5_conv1_4390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361420
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36352(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_4394resblock_part2_5_conv2_4396*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365320
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_4402resblock_part2_6_conv1_4404*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368220
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37032(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_4408resblock_part2_6_conv2_4410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372120
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_4416resblock_part2_7_conv1_4418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375020
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37712(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_4422resblock_part2_7_conv2_4424*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_378920
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_4430resblock_part2_8_conv1_4432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_381820
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38392(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_4436resblock_part2_8_conv2_4438*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_385720
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_4444upsampler_1_4446*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38862%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_4450resblock_part3_1_conv1_4452*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391320
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39342(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_4456resblock_part3_1_conv2_4458*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395220
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_4464resblock_part3_2_conv1_4466*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398120
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40022(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4470resblock_part3_2_conv2_4472*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402020
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4478resblock_part3_3_conv1_4480*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_404920
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40702(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4484resblock_part3_3_conv2_4486*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_408820
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4492resblock_part3_4_conv1_4494*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_411720
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41382(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4498resblock_part3_4_conv2_4500*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_415620
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4506extra_conv_4508*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_41852$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4512upsampler_2_4514*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42122%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4518output_conv_4520*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_42392%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
æ
l
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_6708

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_3857

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_4_relu1_layer_call_fn_7471

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41382
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_1_conv2_layer_call_fn_7346

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_39522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_7222

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_7_conv2_layer_call_fn_7231

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_37892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¢

Ý
D__inference_extra_conv_layer_call_and_return_conditional_losses_7500

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_6_conv2_layer_call_fn_7183

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_37212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_6_conv1_layer_call_fn_7154

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_36822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_3546

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_3_conv1_layer_call_fn_7010

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_34782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


*__inference_output_conv_layer_call_fn_7547

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_42392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
ÑT
__inference__wrapped_model_2950
input_layer:
6ssi_res_unet_input_conv_conv2d_readvariableop_resource;
7ssi_res_unet_input_conv_biasadd_readvariableop_resource=
9ssi_res_unet_downsampler_1_conv2d_readvariableop_resource>
:ssi_res_unet_downsampler_1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_1_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_1_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_1_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_1_conv2_biasadd_readvariableop_resource'
#ssi_res_unet_tf_math_multiply_mul_xF
Bssi_res_unet_resblock_part1_2_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_2_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_2_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_2_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_1_mul_xF
Bssi_res_unet_resblock_part1_3_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_3_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_3_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_3_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_2_mul_xF
Bssi_res_unet_resblock_part1_4_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_4_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_4_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_4_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_3_mul_x=
9ssi_res_unet_downsampler_2_conv2d_readvariableop_resource>
:ssi_res_unet_downsampler_2_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_1_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_1_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_1_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_1_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_4_mul_xF
Bssi_res_unet_resblock_part2_2_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_2_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_2_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_2_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_5_mul_xF
Bssi_res_unet_resblock_part2_3_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_3_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_3_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_3_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_6_mul_xF
Bssi_res_unet_resblock_part2_4_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_4_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_4_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_4_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_7_mul_xF
Bssi_res_unet_resblock_part2_5_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_5_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_5_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_5_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_8_mul_xF
Bssi_res_unet_resblock_part2_6_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_6_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_6_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_6_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_9_mul_xF
Bssi_res_unet_resblock_part2_7_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_7_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_7_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_7_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_10_mul_xF
Bssi_res_unet_resblock_part2_8_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_8_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_8_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_8_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_11_mul_x;
7ssi_res_unet_upsampler_1_conv2d_readvariableop_resource<
8ssi_res_unet_upsampler_1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_1_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_1_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_1_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_1_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_12_mul_xF
Bssi_res_unet_resblock_part3_2_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_2_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_2_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_2_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_13_mul_xF
Bssi_res_unet_resblock_part3_3_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_3_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_3_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_3_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_14_mul_xF
Bssi_res_unet_resblock_part3_4_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_4_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_4_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_4_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_15_mul_x:
6ssi_res_unet_extra_conv_conv2d_readvariableop_resource;
7ssi_res_unet_extra_conv_biasadd_readvariableop_resource;
7ssi_res_unet_upsampler_2_conv2d_readvariableop_resource<
8ssi_res_unet_upsampler_2_biasadd_readvariableop_resource;
7ssi_res_unet_output_conv_conv2d_readvariableop_resource<
8ssi_res_unet_output_conv_biasadd_readvariableop_resource
identity¢1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp¢0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp¢1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp¢0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp¢.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp¢-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp¢.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp¢-ssi_res_unet/input_conv/Conv2D/ReadVariableOp¢/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp¢.ssi_res_unet/output_conv/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp¢/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp¢.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp¢/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp¢.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOpÝ
-ssi_res_unet/input_conv/Conv2D/ReadVariableOpReadVariableOp6ssi_res_unet_input_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02/
-ssi_res_unet/input_conv/Conv2D/ReadVariableOp
ssi_res_unet/input_conv/Conv2DConv2Dinput_layer5ssi_res_unet/input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2 
ssi_res_unet/input_conv/Conv2DÔ
.ssi_res_unet/input_conv/BiasAdd/ReadVariableOpReadVariableOp7ssi_res_unet_input_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp
ssi_res_unet/input_conv/BiasAddBiasAdd'ssi_res_unet/input_conv/Conv2D:output:06ssi_res_unet/input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2!
ssi_res_unet/input_conv/BiasAddÅ
(ssi_res_unet/zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2*
(ssi_res_unet/zero_padding2d/Pad/paddingsâ
ssi_res_unet/zero_padding2d/PadPad(ssi_res_unet/input_conv/BiasAdd:output:01ssi_res_unet/zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
ssi_res_unet/zero_padding2d/Padæ
0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOpReadVariableOp9ssi_res_unet_downsampler_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp°
!ssi_res_unet/downsampler_1/Conv2DConv2D(ssi_res_unet/zero_padding2d/Pad:output:08ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2#
!ssi_res_unet/downsampler_1/Conv2DÝ
1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOpReadVariableOp:ssi_res_unet_downsampler_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp
"ssi_res_unet/downsampler_1/BiasAddBiasAdd*ssi_res_unet/downsampler_1/Conv2D:output:09ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2$
"ssi_res_unet/downsampler_1/BiasAdd
9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part1_1_conv1/Conv2DConv2D+ssi_res_unet/downsampler_1/BiasAdd:output:0Assi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_1_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_1_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_1_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_1_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_1_relu1/ReluRelu4ssi_res_unet/resblock_part1_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_1_relu1/Relu
9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_1_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_1_relu1/Relu:activations:0Assi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_1_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_1_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_1_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_1_conv2/BiasAddä
!ssi_res_unet/tf.math.multiply/MulMul#ssi_res_unet_tf_math_multiply_mul_x4ssi_res_unet/resblock_part1_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!ssi_res_unet/tf.math.multiply/Mulë
'ssi_res_unet/tf.__operators__.add/AddV2AddV2%ssi_res_unet/tf.math.multiply/Mul:z:0+ssi_res_unet/downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'ssi_res_unet/tf.__operators__.add/AddV2
9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part1_2_conv1/Conv2DConv2D+ssi_res_unet/tf.__operators__.add/AddV2:z:0Assi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_2_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_2_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_2_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_2_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_2_relu1/ReluRelu4ssi_res_unet/resblock_part1_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_2_relu1/Relu
9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_2_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_2_relu1/Relu:activations:0Assi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_2_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_2_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_2_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_2_conv2/BiasAddê
#ssi_res_unet/tf.math.multiply_1/MulMul%ssi_res_unet_tf_math_multiply_1_mul_x4ssi_res_unet/resblock_part1_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#ssi_res_unet/tf.math.multiply_1/Mulñ
)ssi_res_unet/tf.__operators__.add_1/AddV2AddV2'ssi_res_unet/tf.math.multiply_1/Mul:z:0+ssi_res_unet/tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)ssi_res_unet/tf.__operators__.add_1/AddV2
9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOpÏ
*ssi_res_unet/resblock_part1_3_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_1/AddV2:z:0Assi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_3_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_3_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_3_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_3_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_3_relu1/ReluRelu4ssi_res_unet/resblock_part1_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_3_relu1/Relu
9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_3_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_3_relu1/Relu:activations:0Assi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_3_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_3_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_3_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_3_conv2/BiasAddê
#ssi_res_unet/tf.math.multiply_2/MulMul%ssi_res_unet_tf_math_multiply_2_mul_x4ssi_res_unet/resblock_part1_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#ssi_res_unet/tf.math.multiply_2/Muló
)ssi_res_unet/tf.__operators__.add_2/AddV2AddV2'ssi_res_unet/tf.math.multiply_2/Mul:z:0-ssi_res_unet/tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)ssi_res_unet/tf.__operators__.add_2/AddV2
9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOpÏ
*ssi_res_unet/resblock_part1_4_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_2/AddV2:z:0Assi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_4_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_4_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_4_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_4_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_4_relu1/ReluRelu4ssi_res_unet/resblock_part1_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_4_relu1/Relu
9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_4_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_4_relu1/Relu:activations:0Assi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_4_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_4_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_4_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_4_conv2/BiasAddê
#ssi_res_unet/tf.math.multiply_3/MulMul%ssi_res_unet_tf_math_multiply_3_mul_x4ssi_res_unet/resblock_part1_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#ssi_res_unet/tf.math.multiply_3/Muló
)ssi_res_unet/tf.__operators__.add_3/AddV2AddV2'ssi_res_unet/tf.math.multiply_3/Mul:z:0-ssi_res_unet/tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)ssi_res_unet/tf.__operators__.add_3/AddV2É
*ssi_res_unet/zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2,
*ssi_res_unet/zero_padding2d_1/Pad/paddingsí
!ssi_res_unet/zero_padding2d_1/PadPad-ssi_res_unet/tf.__operators__.add_3/AddV2:z:03ssi_res_unet/zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!ssi_res_unet/zero_padding2d_1/Padæ
0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOpReadVariableOp9ssi_res_unet_downsampler_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp°
!ssi_res_unet/downsampler_2/Conv2DConv2D*ssi_res_unet/zero_padding2d_1/Pad:output:08ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2#
!ssi_res_unet/downsampler_2/Conv2DÝ
1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOpReadVariableOp:ssi_res_unet_downsampler_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp
"ssi_res_unet/downsampler_2/BiasAddBiasAdd*ssi_res_unet/downsampler_2/Conv2D:output:09ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2$
"ssi_res_unet/downsampler_2/BiasAdd
9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOpË
*ssi_res_unet/resblock_part2_1_conv1/Conv2DConv2D+ssi_res_unet/downsampler_2/BiasAdd:output:0Assi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_1_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_1_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_1_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_1_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_1_relu1/ReluRelu4ssi_res_unet/resblock_part2_1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_1_relu1/Relu
9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_1_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_1_relu1/Relu:activations:0Assi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_1_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_1_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_1_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_1_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_4/MulMul%ssi_res_unet_tf_math_multiply_4_mul_x4ssi_res_unet/resblock_part2_1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_4/Mulï
)ssi_res_unet/tf.__operators__.add_4/AddV2AddV2'ssi_res_unet/tf.math.multiply_4/Mul:z:0+ssi_res_unet/downsampler_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_4/AddV2
9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_2_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_4/AddV2:z:0Assi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_2_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_2_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_2_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_2_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_2_relu1/ReluRelu4ssi_res_unet/resblock_part2_2_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_2_relu1/Relu
9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_2_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_2_relu1/Relu:activations:0Assi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_2_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_2_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_2_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_2_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_5/MulMul%ssi_res_unet_tf_math_multiply_5_mul_x4ssi_res_unet/resblock_part2_2_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_5/Mulñ
)ssi_res_unet/tf.__operators__.add_5/AddV2AddV2'ssi_res_unet/tf.math.multiply_5/Mul:z:0-ssi_res_unet/tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_5/AddV2
9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_3_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_5/AddV2:z:0Assi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_3_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_3_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_3_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_3_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_3_relu1/ReluRelu4ssi_res_unet/resblock_part2_3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_3_relu1/Relu
9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_3_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_3_relu1/Relu:activations:0Assi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_3_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_3_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_3_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_3_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_6/MulMul%ssi_res_unet_tf_math_multiply_6_mul_x4ssi_res_unet/resblock_part2_3_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_6/Mulñ
)ssi_res_unet/tf.__operators__.add_6/AddV2AddV2'ssi_res_unet/tf.math.multiply_6/Mul:z:0-ssi_res_unet/tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_6/AddV2
9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_4_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_6/AddV2:z:0Assi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_4_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_4_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_4_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_4_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_4_relu1/ReluRelu4ssi_res_unet/resblock_part2_4_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_4_relu1/Relu
9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_4_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_4_relu1/Relu:activations:0Assi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_4_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_4_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_4_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_4_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_7/MulMul%ssi_res_unet_tf_math_multiply_7_mul_x4ssi_res_unet/resblock_part2_4_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_7/Mulñ
)ssi_res_unet/tf.__operators__.add_7/AddV2AddV2'ssi_res_unet/tf.math.multiply_7/Mul:z:0-ssi_res_unet/tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_7/AddV2
9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_5_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_5_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_7/AddV2:z:0Assi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_5_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_5_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_5_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_5_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_5_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_5_relu1/ReluRelu4ssi_res_unet/resblock_part2_5_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_5_relu1/Relu
9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_5_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_5_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_5_relu1/Relu:activations:0Assi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_5_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_5_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_5_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_5_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_5_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_8/MulMul%ssi_res_unet_tf_math_multiply_8_mul_x4ssi_res_unet/resblock_part2_5_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_8/Mulñ
)ssi_res_unet/tf.__operators__.add_8/AddV2AddV2'ssi_res_unet/tf.math.multiply_8/Mul:z:0-ssi_res_unet/tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_8/AddV2
9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_6_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_6_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_8/AddV2:z:0Assi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_6_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_6_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_6_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_6_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_6_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_6_relu1/ReluRelu4ssi_res_unet/resblock_part2_6_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_6_relu1/Relu
9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_6_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_6_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_6_relu1/Relu:activations:0Assi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_6_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_6_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_6_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_6_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_6_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_9/MulMul%ssi_res_unet_tf_math_multiply_9_mul_x4ssi_res_unet/resblock_part2_6_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_9/Mulñ
)ssi_res_unet/tf.__operators__.add_9/AddV2AddV2'ssi_res_unet/tf.math.multiply_9/Mul:z:0-ssi_res_unet/tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_9/AddV2
9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_7_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_7_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_9/AddV2:z:0Assi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_7_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_7_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_7_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_7_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_7_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_7_relu1/ReluRelu4ssi_res_unet/resblock_part2_7_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_7_relu1/Relu
9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_7_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_7_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_7_relu1/Relu:activations:0Assi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_7_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_7_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_7_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_7_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_7_conv2/BiasAddë
$ssi_res_unet/tf.math.multiply_10/MulMul&ssi_res_unet_tf_math_multiply_10_mul_x4ssi_res_unet/resblock_part2_7_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2&
$ssi_res_unet/tf.math.multiply_10/Mulô
*ssi_res_unet/tf.__operators__.add_10/AddV2AddV2(ssi_res_unet/tf.math.multiply_10/Mul:z:0-ssi_res_unet/tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2,
*ssi_res_unet/tf.__operators__.add_10/AddV2
9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOpÎ
*ssi_res_unet/resblock_part2_8_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_10/AddV2:z:0Assi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_8_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_8_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_8_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_8_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_8_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_8_relu1/ReluRelu4ssi_res_unet/resblock_part2_8_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_8_relu1/Relu
9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_8_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_8_relu1/Relu:activations:0Assi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_8_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_8_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_8_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_8_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_8_conv2/BiasAddë
$ssi_res_unet/tf.math.multiply_11/MulMul&ssi_res_unet_tf_math_multiply_11_mul_x4ssi_res_unet/resblock_part2_8_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2&
$ssi_res_unet/tf.math.multiply_11/Mulõ
*ssi_res_unet/tf.__operators__.add_11/AddV2AddV2(ssi_res_unet/tf.math.multiply_11/Mul:z:0.ssi_res_unet/tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2,
*ssi_res_unet/tf.__operators__.add_11/AddV2á
.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOpReadVariableOp7ssi_res_unet_upsampler_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype020
.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp®
ssi_res_unet/upsampler_1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_11/AddV2:z:06ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2!
ssi_res_unet/upsampler_1/Conv2DØ
/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOpReadVariableOp8ssi_res_unet_upsampler_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp
 ssi_res_unet/upsampler_1/BiasAddBiasAdd(ssi_res_unet/upsampler_1/Conv2D:output:07ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2"
 ssi_res_unet/upsampler_1/BiasAdd
.ssi_res_unet/tf.nn.depth_to_space/DepthToSpaceDepthToSpace)ssi_res_unet/upsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW20
.ssi_res_unet/tf.nn.depth_to_space/DepthToSpace
9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOpÙ
*ssi_res_unet/resblock_part3_1_conv1/Conv2DConv2D7ssi_res_unet/tf.nn.depth_to_space/DepthToSpace:output:0Assi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_1_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_1_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_1_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_1_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_1_relu1/ReluRelu4ssi_res_unet/resblock_part3_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_1_relu1/Relu
9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_1_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_1_relu1/Relu:activations:0Assi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_1_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_1_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_1_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_1_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_12/MulMul&ssi_res_unet_tf_math_multiply_12_mul_x4ssi_res_unet/resblock_part3_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_12/Mul
*ssi_res_unet/tf.__operators__.add_12/AddV2AddV2(ssi_res_unet/tf.math.multiply_12/Mul:z:07ssi_res_unet/tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_12/AddV2
9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOpÐ
*ssi_res_unet/resblock_part3_2_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_12/AddV2:z:0Assi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_2_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_2_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_2_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_2_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_2_relu1/ReluRelu4ssi_res_unet/resblock_part3_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_2_relu1/Relu
9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_2_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_2_relu1/Relu:activations:0Assi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_2_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_2_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_2_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_2_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_13/MulMul&ssi_res_unet_tf_math_multiply_13_mul_x4ssi_res_unet/resblock_part3_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_13/Mul÷
*ssi_res_unet/tf.__operators__.add_13/AddV2AddV2(ssi_res_unet/tf.math.multiply_13/Mul:z:0.ssi_res_unet/tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_13/AddV2
9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOpÐ
*ssi_res_unet/resblock_part3_3_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_13/AddV2:z:0Assi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_3_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_3_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_3_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_3_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_3_relu1/ReluRelu4ssi_res_unet/resblock_part3_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_3_relu1/Relu
9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_3_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_3_relu1/Relu:activations:0Assi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_3_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_3_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_3_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_3_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_14/MulMul&ssi_res_unet_tf_math_multiply_14_mul_x4ssi_res_unet/resblock_part3_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_14/Mul÷
*ssi_res_unet/tf.__operators__.add_14/AddV2AddV2(ssi_res_unet/tf.math.multiply_14/Mul:z:0.ssi_res_unet/tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_14/AddV2
9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOpÐ
*ssi_res_unet/resblock_part3_4_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_14/AddV2:z:0Assi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_4_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_4_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_4_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_4_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_4_relu1/ReluRelu4ssi_res_unet/resblock_part3_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_4_relu1/Relu
9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_4_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_4_relu1/Relu:activations:0Assi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_4_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_4_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_4_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_4_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_15/MulMul&ssi_res_unet_tf_math_multiply_15_mul_x4ssi_res_unet/resblock_part3_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_15/Mul÷
*ssi_res_unet/tf.__operators__.add_15/AddV2AddV2(ssi_res_unet/tf.math.multiply_15/Mul:z:0.ssi_res_unet/tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_15/AddV2Ý
-ssi_res_unet/extra_conv/Conv2D/ReadVariableOpReadVariableOp6ssi_res_unet_extra_conv_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp¬
ssi_res_unet/extra_conv/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_15/AddV2:z:05ssi_res_unet/extra_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2 
ssi_res_unet/extra_conv/Conv2DÔ
.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOpReadVariableOp7ssi_res_unet_extra_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp
ssi_res_unet/extra_conv/BiasAddBiasAdd'ssi_res_unet/extra_conv/Conv2D:output:06ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2!
ssi_res_unet/extra_conv/BiasAddô
*ssi_res_unet/tf.__operators__.add_16/AddV2AddV2(ssi_res_unet/extra_conv/BiasAdd:output:0+ssi_res_unet/downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_16/AddV2á
.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOpReadVariableOp7ssi_res_unet_upsampler_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype020
.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp°
ssi_res_unet/upsampler_2/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_16/AddV2:z:06ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2!
ssi_res_unet/upsampler_2/Conv2DØ
/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOpReadVariableOp8ssi_res_unet_upsampler_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp
 ssi_res_unet/upsampler_2/BiasAddBiasAdd(ssi_res_unet/upsampler_2/Conv2D:output:07ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2"
 ssi_res_unet/upsampler_2/BiasAdd
0ssi_res_unet/tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace)ssi_res_unet/upsampler_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW22
0ssi_res_unet/tf.nn.depth_to_space_1/DepthToSpaceà
.ssi_res_unet/output_conv/Conv2D/ReadVariableOpReadVariableOp7ssi_res_unet_output_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.ssi_res_unet/output_conv/Conv2D/ReadVariableOpº
ssi_res_unet/output_conv/Conv2DConv2D9ssi_res_unet/tf.nn.depth_to_space_1/DepthToSpace:output:06ssi_res_unet/output_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2!
ssi_res_unet/output_conv/Conv2D×
/ssi_res_unet/output_conv/BiasAdd/ReadVariableOpReadVariableOp8ssi_res_unet_output_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp
 ssi_res_unet/output_conv/BiasAddBiasAdd(ssi_res_unet/output_conv/Conv2D:output:07ssi_res_unet/output_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2"
 ssi_res_unet/output_conv/BiasAddà$
IdentityIdentity)ssi_res_unet/output_conv/BiasAdd:output:02^ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp1^ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp2^ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp1^ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp/^ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp.^ssi_res_unet/extra_conv/Conv2D/ReadVariableOp/^ssi_res_unet/input_conv/BiasAdd/ReadVariableOp.^ssi_res_unet/input_conv/Conv2D/ReadVariableOp0^ssi_res_unet/output_conv/BiasAdd/ReadVariableOp/^ssi_res_unet/output_conv/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp0^ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp/^ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp0^ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp/^ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2f
1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp2d
0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp2f
1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp2d
0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp2`
.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp2^
-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp2`
.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp2^
-ssi_res_unet/input_conv/Conv2D/ReadVariableOp-ssi_res_unet/input_conv/Conv2D/ReadVariableOp2b
/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp2`
.ssi_res_unet/output_conv/Conv2D/ReadVariableOp.ssi_res_unet/output_conv/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp2b
/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp2`
.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp2b
/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp2`
.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
Í
Q
5__inference_resblock_part2_2_relu1_layer_call_fn_6972

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34312
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


*__inference_upsampler_2_layer_call_fn_7528

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_1_conv1_layer_call_fn_6703

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_30432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_3567

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_3_relu1_layer_call_fn_7423

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40702
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_6838

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_3247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_1_relu1_layer_call_fn_6713

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30642
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_1_conv1_layer_call_fn_6914

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_33422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ê
serving_default¶
M
input_layer>
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿI
output_conv:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:è
Þ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer-24
layer_with_weights-10
layer-25
layer_with_weights-11
layer-26
layer-27
layer_with_weights-12
layer-28
layer-29
layer-30
 layer_with_weights-13
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$layer-35
%layer_with_weights-15
%layer-36
&layer-37
'layer_with_weights-16
'layer-38
(layer-39
)layer-40
*layer_with_weights-17
*layer-41
+layer-42
,layer_with_weights-18
,layer-43
-layer-44
.layer-45
/layer_with_weights-19
/layer-46
0layer-47
1layer_with_weights-20
1layer-48
2layer-49
3layer-50
4layer_with_weights-21
4layer-51
5layer-52
6layer_with_weights-22
6layer-53
7layer-54
8layer-55
9layer_with_weights-23
9layer-56
:layer-57
;layer_with_weights-24
;layer-58
<layer-59
=layer-60
>layer_with_weights-25
>layer-61
?layer-62
@layer_with_weights-26
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-27
Clayer-66
Dlayer-67
Elayer_with_weights-28
Elayer-68
Flayer-69
Glayer_with_weights-29
Glayer-70
Hlayer-71
Ilayer-72
Jlayer_with_weights-30
Jlayer-73
Klayer-74
Llayer_with_weights-31
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-32
Olayer-78
Player-79
Qlayer_with_weights-33
Qlayer-80
Rlayer-81
Slayer-82
Tlayer_with_weights-34
Tlayer-83
Ulayer-84
Vlayer_with_weights-35
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-36
Ylayer-88
Zlayer-89
[layer_with_weights-37
[layer-90
\layer-91
]layer_with_weights-38
]layer-92
^trainable_variables
_regularization_losses
`	variables
a	keras_api
b
signatures
Ú_default_save_signature
+Û&call_and_return_all_conditional_losses
Ü__call__"´Ê
_tf_keras_networkÊ{"class_name": "Functional", "name": "ssi_res_unet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "ssi_res_unet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "input_conv", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d", "inbound_nodes": [[["input_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_1", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv1", "inbound_nodes": [[["downsampler_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_1_relu1", "inbound_nodes": [[["resblock_part1_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv2", "inbound_nodes": [[["resblock_part1_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_2_relu1", "inbound_nodes": [[["resblock_part1_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv2", "inbound_nodes": [[["resblock_part1_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_3_relu1", "inbound_nodes": [[["resblock_part1_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv2", "inbound_nodes": [[["resblock_part1_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.__operators__.add_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv1", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_4_relu1", "inbound_nodes": [[["resblock_part1_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv2", "inbound_nodes": [[["resblock_part1_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": ["tf.__operators__.add_2", 0, 0], "name": null}]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d_1", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_2", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv1", "inbound_nodes": [[["downsampler_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_1_relu1", "inbound_nodes": [[["resblock_part2_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv2", "inbound_nodes": [[["resblock_part2_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": ["downsampler_2", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv1", "inbound_nodes": [[["tf.__operators__.add_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_2_relu1", "inbound_nodes": [[["resblock_part2_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv2", "inbound_nodes": [[["resblock_part2_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"y": ["tf.__operators__.add_4", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv1", "inbound_nodes": [[["tf.__operators__.add_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_3_relu1", "inbound_nodes": [[["resblock_part2_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv2", "inbound_nodes": [[["resblock_part2_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"y": ["tf.__operators__.add_5", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv1", "inbound_nodes": [[["tf.__operators__.add_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_4_relu1", "inbound_nodes": [[["resblock_part2_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv2", "inbound_nodes": [[["resblock_part2_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"y": ["tf.__operators__.add_6", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv1", "inbound_nodes": [[["tf.__operators__.add_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_5_relu1", "inbound_nodes": [[["resblock_part2_5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv2", "inbound_nodes": [[["resblock_part2_5_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_5_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["tf.__operators__.add_7", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv1", "inbound_nodes": [[["tf.__operators__.add_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_6_relu1", "inbound_nodes": [[["resblock_part2_6_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv2", "inbound_nodes": [[["resblock_part2_6_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_6_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["tf.__operators__.add_8", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv1", "inbound_nodes": [[["tf.__operators__.add_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_7_relu1", "inbound_nodes": [[["resblock_part2_7_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv2", "inbound_nodes": [[["resblock_part2_7_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_7_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"y": ["tf.__operators__.add_9", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv1", "inbound_nodes": [[["tf.__operators__.add_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_8_relu1", "inbound_nodes": [[["resblock_part2_8_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv2", "inbound_nodes": [[["resblock_part2_8_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_8_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"y": ["tf.__operators__.add_10", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_1", "inbound_nodes": [[["tf.__operators__.add_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space", "inbound_nodes": [["upsampler_1", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv1", "inbound_nodes": [[["tf.nn.depth_to_space", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_1_relu1", "inbound_nodes": [[["resblock_part3_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv2", "inbound_nodes": [[["resblock_part3_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_12", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["tf.math.multiply_12", 0, 0, {"y": ["tf.nn.depth_to_space", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv1", "inbound_nodes": [[["tf.__operators__.add_12", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_2_relu1", "inbound_nodes": [[["resblock_part3_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv2", "inbound_nodes": [[["resblock_part3_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_13", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.math.multiply_13", 0, 0, {"y": ["tf.__operators__.add_12", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv1", "inbound_nodes": [[["tf.__operators__.add_13", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_3_relu1", "inbound_nodes": [[["resblock_part3_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv2", "inbound_nodes": [[["resblock_part3_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_14", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["tf.math.multiply_14", 0, 0, {"y": ["tf.__operators__.add_13", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv1", "inbound_nodes": [[["tf.__operators__.add_14", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_4_relu1", "inbound_nodes": [[["resblock_part3_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv2", "inbound_nodes": [[["resblock_part3_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.multiply_15", 0, 0, {"y": ["tf.__operators__.add_14", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "extra_conv", "inbound_nodes": [[["tf.__operators__.add_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["extra_conv", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_2", "inbound_nodes": [[["tf.__operators__.add_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space_1", "inbound_nodes": [["upsampler_2", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_conv", "inbound_nodes": [[["tf.nn.depth_to_space_1", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_conv", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 256, 256]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "ssi_res_unet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "input_conv", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d", "inbound_nodes": [[["input_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_1", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv1", "inbound_nodes": [[["downsampler_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_1_relu1", "inbound_nodes": [[["resblock_part1_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv2", "inbound_nodes": [[["resblock_part1_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_2_relu1", "inbound_nodes": [[["resblock_part1_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv2", "inbound_nodes": [[["resblock_part1_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_3_relu1", "inbound_nodes": [[["resblock_part1_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv2", "inbound_nodes": [[["resblock_part1_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.__operators__.add_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv1", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_4_relu1", "inbound_nodes": [[["resblock_part1_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv2", "inbound_nodes": [[["resblock_part1_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": ["tf.__operators__.add_2", 0, 0], "name": null}]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d_1", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_2", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv1", "inbound_nodes": [[["downsampler_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_1_relu1", "inbound_nodes": [[["resblock_part2_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv2", "inbound_nodes": [[["resblock_part2_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": ["downsampler_2", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv1", "inbound_nodes": [[["tf.__operators__.add_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_2_relu1", "inbound_nodes": [[["resblock_part2_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv2", "inbound_nodes": [[["resblock_part2_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"y": ["tf.__operators__.add_4", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv1", "inbound_nodes": [[["tf.__operators__.add_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_3_relu1", "inbound_nodes": [[["resblock_part2_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv2", "inbound_nodes": [[["resblock_part2_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"y": ["tf.__operators__.add_5", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv1", "inbound_nodes": [[["tf.__operators__.add_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_4_relu1", "inbound_nodes": [[["resblock_part2_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv2", "inbound_nodes": [[["resblock_part2_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"y": ["tf.__operators__.add_6", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv1", "inbound_nodes": [[["tf.__operators__.add_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_5_relu1", "inbound_nodes": [[["resblock_part2_5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv2", "inbound_nodes": [[["resblock_part2_5_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_5_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["tf.__operators__.add_7", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv1", "inbound_nodes": [[["tf.__operators__.add_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_6_relu1", "inbound_nodes": [[["resblock_part2_6_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv2", "inbound_nodes": [[["resblock_part2_6_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_6_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["tf.__operators__.add_8", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv1", "inbound_nodes": [[["tf.__operators__.add_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_7_relu1", "inbound_nodes": [[["resblock_part2_7_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv2", "inbound_nodes": [[["resblock_part2_7_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_7_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"y": ["tf.__operators__.add_9", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv1", "inbound_nodes": [[["tf.__operators__.add_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_8_relu1", "inbound_nodes": [[["resblock_part2_8_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv2", "inbound_nodes": [[["resblock_part2_8_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_8_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"y": ["tf.__operators__.add_10", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_1", "inbound_nodes": [[["tf.__operators__.add_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space", "inbound_nodes": [["upsampler_1", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv1", "inbound_nodes": [[["tf.nn.depth_to_space", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_1_relu1", "inbound_nodes": [[["resblock_part3_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv2", "inbound_nodes": [[["resblock_part3_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_12", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["tf.math.multiply_12", 0, 0, {"y": ["tf.nn.depth_to_space", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv1", "inbound_nodes": [[["tf.__operators__.add_12", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_2_relu1", "inbound_nodes": [[["resblock_part3_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv2", "inbound_nodes": [[["resblock_part3_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_13", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.math.multiply_13", 0, 0, {"y": ["tf.__operators__.add_12", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv1", "inbound_nodes": [[["tf.__operators__.add_13", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_3_relu1", "inbound_nodes": [[["resblock_part3_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv2", "inbound_nodes": [[["resblock_part3_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_14", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["tf.math.multiply_14", 0, 0, {"y": ["tf.__operators__.add_13", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv1", "inbound_nodes": [[["tf.__operators__.add_14", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_4_relu1", "inbound_nodes": [[["resblock_part3_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv2", "inbound_nodes": [[["resblock_part3_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.multiply_15", 0, 0, {"y": ["tf.__operators__.add_14", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "extra_conv", "inbound_nodes": [[["tf.__operators__.add_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["extra_conv", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_2", "inbound_nodes": [[["tf.__operators__.add_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space_1", "inbound_nodes": [["upsampler_2", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_conv", "inbound_nodes": [[["tf.nn.depth_to_space_1", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_conv", 0, 0]]}}}
"
_tf_keras_input_layerä{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
ý	

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"Ö
_tf_keras_layer¼{"class_name": "Conv2D", "name": "input_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 28}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 256, 256]}}

i	variables
jtrainable_variables
kregularization_losses
l	keras_api
+ß&call_and_return_all_conditional_losses
à__call__"÷
_tf_keras_layerÝ{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
+á&call_and_return_all_conditional_losses
â__call__"Ý
_tf_keras_layerÃ{"class_name": "Conv2D", "name": "downsampler_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 258, 258]}}



skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
+ã&call_and_return_all_conditional_losses
ä__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

y	variables
ztrainable_variables
{regularization_losses
|	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}



}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api
+ç&call_and_return_all_conditional_losses
è__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ç
	keras_api"Ô
_tf_keras_layerº{"class_name": "TFOpLambda", "name": "tf.math.multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ó
	keras_api"à
_tf_keras_layerÆ{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+é&call_and_return_all_conditional_losses
ê__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+í&call_and_return_all_conditional_losses
î__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
 	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


¡kernel
	¢bias
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ë
§	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
¨	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

¯	variables
°trainable_variables
±regularization_losses
²	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
+ù&call_and_return_all_conditional_losses
ú__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ë
¹	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
º	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}

»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
+û&call_and_return_all_conditional_losses
ü__call__"û
_tf_keras_layerá{"class_name": "ZeroPadding2D", "name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}


¿kernel
	Àbias
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"Ý
_tf_keras_layerÃ{"class_name": "Conv2D", "name": "downsampler_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 130, 130]}}


Åkernel
	Æbias
Ç	variables
Ètrainable_variables
Éregularization_losses
Ê	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
+&call_and_return_all_conditional_losses
__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


Ïkernel
	Ðbias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
Õ	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
Ö	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


×kernel
	Øbias
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
+&call_and_return_all_conditional_losses
__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ákernel
	âbias
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
ç	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
è	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


ékernel
	êbias
ë	variables
ìtrainable_variables
íregularization_losses
î	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
+&call_and_return_all_conditional_losses
__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ókernel
	ôbias
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
ù	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
ú	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


ûkernel
	übias
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_5_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_5_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_5_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	 bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_6_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
+&call_and_return_all_conditional_losses
 __call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_6_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
+¡&call_and_return_all_conditional_losses
¢__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_6_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
¯	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
°	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


±kernel
	²bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
+£&call_and_return_all_conditional_losses
¤__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_7_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
+¥&call_and_return_all_conditional_losses
¦__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_7_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


»kernel
	¼bias
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_7_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
í
Á	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
Â	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


Ãkernel
	Äbias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_8_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_8_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


Íkernel
	Îbias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
+­&call_and_return_all_conditional_losses
®__call__"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_8_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
í
Ó	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
Ô	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


Õkernel
	Öbias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"×
_tf_keras_layer½{"class_name": "Conv2D", "name": "upsampler_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ó
Û	keras_api"à
_tf_keras_layerÆ{"class_name": "TFOpLambda", "name": "tf.nn.depth_to_space", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}}


Ükernel
	Ýbias
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
+±&call_and_return_all_conditional_losses
²__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
+³&call_and_return_all_conditional_losses
´__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ækernel
	çbias
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
ì	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
í	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


îkernel
	ïbias
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


økernel
	ùbias
ú	variables
ûtrainable_variables
üregularization_losses
ý	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
þ	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
ÿ	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
 regularization_losses
¡	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
¢	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
£	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


¤kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"Ö
_tf_keras_layer¼{"class_name": "Conv2D", "name": "extra_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ù
ª	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


«kernel
	¬bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"Ù
_tf_keras_layer¿{"class_name": "Conv2D", "name": "upsampler_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
÷
±	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.nn.depth_to_space_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}}


²kernel
	³bias
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"Ø
_tf_keras_layer¾{"class_name": "Conv2D", "name": "output_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 256, 256]}}
Ì
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77"
trackable_list_wrapper
 "
trackable_list_wrapper
Ì
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77"
trackable_list_wrapper
Ó
¸metrics
¹layers
^trainable_variables
ºlayer_metrics
_regularization_losses
`	variables
 »layer_regularization_losses
¼non_trainable_variables
Ü__call__
Ú_default_save_signature
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
-
Ïserving_default"
signature_map
+:)@2input_conv/kernel
:@2input_conv/bias
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
µ
e	variables
½layers
ftrainable_variables
¾layer_metrics
gregularization_losses
¿metrics
 Àlayer_regularization_losses
Ánon_trainable_variables
Þ__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
i	variables
Âlayers
jtrainable_variables
Ãlayer_metrics
kregularization_losses
Ämetrics
 Ålayer_regularization_losses
Ænon_trainable_variables
à__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
.:,@@2downsampler_1/kernel
 :@2downsampler_1/bias
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
µ
o	variables
Çlayers
ptrainable_variables
Èlayer_metrics
qregularization_losses
Émetrics
 Êlayer_regularization_losses
Ënon_trainable_variables
â__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_1_conv1/kernel
):'@2resblock_part1_1_conv1/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
u	variables
Ìlayers
vtrainable_variables
Ílayer_metrics
wregularization_losses
Îmetrics
 Ïlayer_regularization_losses
Ðnon_trainable_variables
ä__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
y	variables
Ñlayers
ztrainable_variables
Òlayer_metrics
{regularization_losses
Ómetrics
 Ôlayer_regularization_losses
Õnon_trainable_variables
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_1_conv2/kernel
):'@2resblock_part1_1_conv2/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
	variables
Ölayers
trainable_variables
×layer_metrics
regularization_losses
Ømetrics
 Ùlayer_regularization_losses
Únon_trainable_variables
è__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part1_2_conv1/kernel
):'@2resblock_part1_2_conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Ûlayers
trainable_variables
Ülayer_metrics
regularization_losses
Ýmetrics
 Þlayer_regularization_losses
ßnon_trainable_variables
ê__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
àlayers
trainable_variables
álayer_metrics
regularization_losses
âmetrics
 ãlayer_regularization_losses
änon_trainable_variables
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_2_conv2/kernel
):'@2resblock_part1_2_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
ålayers
trainable_variables
ælayer_metrics
regularization_losses
çmetrics
 èlayer_regularization_losses
énon_trainable_variables
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part1_3_conv1/kernel
):'@2resblock_part1_3_conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
êlayers
trainable_variables
ëlayer_metrics
regularization_losses
ìmetrics
 ílayer_regularization_losses
înon_trainable_variables
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
ïlayers
trainable_variables
ðlayer_metrics
regularization_losses
ñmetrics
 òlayer_regularization_losses
ónon_trainable_variables
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_3_conv2/kernel
):'@2resblock_part1_3_conv2/bias
0
¡0
¢1"
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
£	variables
ôlayers
¤trainable_variables
õlayer_metrics
¥regularization_losses
ömetrics
 ÷layer_regularization_losses
ønon_trainable_variables
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part1_4_conv1/kernel
):'@2resblock_part1_4_conv1/bias
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«	variables
ùlayers
¬trainable_variables
úlayer_metrics
­regularization_losses
ûmetrics
 ülayer_regularization_losses
ýnon_trainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯	variables
þlayers
°trainable_variables
ÿlayer_metrics
±regularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_4_conv2/kernel
):'@2resblock_part1_4_conv2/bias
0
³0
´1"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µ	variables
layers
¶trainable_variables
layer_metrics
·regularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
ú__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»	variables
layers
¼trainable_variables
layer_metrics
½regularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
ü__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
.:,@@2downsampler_2/kernel
 :@2downsampler_2/bias
0
¿0
À1"
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Á	variables
layers
Âtrainable_variables
layer_metrics
Ãregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_1_conv1/kernel
):'@2resblock_part2_1_conv1/bias
0
Å0
Æ1"
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ç	variables
layers
Ètrainable_variables
layer_metrics
Éregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ë	variables
layers
Ìtrainable_variables
layer_metrics
Íregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_1_conv2/kernel
):'@2resblock_part2_1_conv2/bias
0
Ï0
Ð1"
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñ	variables
layers
Òtrainable_variables
layer_metrics
Óregularization_losses
metrics
 layer_regularization_losses
 non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_2_conv1/kernel
):'@2resblock_part2_2_conv1/bias
0
×0
Ø1"
trackable_list_wrapper
0
×0
Ø1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ù	variables
¡layers
Útrainable_variables
¢layer_metrics
Ûregularization_losses
£metrics
 ¤layer_regularization_losses
¥non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ý	variables
¦layers
Þtrainable_variables
§layer_metrics
ßregularization_losses
¨metrics
 ©layer_regularization_losses
ªnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_2_conv2/kernel
):'@2resblock_part2_2_conv2/bias
0
á0
â1"
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ã	variables
«layers
ätrainable_variables
¬layer_metrics
åregularization_losses
­metrics
 ®layer_regularization_losses
¯non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_3_conv1/kernel
):'@2resblock_part2_3_conv1/bias
0
é0
ê1"
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ë	variables
°layers
ìtrainable_variables
±layer_metrics
íregularization_losses
²metrics
 ³layer_regularization_losses
´non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ï	variables
µlayers
ðtrainable_variables
¶layer_metrics
ñregularization_losses
·metrics
 ¸layer_regularization_losses
¹non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_3_conv2/kernel
):'@2resblock_part2_3_conv2/bias
0
ó0
ô1"
trackable_list_wrapper
0
ó0
ô1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õ	variables
ºlayers
ötrainable_variables
»layer_metrics
÷regularization_losses
¼metrics
 ½layer_regularization_losses
¾non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_4_conv1/kernel
):'@2resblock_part2_4_conv1/bias
0
û0
ü1"
trackable_list_wrapper
0
û0
ü1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ý	variables
¿layers
þtrainable_variables
Àlayer_metrics
ÿregularization_losses
Ámetrics
 Âlayer_regularization_losses
Ãnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Älayers
trainable_variables
Ålayer_metrics
regularization_losses
Æmetrics
 Çlayer_regularization_losses
Ènon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_4_conv2/kernel
):'@2resblock_part2_4_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Élayers
trainable_variables
Êlayer_metrics
regularization_losses
Ëmetrics
 Ìlayer_regularization_losses
Ínon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_5_conv1/kernel
):'@2resblock_part2_5_conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Îlayers
trainable_variables
Ïlayer_metrics
regularization_losses
Ðmetrics
 Ñlayer_regularization_losses
Ònon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Ólayers
trainable_variables
Ôlayer_metrics
regularization_losses
Õmetrics
 Ölayer_regularization_losses
×non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_5_conv2/kernel
):'@2resblock_part2_5_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Ølayers
trainable_variables
Ùlayer_metrics
regularization_losses
Úmetrics
 Ûlayer_regularization_losses
Ünon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_6_conv1/kernel
):'@2resblock_part2_6_conv1/bias
0
0
 1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡	variables
Ýlayers
¢trainable_variables
Þlayer_metrics
£regularization_losses
ßmetrics
 àlayer_regularization_losses
ánon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¥	variables
âlayers
¦trainable_variables
ãlayer_metrics
§regularization_losses
ämetrics
 ålayer_regularization_losses
ænon_trainable_variables
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_6_conv2/kernel
):'@2resblock_part2_6_conv2/bias
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«	variables
çlayers
¬trainable_variables
èlayer_metrics
­regularization_losses
émetrics
 êlayer_regularization_losses
ënon_trainable_variables
¢__call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_7_conv1/kernel
):'@2resblock_part2_7_conv1/bias
0
±0
²1"
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³	variables
ìlayers
´trainable_variables
ílayer_metrics
µregularization_losses
îmetrics
 ïlayer_regularization_losses
ðnon_trainable_variables
¤__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·	variables
ñlayers
¸trainable_variables
òlayer_metrics
¹regularization_losses
ómetrics
 ôlayer_regularization_losses
õnon_trainable_variables
¦__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_7_conv2/kernel
):'@2resblock_part2_7_conv2/bias
0
»0
¼1"
trackable_list_wrapper
0
»0
¼1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½	variables
ölayers
¾trainable_variables
÷layer_metrics
¿regularization_losses
ømetrics
 ùlayer_regularization_losses
únon_trainable_variables
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_8_conv1/kernel
):'@2resblock_part2_8_conv1/bias
0
Ã0
Ä1"
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Å	variables
ûlayers
Ætrainable_variables
ülayer_metrics
Çregularization_losses
ýmetrics
 þlayer_regularization_losses
ÿnon_trainable_variables
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
É	variables
layers
Êtrainable_variables
layer_metrics
Ëregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_8_conv2/kernel
):'@2resblock_part2_8_conv2/bias
0
Í0
Î1"
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ï	variables
layers
Ðtrainable_variables
layer_metrics
Ñregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
®__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
-:+@2upsampler_1/kernel
:2upsampler_1/bias
0
Õ0
Ö1"
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×	variables
layers
Øtrainable_variables
layer_metrics
Ùregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_1_conv1/kernel
):'@2resblock_part3_1_conv1/bias
0
Ü0
Ý1"
trackable_list_wrapper
0
Ü0
Ý1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þ	variables
layers
ßtrainable_variables
layer_metrics
àregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
â	variables
layers
ãtrainable_variables
layer_metrics
äregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_1_conv2/kernel
):'@2resblock_part3_1_conv2/bias
0
æ0
ç1"
trackable_list_wrapper
0
æ0
ç1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
è	variables
layers
étrainable_variables
layer_metrics
êregularization_losses
metrics
 layer_regularization_losses
non_trainable_variables
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_2_conv1/kernel
):'@2resblock_part3_2_conv1/bias
0
î0
ï1"
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ð	variables
layers
ñtrainable_variables
layer_metrics
òregularization_losses
 metrics
 ¡layer_regularization_losses
¢non_trainable_variables
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ô	variables
£layers
õtrainable_variables
¤layer_metrics
öregularization_losses
¥metrics
 ¦layer_regularization_losses
§non_trainable_variables
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_2_conv2/kernel
):'@2resblock_part3_2_conv2/bias
0
ø0
ù1"
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ú	variables
¨layers
ûtrainable_variables
©layer_metrics
üregularization_losses
ªmetrics
 «layer_regularization_losses
¬non_trainable_variables
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_3_conv1/kernel
):'@2resblock_part3_3_conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
­layers
trainable_variables
®layer_metrics
regularization_losses
¯metrics
 °layer_regularization_losses
±non_trainable_variables
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
²layers
trainable_variables
³layer_metrics
regularization_losses
´metrics
 µlayer_regularization_losses
¶non_trainable_variables
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_3_conv2/kernel
):'@2resblock_part3_3_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
·layers
trainable_variables
¸layer_metrics
regularization_losses
¹metrics
 ºlayer_regularization_losses
»non_trainable_variables
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_4_conv1/kernel
):'@2resblock_part3_4_conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
¼layers
trainable_variables
½layer_metrics
regularization_losses
¾metrics
 ¿layer_regularization_losses
Ànon_trainable_variables
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Álayers
trainable_variables
Âlayer_metrics
regularization_losses
Ãmetrics
 Älayer_regularization_losses
Ånon_trainable_variables
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_4_conv2/kernel
):'@2resblock_part3_4_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Ælayers
trainable_variables
Çlayer_metrics
 regularization_losses
Èmetrics
 Élayer_regularization_losses
Ênon_trainable_variables
È__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
+:)@@2extra_conv/kernel
:@2extra_conv/bias
0
¤0
¥1"
trackable_list_wrapper
0
¤0
¥1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦	variables
Ëlayers
§trainable_variables
Ìlayer_metrics
¨regularization_losses
Ímetrics
 Îlayer_regularization_losses
Ïnon_trainable_variables
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
-:+@2upsampler_2/kernel
:2upsampler_2/bias
0
«0
¬1"
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­	variables
Ðlayers
®trainable_variables
Ñlayer_metrics
¯regularization_losses
Òmetrics
 Ólayer_regularization_losses
Ônon_trainable_variables
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
,:*@2output_conv/kernel
:2output_conv/bias
0
²0
³1"
trackable_list_wrapper
0
²0
³1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
´	variables
Õlayers
µtrainable_variables
Ölayer_metrics
¶regularization_losses
×metrics
 Ølayer_regularization_losses
Ùnon_trainable_variables
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89
[90
\91
]92"
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
ë2è
__inference__wrapped_model_2950Ä
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ
æ2ã
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4524
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5951
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4256
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_6260À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_ssi_res_unet_layer_call_fn_5447
+__inference_ssi_res_unet_layer_call_fn_4986
+__inference_ssi_res_unet_layer_call_fn_6646
+__inference_ssi_res_unet_layer_call_fn_6453À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_input_conv_layer_call_and_return_conditional_losses_6656¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_input_conv_layer_call_fn_6665¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2957à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_zero_padding2d_layer_call_fn_2963à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ñ2î
G__inference_downsampler_1_layer_call_and_return_conditional_losses_6675¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_downsampler_1_layer_call_fn_6684¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_6694¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_1_conv1_layer_call_fn_6703¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_6708¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_1_relu1_layer_call_fn_6713¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_6723¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_1_conv2_layer_call_fn_6732¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_6742¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_2_conv1_layer_call_fn_6751¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_6756¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_2_relu1_layer_call_fn_6761¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_6771¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_2_conv2_layer_call_fn_6780¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_6790¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_3_conv1_layer_call_fn_6799¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_6804¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_3_relu1_layer_call_fn_6809¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_6819¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_3_conv2_layer_call_fn_6828¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_6838¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_4_conv1_layer_call_fn_6847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_6852¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_4_relu1_layer_call_fn_6857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_6867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_4_conv2_layer_call_fn_6876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
²2¯
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2970à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_zero_padding2d_1_layer_call_fn_2976à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ñ2î
G__inference_downsampler_2_layer_call_and_return_conditional_losses_6886¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_downsampler_2_layer_call_fn_6895¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6905¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_1_conv1_layer_call_fn_6914¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_1_relu1_layer_call_fn_6924¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6934¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_1_conv2_layer_call_fn_6943¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6953¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_2_conv1_layer_call_fn_6962¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6967¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_2_relu1_layer_call_fn_6972¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6982¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_2_conv2_layer_call_fn_6991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_7001¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_3_conv1_layer_call_fn_7010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_7015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_3_relu1_layer_call_fn_7020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_7030¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_3_conv2_layer_call_fn_7039¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_7049¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_4_conv1_layer_call_fn_7058¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_7063¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_4_relu1_layer_call_fn_7068¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_7078¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_4_conv2_layer_call_fn_7087¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_7097¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_5_conv1_layer_call_fn_7106¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_7111¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_5_relu1_layer_call_fn_7116¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_7126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_5_conv2_layer_call_fn_7135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_7145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_6_conv1_layer_call_fn_7154¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_7159¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_6_relu1_layer_call_fn_7164¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_7174¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_6_conv2_layer_call_fn_7183¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_7193¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_7_conv1_layer_call_fn_7202¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_7207¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_7_relu1_layer_call_fn_7212¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_7222¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_7_conv2_layer_call_fn_7231¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_7241¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_8_conv1_layer_call_fn_7250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_7255¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_8_relu1_layer_call_fn_7260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_7270¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_8_conv2_layer_call_fn_7279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_upsampler_1_layer_call_and_return_conditional_losses_7289¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_upsampler_1_layer_call_fn_7298¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_7308¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_1_conv1_layer_call_fn_7317¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_7322¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_1_relu1_layer_call_fn_7327¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_7337¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_1_conv2_layer_call_fn_7346¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_7356¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_2_conv1_layer_call_fn_7365¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_7370¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_2_relu1_layer_call_fn_7375¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_7385¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_2_conv2_layer_call_fn_7394¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_7404¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_3_conv1_layer_call_fn_7413¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_7418¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_3_relu1_layer_call_fn_7423¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_7433¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_3_conv2_layer_call_fn_7442¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_7452¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_4_conv1_layer_call_fn_7461¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_7466¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_4_relu1_layer_call_fn_7471¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_7481¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_4_conv2_layer_call_fn_7490¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_extra_conv_layer_call_and_return_conditional_losses_7500¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_extra_conv_layer_call_fn_7509¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_upsampler_2_layer_call_and_return_conditional_losses_7519¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_upsampler_2_layer_call_fn_7528¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_output_conv_layer_call_and_return_conditional_losses_7538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_output_conv_layer_call_fn_7547¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
"__inference_signature_wrapper_5642input_layer"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15à
__inference__wrapped_model_2950¼´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³>¢;
4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ
ª "Cª@
>
output_conv/,
output_convÿÿÿÿÿÿÿÿÿ»
G__inference_downsampler_1_layer_call_and_return_conditional_losses_6675pmn9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_downsampler_1_layer_call_fn_6684cmn9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@»
G__inference_downsampler_2_layer_call_and_return_conditional_losses_6886p¿À9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
,__inference_downsampler_2_layer_call_fn_6895c¿À9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@@@º
D__inference_extra_conv_layer_call_and_return_conditional_losses_7500r¤¥9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_extra_conv_layer_call_fn_7509e¤¥9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@¸
D__inference_input_conv_layer_call_and_return_conditional_losses_6656pcd9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_input_conv_layer_call_fn_6665ccd9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ@»
E__inference_output_conv_layer_call_and_return_conditional_losses_7538r²³9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_output_conv_layer_call_fn_7547e²³9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿÄ
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_6694pst9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_conv1_layer_call_fn_6703cst9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Ä
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_6723p}~9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_conv2_layer_call_fn_6732c}~9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_6708l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_relu1_layer_call_fn_6713_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_6742r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_conv1_layer_call_fn_6751e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_6771r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_conv2_layer_call_fn_6780e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_6756l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_relu1_layer_call_fn_6761_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_6790r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_conv1_layer_call_fn_6799e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_6819r¡¢9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_conv2_layer_call_fn_6828e¡¢9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_6804l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_relu1_layer_call_fn_6809_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_6838r©ª9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_conv1_layer_call_fn_6847e©ª9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_6867r³´9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_conv2_layer_call_fn_6876e³´9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_6852l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_relu1_layer_call_fn_6857_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Â
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6905nÅÆ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_conv1_layer_call_fn_6914aÅÆ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6934nÏÐ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_conv2_layer_call_fn_6943aÏÐ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6919h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_relu1_layer_call_fn_6924[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6953n×Ø7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_conv1_layer_call_fn_6962a×Ø7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6982náâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_conv2_layer_call_fn_6991aáâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6967h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_relu1_layer_call_fn_6972[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_7001néê7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_conv1_layer_call_fn_7010aéê7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_7030nóô7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_conv2_layer_call_fn_7039aóô7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_7015h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_relu1_layer_call_fn_7020[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_7049nûü7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_conv1_layer_call_fn_7058aûü7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_7078n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_conv2_layer_call_fn_7087a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_7063h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_relu1_layer_call_fn_7068[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_7097n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_conv1_layer_call_fn_7106a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_7126n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_conv2_layer_call_fn_7135a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_7111h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_relu1_layer_call_fn_7116[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_7145n 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_conv1_layer_call_fn_7154a 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_7174n©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_conv2_layer_call_fn_7183a©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_7159h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_relu1_layer_call_fn_7164[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_7193n±²7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_conv1_layer_call_fn_7202a±²7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_7222n»¼7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_conv2_layer_call_fn_7231a»¼7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_7207h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_relu1_layer_call_fn_7212[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_7241nÃÄ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_conv1_layer_call_fn_7250aÃÄ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_7270nÍÎ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_conv2_layer_call_fn_7279aÍÎ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_7255h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_relu1_layer_call_fn_7260[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Æ
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_7308rÜÝ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_conv1_layer_call_fn_7317eÜÝ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_7337ræç9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_conv2_layer_call_fn_7346eæç9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_7322l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_relu1_layer_call_fn_7327_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_7356rîï9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_conv1_layer_call_fn_7365eîï9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_7385røù9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_conv2_layer_call_fn_7394eøù9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_7370l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_relu1_layer_call_fn_7375_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_7404r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_conv1_layer_call_fn_7413e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_7433r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_conv2_layer_call_fn_7442e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_7418l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_relu1_layer_call_fn_7423_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_7452r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_conv1_layer_call_fn_7461e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_7481r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_conv2_layer_call_fn_7490e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_7466l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_relu1_layer_call_fn_7471_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@ò
"__inference_signature_wrapper_5642Ë´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³M¢J
¢ 
Cª@
>
input_layer/,
input_layerÿÿÿÿÿÿÿÿÿ"Cª@
>
output_conv/,
output_convÿÿÿÿÿÿÿÿÿû
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4256°´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 û
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4524°´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ö
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5951«´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ö
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_6260«´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ó
+__inference_ssi_res_unet_layer_call_fn_4986£´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÓ
+__inference_ssi_res_unet_layer_call_fn_5447£´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿÎ
+__inference_ssi_res_unet_layer_call_fn_6453´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÎ
+__inference_ssi_res_unet_layer_call_fn_6646´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¸
E__inference_upsampler_1_layer_call_and_return_conditional_losses_7289oÕÖ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
*__inference_upsampler_1_layer_call_fn_7298bÕÖ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "!ÿÿÿÿÿÿÿÿÿ@@¼
E__inference_upsampler_2_layer_call_and_return_conditional_losses_7519s«¬9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_upsampler_2_layer_call_fn_7528f«¬9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "# ÿÿÿÿÿÿÿÿÿí
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2970R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_zero_padding2d_1_layer_call_fn_2976R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2957R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_zero_padding2d_layer_call_fn_2963R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ