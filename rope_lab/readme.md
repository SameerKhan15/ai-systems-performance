# RoPE Geometry  
Rotation matrix that rotates vectors in 2D space.  

## Key Formulas  
![](6.png "This is a sample image.")  

## Single Q/K Vector versus entire Q/K Matrix  
### Single Token  
Suppose we have one token with d=4:  

After projection:  
![](1.png "This is a sample image.")  

This is a single query vector.  

RoPE operates on this vector by rotating:
* (q0, q1)  
* (q2, q3)  
independently.  

### Entire Sequence  
Now suppose sentence:  
`The cat sat down`  
4 tokens, each token being of 4D (4 dimensions)  

![](2.png "This is a sample image.")  

where each row is a query vector.  

For example:  
![](3.png "This is a sample image.")  

Shape:  
4x4 (sequence_length * head_dimension)  

### Where Does RoPE Apply  
RoPE does NOT rotate the entire matrix as one object.  

Instead:  
For row 0:  
* rotate using position 0.  
For row 1:  
* rotate using position 1.  
For row 2:  
* rotate using position 2.  
For row 3:  
* rotate using position 3.  

#### Visual Picture  
Before:  
![](4.png "This is a sample image.")  

RoPE applies:  
R0 to first row, R1 to 2nd row, R2 to third row, R3 to fourth row  
where R is the rotation matrix  

#### Real LLM Example  
Suppose:  
* sequence_length = 8192  
* head_dim = 128  

Then:  
Q = 8192 x 128  for one attention head  

RoPE would process 8192 rows  
Within each row, 64 independent 2D rotation, because 128/2 = 64 dimension pairs   

#### Mental Model  
Think of RoPE as:  
for every token position:  
    for every dimension pair:  
        rotate that 2D pair  

#### Then Attention Happens  
Only AFTER RoPE modifies Q and K do we compute: QK^T  
The flow is:  
![](5.png "This is a sample image.")  

RoPE is fundamentally a per-token vector transformation.  

Attention is fundamentally a matrix operation across all tokens.  

