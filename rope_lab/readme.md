# RoPE Geometry  
Rotation matrix that rotates vectors in 2D space.  

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




