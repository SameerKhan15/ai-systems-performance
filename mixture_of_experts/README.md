# MoE = Mixture of Experts
# Motivation  
```
          +----------------------+
          |                      |
x ───────▶|     Huge Model       |──────▶ ŷ
          |                      |
          +----------------------+
```
**Question: Do we need to activate ALL the parameters for every input prompt?**  

**Analogy:**  
A room with a chemist, mathematician, physicist, cardiologist. 
If a given question is about mathmatics, who are we going to direct the question to? Or ask everyone?  
Without MoE, LLM involve asking "everyone". All parameters in the model take part in the inference, although a significant subset may NOT have meaningful influence on the output token generation.  

Motivation: Not all weights are useful in the forward pass. 

## FLOPs  
`Floating point operations`  
Measure the number of operations (add, multiply etc) involved in a forward pass. It quantifies how compute heavy a given task is.  

## MoEs General Architecture   
```text
                 G
Input ─────────►[G]─────────────────────────────────────────────
                    │            │             │
                    │            │             │
                    ▼            ▼             ▼
                 (broadcast to experts)

          x ───────────────► [E₁] ──┐
           ├───────────────► [E₂] ──┤
           │                        │
           ├───────────────►   ⋮    ├──► ŷ
           │                        │
           └───────────────► [Eₙ] ──┘
- **G** = Gating network
- **x** = Input
- **E₁ ... Eₙ** = Expert networks
- The gating network computes routing decisions that determine which experts receive the input.
- The outputs of the selected experts are combined to produce the final prediction **ŷ**.
```

$\hat{y} = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$  
where:  
G = weight quantity output by G, assigning weightage to the output of the given expert  
`"how important the output of each expert is"`  

## Types of MoEs  
`Sparse MoE: lower amount of FLOPs`  
`Dense MoE: higher amount of FLOPs`  

**Dense MoE:** Output is weighted average of ALL expert outputs  
```text
                   G
                   │
                   ├──────────────────────────────┐
                   ^       │        |             |         
                   |     0.05       |             |      
                   │       |        |             |
                   |       |       0.8            |
                   x       |        |            0.1
                   │       |        |              │
                   ├────────────────|──────────────|────────► [E1]
                   │                |              |
                   ├────────────────|──────────────|────────► [E2]
                   │                               |
                   │                               |
                   │                               |
                   ⋮                               |
                   │                               |
                   └───────────────────────────────────────► [En]

[E1] ───────────────────────────────┐
[E2] ───────────────────────────────┼────────► ŷ
 ⋮                                  │
[En] ───────────────────────────────┘
```
**Sparse MoE:** Constrain the # of experts that are activated  
$\hat{y} = \sum_{i \in I_k} G(x)_i \cdot \varepsilon_i(x)$  

**experts chosen via top-k selection**  

A key objective with Sparse MoE is to reduce the amount of compute (measured via FLOPs)  
