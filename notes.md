# Callan-Symanzik Equation Notes

## Key Relationships

For a correlation function C(x1, x2, g) = |x2 - x1|^(-1 + γ(g)), the Callan-Symanzik equation must be satisfied:

β(g)∂_g C + γ(g)D C - d C = 0

Where:
- γ(g) = g²/(16π²) is the anomalous dimension
- ∂_g γ(g) = g/(8π²) is its derivative
- β(g) needs to satisfy β(g)∂_g γ(g) = γ(g)²

## Solving for β(g)

Given:
1. γ(g) = g²/(16π²)
2. ∂_g γ(g) = g/(8π²)
3. Need: β(g)∂_g γ(g) = γ(g)²

Therefore:
- β(g) * (g/(8π²)) = (g²/(16π²))²
- β(g) * (g/(8π²)) = g⁴/(256π⁴)
- β(g) = (g⁴/(256π⁴)) * (8π²/g) = g³/(32π²)

## Verification

Let's verify β(g)∂_g γ(g) = γ(g)²:

1. β(g)∂_g γ(g):
   - = (g³/(32π²)) * (g/(8π²))
   - = g⁴/(256π⁴)

2. γ(g)²:
   - = (g²/(16π²))²
   - = g⁴/(256π⁴)

They match! Therefore β(g) = g³/(32π²) is the correct beta function.

## TODO: Verify with Symbolic Solver 