using Convex, SCS, LinearAlgebra, Plots, ECOS#, GLPK, Mosek

N = 101;
S = [0, 180];
tt = 0; theta = [0.05, 0.5];
 obj_val = 0;

function FUN(xi, theta)
  deno = (theta[1] + xi * theta[2])^2;
  return [-xi/deno, -xi^2/deno] ;
end

# Initilize
u = LinRange(S[1], S[2], N);
n = length(theta);

# Start CVX
## Define variables
w = Variable(N);
del = Variable(1);

## Define constraints
c1 = w>= 0;
C = vcat(zeros(n)', diagm(ones(n)));
g1 = zeros(n);
G2 = zeros(n, n);
obj_val = 0 ;
for i in 1:N
  f = FUN(u[i], theta);
  g1 = g1 + w[i] * f;
  G2 = G2 + w[i] * f *  f';
end
B = vcat([1; sqrt(tt) * g1]', hcat(sqrt(tt) * g1, G2));

for k in 1:n
  obj_val = obj_val + matrixfrac(C[:, k], B)
end

c2 =  obj_val <= del;
c3 = w' * ones(N) == 1;
constraints = [c1, c2, c3];
# constraints  = [w >= 0, obj_val <= del, w' * ones(N) == 1]

## Define the problem
objective = minimize(del, constraints);
solve!(objective, () -> SCS.Optimizer(max_iters = 1E6, verbose = false, eps = 1e-8))

## Check the status of the problem
objective.status # :Optimal, :Infeasible, :Unbounded etc.

# Get the optimal value
objective.optval
w.value

plot(u, w.value, seriestype = :scatter, title = "weight plot")
