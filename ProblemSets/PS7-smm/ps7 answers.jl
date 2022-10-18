
#PROBLEM SET 7
#SARA ASGARI

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
using SMM
function wrapper()
#Q1:

df = CSV.read("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022-1\\ProblemSets\\PS1-julia-intro\\nlsw88.csv",DataFrame,header=true,missingstring="")
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols_gmm(alpha, X,y)
    g = y .- X*alpha
    J = g'*I*g
    return J
end

alphhat_optim = optimize(a -> ols_gmm(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100))
println(alphhat_optim.minimizer)
# results: [0.661350950970495, -0.004625908030053608, 0.22595043228407072, -0.012184197714197221]


#Q1.B:Checking work

bols = inv(X'*X)*X'*y
println(bols)
#very close results : [0.661350951187714, -0.004625908035375659, 0.22595043227530143, -0.012184197720655598]


#Q2:

df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, y)
        
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)

    loglike = -sum( bigY.*log.(P) )
        
    return loglike
end



alpha_rand = rand(6*size(X,2))
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
alpha_start = alpha_true.*rand(size(alpha_true))
println(size(alpha_true))

alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)

#Q2.B:

function logit_gmm(alpha, X, y)
        
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

    num   = zeros(N,J)                      
    dem   = zeros(N)                        
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end
    P = num./repeat(dem,1,J)
        
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)

#Q2.C:

alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)
# I got different coeficients which indicates that the function isn't globally concave


#Q3:
using Distributions

N=size(X,1)

J = length(unique(y))
Xrand = rand(N,J)
beta = rand(J)
epsilon= randn(N,1)
for i in 1:N
    YY[i] = argmax(X[i]*beta[:] + epsilon[i,:])
end
#I tried but could not go any furthur, I couldnot set the right value for "beta"


#Q4
#No NEED TO SOLVE.  


#Q5:
function mlogit_smm(θ, X, y, D)
    K = size(X,2)
    J = length(unique(y))
    N = size(y,1)


    bigY = zeros(N,J)
    for j=1:J
            bigY[:,j] = y.==j
    end
   

    β = bigθ[1:end-1]
    σ = bigθ[end]
    if length(β)==1
        β = β[1]
    end 

    gmodel = zeros(N+1,D)
    gdata  = vcat(bigY,var(bigY))
  
    Random.seed!(1234)  
    for d=1:D
        ε = σ*randn(N,J)
        ỹ = P .+ ε
        gmodel[1:end-1,1:J,d] = ỹ
        gmodel[  end  ,1:J,d] = var(ỹ)
    end

    err = vec(gdata .- mean(gmodel; dims=2))

    J = err'*I*err
    return J
end
  


alpha_hat_smm = optimize(a -> mlogit_smm(θ, X, y, D), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
println(alpha_hat_smm)
#couldnot get into the results.


#Q6:
return nothing
end
wrapper()
