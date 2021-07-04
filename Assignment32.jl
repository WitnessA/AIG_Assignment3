### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 7b486830-dcbb-11eb-0818-d9644a59e754
using POMDPs

# ╔═╡ 10f0adc0-0b3c-437a-be2f-f594300005fd
using POMDPModelTools

# ╔═╡ 65fa6af2-0a88-4a4d-bb75-8f1d11d92327
using POMDPPolicies

# ╔═╡ 20ec439d-1f47-4aa2-ae74-cefc089f8de3
using POMDPSimulators

# ╔═╡ d4002513-694d-42a2-ba02-4af72d036006
using DiscreteValueIteration

# ╔═╡ 3c462140-c40b-4461-b8c5-d0a7da35baba
struct GridState
	x::Int64
	y::Int64
	done::Bool
end

# ╔═╡ e1e64951-737b-4c40-adac-63b45051850b
GS(x::Int64, y::Int64) = GridState(x,y,false)

# ╔═╡ 89132698-3c7b-485b-8508-46f55e59a05b
posequal(s1::GridState, s2::GridState) = s1.x == s2.x && s1.y == s2.y

# ╔═╡ e009f038-8cb1-4580-87b9-c8293a859d19
mutable struct GridStruct <: MDP{GridState, Symbol}
    xaxis::Int64
    yaxis::Int64
    reward_states::Vector{GridState}
    reward_values::Vector{Float64}
    transprob::Float64
    discount_factor::Float64
end

# ╔═╡ 3c435018-7445-4b45-a844-ad0b004520bb
function GW(;sx::Int64=5,
                    sy::Int64=5,
                    rs::Vector{GridState}=[GS(1,3), GS(1,4), GS(3,1),GS(3,4),GS(3,5),GS(4,2),GS(4,5),GS(5,3),GS(5,5)],
                    rv::Vector{Float64}=rv = [-10.,3,-3,3,-3,7,3,-3,10],
                    tp::Float64=0.7,
                    discount_factor::Float64=0.9)
    return GridStruct(sx, sy, rs, rv, tp, discount_factor)
end

# ╔═╡ 41fa4e2d-a712-4322-8ce4-14caf08c8b60
function POMDPs.states(mdp::GridStruct)
    s = GridState[]
    for d = 0:1, y = 1:mdp.yaxis, x = 1:mdp.xaxis
        push!(s, GridState(x,y,d))
    end
    return s
end;

# ╔═╡ 2ff54443-3fa6-417b-aafe-18907b1ab019
POMDPs.actions(mdp::GridStruct) = [:up, :down, :left, :right];

# ╔═╡ 0021375d-9458-46ff-9c72-50159bee24c8
function inbounds(mdp::GridStruct,x::Int64,y::Int64)
    if 1 <= x <= mdp.xaxis && 1 <= y <= mdp.yaxis
        return true
    else
        return false
    end
end

# ╔═╡ 8454c1a1-c052-492d-ae8c-ce7d6b64ad81
inbounds(mdp::GridStruct, state::GridState) = inbounds(mdp, state.x, state.y);

# ╔═╡ 823de16c-f0f7-41b8-89b1-fa642bfa6ca0
function POMDPs.transition(mdp::GridStruct, state::GridState, action::Symbol)
    a = action
    x = state.x
    y = state.y
    
    if state.done
        return SparseCat([GridState(x, y, true)], [1.0])
    elseif state in mdp.reward_states
        return SparseCat([GridState(x, y, true)], [1.0])
    end

    neighbors = [
        GridState(x+1, y, false),
        GridState(x-1, y, false),
        GridState(x, y-1, false),
        GridState(x, y+1, false),
        ]
    
    targets = Dict(:right=>1, :left=>2, :down=>3, :up=>4)
    target = targets[a]
    
    probability = fill(0.0, 4)

    if !inbounds(mdp, neighbors[target])
        return SparseCat([GS(x, y)], [1.0])
    else
        probability[target] = mdp.transprob

        oob_count = sum(!inbounds(mdp, n) for n in neighbors) 

        new_probability = (1.0 - mdp.transprob)/(3-oob_count)

        for i = 1:4
            if inbounds(mdp, neighbors[i]) && i != target
                probability[i] = new_probability
            end
        end
    end

    return SparseCat(neighbors, probability)
end;

# ╔═╡ b9d86e16-6305-485d-9ae5-034c76a9db7e
function POMDPs.reward(mdp::GridStruct, state::GridState, action::Symbol, statep::GridState) #deleted action
    if state.done
        return 0.0
    end
    r = 0.0
    n = length(mdp.reward_states)
    for i = 1:n
        if posequal(state, mdp.reward_states[i])
            r += mdp.reward_values[i]
        end
    end
    return r
end;

# ╔═╡ e709f81b-bdc8-4c04-b485-c95b09e0fc11
POMDPs.discount(mdp::GridStruct) = mdp.discount_factor;

# ╔═╡ d9c7cd7b-fb07-42c2-a741-9c111063b0ee
function POMDPs.stateindex(mdp::GridStruct, state::GridState)
    sd = Int(state.done + 1)
    ci = CartesianIndices((mdp.xaxis, mdp.yaxis, 2))
    return LinearIndices(ci)[state.x, state.y, sd]
end

# ╔═╡ 36cc0c13-bd0d-4433-8c24-ddc59495a93d
function POMDPs.actionindex(mdp::GridStruct, act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    end
    error("Invalid GridStruct action: $act")
end;

# ╔═╡ a2e64b64-dbe6-4d62-bc4f-aab5a16cc408
POMDPs.isterminal(mdp::GridStruct, s::GridState) = s.done

# ╔═╡ b5b00975-9cc5-436b-8c3a-fec3c5663b31
POMDPs.initialstate(pomdp::GridStruct) = Deterministic(GS(1,1))

# ╔═╡ 702af6ef-97e8-4e79-88e2-c7609d03914e
mdp = GW()

# ╔═╡ aef9a796-a10f-4bab-a140-f4b73711fe87
solver = ValueIterationSolver(max_iterations=100, belres=1e-3; verbose=true)

# ╔═╡ faeeb566-110a-49c2-b23d-d60fad695ad8
policy = solve(solver, mdp);

# ╔═╡ dd5ebdbe-eb6d-4260-b34f-4706e838f711
for (s,a,r) in stepthrough(mdp, policy, "s,a,r", max_steps=20)
    @show s
    @show a
    @show r
    println()
end

# ╔═╡ 401e1c66-8357-4cc0-ae58-98e51bcbc22d


# ╔═╡ Cell order:
# ╠═7b486830-dcbb-11eb-0818-d9644a59e754
# ╠═10f0adc0-0b3c-437a-be2f-f594300005fd
# ╠═65fa6af2-0a88-4a4d-bb75-8f1d11d92327
# ╠═20ec439d-1f47-4aa2-ae74-cefc089f8de3
# ╠═d4002513-694d-42a2-ba02-4af72d036006
# ╠═3c462140-c40b-4461-b8c5-d0a7da35baba
# ╠═e1e64951-737b-4c40-adac-63b45051850b
# ╠═89132698-3c7b-485b-8508-46f55e59a05b
# ╠═e009f038-8cb1-4580-87b9-c8293a859d19
# ╠═3c435018-7445-4b45-a844-ad0b004520bb
# ╠═41fa4e2d-a712-4322-8ce4-14caf08c8b60
# ╠═2ff54443-3fa6-417b-aafe-18907b1ab019
# ╠═0021375d-9458-46ff-9c72-50159bee24c8
# ╠═8454c1a1-c052-492d-ae8c-ce7d6b64ad81
# ╠═823de16c-f0f7-41b8-89b1-fa642bfa6ca0
# ╠═b9d86e16-6305-485d-9ae5-034c76a9db7e
# ╠═e709f81b-bdc8-4c04-b485-c95b09e0fc11
# ╠═d9c7cd7b-fb07-42c2-a741-9c111063b0ee
# ╠═36cc0c13-bd0d-4433-8c24-ddc59495a93d
# ╠═a2e64b64-dbe6-4d62-bc4f-aab5a16cc408
# ╠═b5b00975-9cc5-436b-8c3a-fec3c5663b31
# ╠═702af6ef-97e8-4e79-88e2-c7609d03914e
# ╠═aef9a796-a10f-4bab-a140-f4b73711fe87
# ╠═faeeb566-110a-49c2-b23d-d60fad695ad8
# ╠═dd5ebdbe-eb6d-4260-b34f-4706e838f711
# ╠═401e1c66-8357-4cc0-ae58-98e51bcbc22d
