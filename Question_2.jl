### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ ef459110-dd09-11eb-327d-7f6fa4737a51
using Games

# ╔═╡ 8561cceb-f46e-4e3a-b257-78d460bd827a
matching_pennies_bimatrix = Array{Float64}(undef, 2, 2, 2)

# ╔═╡ 3dbbbba3-fe72-4fa4-bc77-884d8b588de9
matching_pennies_bimatrix[1, 1, :] = [1, -1]

# ╔═╡ f90c8299-dff7-4e61-ac76-284d4f5c5e6d
matching_pennies_bimatrix[1, 2, :] = [-1, 1]

# ╔═╡ 66f85d91-58d9-4a9e-9bda-e39ea12253dd
matching_pennies_bimatrix[2, 1, :] = [-1, 1]

# ╔═╡ d7a839b3-f1d9-40bc-8610-e3770592aab4
matching_pennies_bimatrix[2, 2, :] = [1, -1]

# ╔═╡ 9b5567c5-0c42-4d39-a865-dc54f8611056
g_MP = NormalFormGame(matching_pennies_bimatrix)

# ╔═╡ cc7ea04e-ff21-4b14-bf4e-2a60a0576c0a
g_MP.players[1]

# ╔═╡ d909bde2-1f79-4a29-996c-e839676825d5
g_MP.players[2]

# ╔═╡ ebd8efa6-fdce-495d-a591-8fc45cd14e0a
g_MP.players[1].payoff_array

# ╔═╡ cb980d54-3322-4634-90a0-3363469926b9
g_MP.players[2].payoff_array

# ╔═╡ de96089c-42ee-4f0c-aee2-a58e011bbc46
g_MP[1, 1]

# ╔═╡ 5b5e63a2-e509-473c-b103-dcf5a391ba37
coordination_game_matrix = [4 0;
                            3 2]

# ╔═╡ 76fc34d6-e39f-4fa2-80ce-bc151af0a87c
g_Coo = NormalFormGame(coordination_game_matrix)

# ╔═╡ d51dcdba-e21f-443d-bc95-06fbfb1058af
g_Coo.players[1].payoff_array

# ╔═╡ 9884efb6-08a6-4636-a6a3-8d3b45714872
g_Coo.players[2].payoff_array

# ╔═╡ 4a0b9006-5532-46c0-8742-aab1041c92f5
RPS_matrix = [0 -1 1;
              1 0 -1;
              -1 1 0]

# ╔═╡ 21110690-7971-48eb-b464-0ce440e75d96
g_RPS = NormalFormGame(RPS_matrix)

# ╔═╡ 71754f38-9658-42ef-a7af-a7a263610e1c
g_PD = NormalFormGame((2, 2))

# ╔═╡ 7ffe3816-49d8-4188-94e0-515d91a2aafe
g_PD[1, 1] = [1, 1]

# ╔═╡ 9f5fe5f7-b941-4f04-bae9-2bb8a3e8f9d1
g_PD[1, 2] = [-2, 3]

# ╔═╡ 62d799f7-34c2-429c-8687-3dc3fd8550aa
g_PD[2, 1] = [3, -2]

# ╔═╡ f4dbf36a-5ea9-432a-a3b4-9969e4b6067f
g_PD[2, 2] = [0, 0];

# ╔═╡ 1849456a-d485-48ca-a444-fe1e6f3d8b6e
g_PD.players[1].payoff_array

# ╔═╡ 71dfa720-0d4b-47cb-9c3c-a6b0a254b272
player1 = Player([3 1; 0 2])

# ╔═╡ 8c7e05ba-4032-4898-a40e-ad79e54b8505
player2 = Player([2 0; 1 3]);

# ╔═╡ 77d75ea0-5e72-4222-898d-0fe87431e833
player1.payoff_array

# ╔═╡ 5889ca2e-2a5d-42b6-8156-6bcbe781977f
player2.payoff_array

# ╔═╡ 1e7f7339-91ed-42be-ac6d-cc1c9a9e685b
g_BoS = NormalFormGame((player1, player2))

# ╔═╡ 10feaece-5095-4545-91e0-fea2e14bbea7
function cournot(a::Real, c::Real, ::Val{N}, q_grid::AbstractVector{T}) where {N,T<:Real}
    nums_actions = ntuple(x->length(q_grid), Val(N))
    S = promote_type(typeof(a), typeof(c), T)
    payoff_array= Array{S}(undef, nums_actions)
    for I in CartesianIndices(nums_actions)
        Q = zero(S)
        for i in 1:N
            Q += q_grid[I[i]]
        end
        payoff_array[I] = (a - c - Q) * q_grid[I[1]]
    end
    players = ntuple(x->Player(payoff_array), Val(N))
    return NormalFormGame(players)
end

# ╔═╡ 78c77f20-6826-4594-b055-0e1e27935fe2
#a, c = 80, 20

# ╔═╡ f040c2a2-cdc1-4923-94eb-6cbd9e4f840f


# ╔═╡ e9250bd1-6cc9-4741-94b7-f4ce94ee0288
#N = 3

# ╔═╡ 1041a0af-3e98-443d-9754-ad62e1b9c57a
#q_grid = [10, 15]

# ╔═╡ d24dff90-9376-4215-8c82-cdaaa16fb7d9
#g_Cou = cournot(a, c, Val(N), q_grid)

# ╔═╡ fc063148-fd9b-47ca-b02d-3b0769d98361
function print_pure_nash_brute(g::NormalFormGame)
    NEs = pure_nash(g)
    num_NEs = length(NEs)
    if num_NEs == 0
        msg = "no pure Nash equilibrium"
    elseif num_NEs == 1
        msg = "1 pure Nash equilibrium:\n$(NEs[1])"
    else
        msg = "$num_NEs pure Nash equilibria:\n"
        for (i, NE) in enumerate(NEs)
            i < num_NEs ? msg *= "$NE, " : msg *= "$NE"
        end
    end
    println(join(["The game has ", msg]))
end

# ╔═╡ 44bae769-6e69-48f9-a26e-a4063d940934
print_pure_nash_brute(g_MP)

# ╔═╡ d12f6c05-39c0-4994-9d98-b617bf516e3a
print_pure_nash_brute(g_Coo)

# ╔═╡ b17583ce-695f-43dc-87cc-e5cab672a455
print_pure_nash_brute(g_RPS)

# ╔═╡ 42a2c8d8-8f31-4951-afaa-1a0a7bd5158a
print_pure_nash_brute(g_BoS)

# ╔═╡ 014035d1-814e-4894-addd-3c9706952dde
function sequential_best_response(g::NormalFormGame{N};
                                  init_actions::Vector{Int}=ones(Int, N),
                                  tie_breaking=:smallest,
                                  verbose=true) where N
    a = copy(init_actions)
    if verbose
        println("init_actions: $a")
    end
    
    new_a = Array{Int}(undef, N)
    max_iter = prod(g.nums_actions)
    
    for t in 1:max_iter
        copyto!(new_a, a)
        for (i, player) in enumerate(g.players)
            if N == 2
                a_except_i = new_a[3-i]
            else
                a_except_i = (new_a[i+1:N]..., new_a[1:i-1]...)
            end
            new_a[i] = best_response(player, a_except_i,
                                     tie_breaking=tie_breaking)
            if verbose
                println("player $i: $new_a")
            end
        end
        if new_a == a
            return a
        else
            copyto!(a, new_a)
        end
    end
    
    println("No pure Nash equilibrium found")
    return a
end

# ╔═╡ 1fe3eeba-8026-4360-a661-1fbe25922378
a, c = 80, 20

# ╔═╡ d2a6d877-5f6d-4a90-981f-480a30530085
#N = 3

# ╔═╡ e7e37985-32d9-4859-81f5-bd570be7144b
#q_grid_size = 13

# ╔═╡ c71c26a7-5591-41e7-b8db-c5bed45fbe4b
#q_grid = range(0, step=div(a-c, q_grid_size-1), length=q_grid_size)

# ╔═╡ 76c065bd-c0bf-4e71-a053-fed25a9853cc
#g_Cou = cournot(a, c, Val(N), q_grid)

# ╔═╡ e2e44eed-e2b7-456e-ae12-931ff8c14e07
#a_star = sequential_best_response(g_Cou)

# ╔═╡ 17105bd3-051e-4bd2-8295-de3161dae8ce
println("Nash equilibrium indices: $a_star")

# ╔═╡ f5340fd8-c260-4896-98b2-79a81eccf84b
#sequential_best_response(g_Cou, init_actions=[13, 13, 13])

# ╔═╡ 93470766-9ebb-4687-ac41-cbbcd9df92f1
#print_pure_nash_brute(g_Cou)

# ╔═╡ 48979918-5897-4e75-8d0d-3d6e60e884f4
N = 4

# ╔═╡ 43aca95d-dd9c-469c-b604-178b24128e16
q_grid_size = 61

# ╔═╡ ecda3b30-512f-477b-8936-96cc4a60a6e1
q_grid = range(0, step=div(a-c, q_grid_size-1), length=q_grid_size)  # [0, 1, 2, ..., 60]

# ╔═╡ d19ffbd4-6af0-475b-a27e-ca6c453d1188
println("Nash equilibrium quantities: $(q_grid[a_star])")

# ╔═╡ 23db7f72-e5c1-4863-954e-3621bbc5b0a6
g_Cou = cournot(a, c, Val(N), q_grid)

# ╔═╡ 4e50142a-1194-4bc2-bbac-f2d5089c60e8
is_nash(g_Cou, tuple(a_star...))

# ╔═╡ e82214e6-4274-4b38-b11d-ea3726e9d88d
sequential_best_response(g_Cou)

# ╔═╡ 90ccd311-cf6a-40ef-be5d-50d0f436156c


# ╔═╡ Cell order:
# ╠═ef459110-dd09-11eb-327d-7f6fa4737a51
# ╠═8561cceb-f46e-4e3a-b257-78d460bd827a
# ╠═3dbbbba3-fe72-4fa4-bc77-884d8b588de9
# ╠═f90c8299-dff7-4e61-ac76-284d4f5c5e6d
# ╠═66f85d91-58d9-4a9e-9bda-e39ea12253dd
# ╠═d7a839b3-f1d9-40bc-8610-e3770592aab4
# ╠═9b5567c5-0c42-4d39-a865-dc54f8611056
# ╠═cc7ea04e-ff21-4b14-bf4e-2a60a0576c0a
# ╠═d909bde2-1f79-4a29-996c-e839676825d5
# ╠═ebd8efa6-fdce-495d-a591-8fc45cd14e0a
# ╠═cb980d54-3322-4634-90a0-3363469926b9
# ╠═de96089c-42ee-4f0c-aee2-a58e011bbc46
# ╠═5b5e63a2-e509-473c-b103-dcf5a391ba37
# ╠═76fc34d6-e39f-4fa2-80ce-bc151af0a87c
# ╠═d51dcdba-e21f-443d-bc95-06fbfb1058af
# ╠═9884efb6-08a6-4636-a6a3-8d3b45714872
# ╠═4a0b9006-5532-46c0-8742-aab1041c92f5
# ╠═21110690-7971-48eb-b464-0ce440e75d96
# ╠═71754f38-9658-42ef-a7af-a7a263610e1c
# ╠═7ffe3816-49d8-4188-94e0-515d91a2aafe
# ╠═9f5fe5f7-b941-4f04-bae9-2bb8a3e8f9d1
# ╠═62d799f7-34c2-429c-8687-3dc3fd8550aa
# ╠═f4dbf36a-5ea9-432a-a3b4-9969e4b6067f
# ╠═1849456a-d485-48ca-a444-fe1e6f3d8b6e
# ╠═71dfa720-0d4b-47cb-9c3c-a6b0a254b272
# ╠═8c7e05ba-4032-4898-a40e-ad79e54b8505
# ╠═77d75ea0-5e72-4222-898d-0fe87431e833
# ╠═5889ca2e-2a5d-42b6-8156-6bcbe781977f
# ╠═1e7f7339-91ed-42be-ac6d-cc1c9a9e685b
# ╠═10feaece-5095-4545-91e0-fea2e14bbea7
# ╠═78c77f20-6826-4594-b055-0e1e27935fe2
# ╠═f040c2a2-cdc1-4923-94eb-6cbd9e4f840f
# ╠═e9250bd1-6cc9-4741-94b7-f4ce94ee0288
# ╠═1041a0af-3e98-443d-9754-ad62e1b9c57a
# ╠═d24dff90-9376-4215-8c82-cdaaa16fb7d9
# ╠═fc063148-fd9b-47ca-b02d-3b0769d98361
# ╠═44bae769-6e69-48f9-a26e-a4063d940934
# ╠═d12f6c05-39c0-4994-9d98-b617bf516e3a
# ╠═b17583ce-695f-43dc-87cc-e5cab672a455
# ╠═42a2c8d8-8f31-4951-afaa-1a0a7bd5158a
# ╠═014035d1-814e-4894-addd-3c9706952dde
# ╠═1fe3eeba-8026-4360-a661-1fbe25922378
# ╠═d2a6d877-5f6d-4a90-981f-480a30530085
# ╠═e7e37985-32d9-4859-81f5-bd570be7144b
# ╠═c71c26a7-5591-41e7-b8db-c5bed45fbe4b
# ╠═76c065bd-c0bf-4e71-a053-fed25a9853cc
# ╠═e2e44eed-e2b7-456e-ae12-931ff8c14e07
# ╠═17105bd3-051e-4bd2-8295-de3161dae8ce
# ╠═d19ffbd4-6af0-475b-a27e-ca6c453d1188
# ╠═f5340fd8-c260-4896-98b2-79a81eccf84b
# ╠═4e50142a-1194-4bc2-bbac-f2d5089c60e8
# ╠═93470766-9ebb-4687-ac41-cbbcd9df92f1
# ╠═48979918-5897-4e75-8d0d-3d6e60e884f4
# ╠═43aca95d-dd9c-469c-b604-178b24128e16
# ╠═ecda3b30-512f-477b-8936-96cc4a60a6e1
# ╠═23db7f72-e5c1-4863-954e-3621bbc5b0a6
# ╠═e82214e6-4274-4b38-b11d-ea3726e9d88d
# ╠═90ccd311-cf6a-40ef-be5d-50d0f436156c
