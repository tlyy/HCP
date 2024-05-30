for i in 1; do
    for game in Assault Asterix BankHeist BattleZone ChopperCommand CrazyClimber Freeway Frostbite Gopher Kangaroo KungFuMaster Pong Qbert UpNDown; do
        python -m bbf.train \
                --agent=HCP \
                --gin_files=bbf/configs/HCP.gin \
                --base_dir=exp/hcp/$game/$i \
                --gin_bindings="DataEfficientAtariRunner.game_name = '$game'"
    done
done


# Assault Asterix BankHeist BattleZone ChopperCommand CrazyClimber Freeway Frostbite Gopher Kangaroo KungFuMaster Pong Qbert UpNDown
# Alien Amidar Boxing Breakout DemonAttack Hero Jamesbond Krull MsPacman PrivateEye RoadRunner Seaquest



# export MUJOCO_GL=osmesa
# for i in 1; do
#     for game in cheetah-run cartpole-swingup reacher-easy finger-spin ball_in_cup-catch walker-walk; do
#         python -m continuous_control.train \
#                 --save_dir=exp_con/train/$game/$i \
#                 --env_name $game \
#                 --max_steps 100000 \
#                 --resets \
#                 --reset_interval 20000 \
#                 --hcp \
#                 --threshold 0.0 \
#                 --rsp_weight 0.8
#     done
# done