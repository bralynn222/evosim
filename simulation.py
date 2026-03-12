"""
Evolution and simulation system with 3 separate neural networks.
1. Gameplay Network - Controls movement, attack, moment-to-moment decisions
2. Stat Plan Network - Decides build order for levels 1-10
3. Shop Plan Network - Decides shopping priority order
"""

import os
import sys
import numpy as np

# CUDA setup patch
def _patch_cupy_compiler():
    base_path = os.path.dirname(os.path.dirname(sys.executable))
    candidates = [
        os.path.join(base_path, 'include'),
        os.path.join(base_path, 'targets', 'x86_64-linux', 'include'),
        '/usr/local/cuda/include',
        '/usr/include',
        os.environ.get('CUDA_PATH', '') + '/include'
    ]
    header_path = None
    for path in candidates:
        if os.path.exists(os.path.join(path, 'cuda_fp16.h')):
            header_path = path
            break

    if header_path:
        os.environ['CUDA_PATH'] = os.path.dirname(header_path)

    try:
        from cupy.cuda import compiler
        _original_compile = compiler.compile_using_nvrtc

        def _compile_with_includes(source, options, *args, **kwargs):
            opts_list = list(options) if options else []
            if header_path:
                flag = f'-I{header_path}'
                if flag not in opts_list:
                    opts_list.append(flag)
            return _original_compile(source, opts_list, *args, **kwargs)

        compiler.compile_using_nvrtc = _compile_with_includes
    except ImportError:
        pass

_patch_cupy_compiler()

try:
    import cupy as cp
except ImportError:
    print("CRITICAL: CuPy not found. This simulation requires an NVIDIA GPU and CuPy.")
    sys.exit(1)

from config_and_kernels import *
from physics_kernels import PHYSICS_KERNELS


class AdamOptimizer:
    """Adam optimizer for Evolution Strategy."""
    def __init__(self, size, alpha=0.01):
        self.m = cp.zeros(size, dtype=cp.float32)
        self.v = cp.zeros(size, dtype=cp.float32)
        self.t = 0
        self.alpha = alpha
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def update(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params + self.alpha * m_hat / (cp.sqrt(v_hat) + self.epsilon)


def convert_neat_to_mlp(genome_np):
    """Convert NEAT genome to MLP weight structure."""
    print("... Converting best NEAT genome to MLP structure ...")
    hidden_nodes = set()
    for gene in genome_np:
        if gene[3] > 0.5:
            src, dst = int(gene[0]), int(gene[1])
            if src >= INPUTS: hidden_nodes.add(src)
            if dst >= INPUTS + OUTPUTS: hidden_nodes.add(dst)

    sorted_hidden = sorted(list(hidden_nodes))
    if len(sorted_hidden) > N_HID:
        sorted_hidden = sorted_hidden[:N_HID]

    node_map = {node_id: i for i, node_id in enumerate(sorted_hidden)}
    W1 = np.zeros((N_IN, N_HID), dtype=np.float32)
    B1 = np.zeros(N_HID, dtype=np.float32)
    W2 = np.zeros((N_HID, N_OUT), dtype=np.float32)
    B2 = np.zeros(N_OUT, dtype=np.float32)

    for gene in genome_np:
        if gene[3] > 0.5:
            src, dst, weight = int(gene[0]), int(gene[1]), gene[2]
            if src < INPUTS and dst in node_map:
                W1[src, node_map[dst]] = weight
            elif src in node_map and INPUTS <= dst < INPUTS + OUTPUTS:
                W2[node_map[src], dst - INPUTS] = weight

    return np.concatenate([W1.flatten(), B1, W2.flatten(), B2])


class Simulation:
    """Main simulation class with 3 separate NNs."""

    def __init__(self):
        # Compile CUDA kernels
        src = CUDA_SRC + PHYSICS_KERNELS
        src = src.replace('${N_IN}', str(N_IN)) \
            .replace('${N_HID}', str(N_HID)) \
            .replace('${N_OUT}', str(N_OUT)) \
            .replace('${TOTAL_PARAMS}', str(TOTAL_PARAMS)) \
            .replace('${STAT_PLAN_HID}', str(STAT_PLAN_HID)) \
            .replace('${STAT_PLAN_OUT}', str(STAT_PLAN_OUT)) \
            .replace('${STAT_PLAN_PARAMS}', str(STAT_PLAN_PARAMS)) \
            .replace('${SHOP_PLAN_HID}', str(SHOP_PLAN_HID)) \
            .replace('${SHOP_PLAN_OUT}', str(SHOP_PLAN_OUT)) \
            .replace('${SHOP_PLAN_PARAMS}', str(SHOP_PLAN_PARAMS)) \
            .replace('${STRAT_IN}', str(STRAT_IN)) \
            .replace('${STRAT_HID}', str(STRAT_HID)) \
            .replace('${STRAT_OUT}', str(STRAT_OUT)) \
            .replace('${STRAT_PARAMS}', str(STRAT_PARAMS))

        self.mod = cp.RawModule(code=src)

        # Get kernel functions
        self.k_neat_to_adj = self.mod.get_function('genome_to_adjacency')
        self.k_neat_brain = self.mod.get_function('neat_brain_inference')
        self.k_neat_evolve = self.mod.get_function('neat_evolve')

        # Three separate networks
        self.k_gameplay_brain = self.mod.get_function('dense_inference')
        self.k_stat_plan_brain = self.mod.get_function('stat_plan_inference')
        self.k_shop_plan_brain = self.mod.get_function('shop_plan_inference')

        self.k_upd_player = self.mod.get_function('update_player')
        self.k_upd_swords = self.mod.get_function('update_swords')
        self.k_upd_enemies = self.mod.get_function('update_enemies')
        self.k_pack = self.mod.get_function('pack_render_single')

        # Initialize GPU arrays
        total_ents = POP_SIZE * TOTAL_ENTITIES
        self.pos_x = cp.zeros(total_ents, cp.float32)
        self.pos_y = cp.zeros(total_ents, cp.float32)
        self.angle = cp.zeros(total_ents, cp.float32)
        self.active = cp.zeros(total_ents, cp.float32)
        self.type = cp.zeros(total_ents, cp.float32)
        self.timers = cp.zeros(total_ents, cp.float32)
        self.idle_timers = cp.zeros(total_ents, cp.float32)

        self.spawn_timers = cp.zeros(POP_SIZE, cp.float32)
        self.fitness = cp.zeros(POP_SIZE, cp.float32)
        self.kill_counts = cp.zeros(POP_SIZE, cp.float32)
        self.survival_frames = cp.zeros(POP_SIZE, cp.float32)
        self.rng = cp.random.randint(0, 2 ** 30, POP_SIZE, dtype=cp.uint32)

        self.exp = cp.zeros(POP_SIZE, cp.float32)
        self.level = cp.zeros(POP_SIZE, cp.float32)
        self.stat_points = cp.zeros(POP_SIZE, cp.float32)
        self.stat_speed = cp.zeros(POP_SIZE, cp.float32)
        self.stat_atk = cp.zeros(POP_SIZE, cp.float32)
        self.stat_hp_max = cp.zeros(POP_SIZE, cp.float32)
        self.gold = cp.zeros(POP_SIZE, cp.float32)
        self.triple_shot_flag = cp.zeros(POP_SIZE, cp.float32)

        self.current_hp = cp.zeros(total_ents, cp.float32)
        self.invincibility = cp.zeros(total_ents, cp.float32)
        self.solid_timestamp = cp.full(POP_SIZE, -1.0, dtype=cp.float32)
        self.shop_reward_claimed = cp.zeros(POP_SIZE, cp.float32)

        self.inputs = cp.zeros(POP_SIZE * INPUTS, cp.float32)
        self.outputs = cp.zeros(POP_SIZE * OUTPUTS, cp.float32)

        # NEAT arrays
        self.genes = cp.zeros((POP_SIZE, MAX_GENES, 6), cp.float32)
        self.new_genes = cp.zeros((POP_SIZE, MAX_GENES, 6), cp.float32)
        self.adj = cp.zeros((POP_SIZE, MAX_NODES, MAX_NODES), cp.float32)
        self.global_innov = cp.array([INPUTS * OUTPUTS + 1], dtype=cp.int32)

        # NETWORK 1: Gameplay
        self.gameplay_weights = cp.zeros(TOTAL_PARAMS, cp.float32)
        self.gameplay_optimizer = AdamOptimizer(TOTAL_PARAMS, alpha=ALPHA)
        self.gameplay_noise = cp.zeros((POP_SIZE, TOTAL_PARAMS), cp.float32)
        self.gameplay_current_weights = cp.zeros((POP_SIZE, TOTAL_PARAMS), cp.float32)

        # NETWORK 2: Stat Plan
        self.stat_plan_weights = cp.zeros(STAT_PLAN_PARAMS, cp.float32)
        self.stat_plan_optimizer = AdamOptimizer(STAT_PLAN_PARAMS, alpha=ALPHA)
        self.stat_plan_noise = cp.zeros((POP_SIZE, STAT_PLAN_PARAMS), cp.float32)
        self.stat_plan_current_weights = cp.zeros((POP_SIZE, STAT_PLAN_PARAMS), cp.float32)
        self.stat_plan_outputs = cp.zeros(POP_SIZE * 10, cp.float32)
        self.stat_plan_choices = cp.zeros(POP_SIZE * 10, cp.int32)

        # NETWORK 3: Shop Plan
        self.shop_plan_weights = cp.zeros(SHOP_PLAN_PARAMS, cp.float32)
        self.shop_plan_optimizer = AdamOptimizer(SHOP_PLAN_PARAMS, alpha=ALPHA)
        self.shop_plan_noise = cp.zeros((POP_SIZE, SHOP_PLAN_PARAMS), cp.float32)
        self.shop_plan_current_weights = cp.zeros((POP_SIZE, SHOP_PLAN_PARAMS), cp.float32)
        self.shop_plan_outputs = cp.zeros(POP_SIZE * 4, cp.float32)
        self.shop_plan_priority = cp.zeros(POP_SIZE * 4, cp.int32)

        self.render_buf = cp.zeros(TOTAL_ENTITIES * 6, dtype=cp.float32)

        self.training_mode = 'NEAT'
        self.gen = 1
        self.frame = 0
        self.shop_objective_active = 0.0

        self.init_neat()
        self.init_physics()
        print("✓ Simulation initialized with 3 separate networks:")
        print(f"  - Gameplay: {TOTAL_PARAMS} params")
        print(f"  - Stat Plan: {STAT_PLAN_PARAMS} params")
        print(f"  - Shop Plan: {SHOP_PLAN_PARAMS} params")

    def init_neat(self):
        """Initialize NEAT population."""
        print("Initializing NEAT population...")
        host_genes = np.zeros((POP_SIZE, MAX_GENES, 6), dtype=np.float32)
        innov = 1
        cnt = 0
        for i in range(INPUTS):
            for j in range(OUTPUTS):
                host_genes[:, cnt, 0] = i
                host_genes[:, cnt, 1] = INPUTS + j
                host_genes[:, cnt, 2] = np.random.randn(POP_SIZE) * 0.1
                host_genes[:, cnt, 3] = 1.0
                host_genes[:, cnt, 4] = innov
                innov += 1
                cnt += 1
        self.genes = cp.asarray(host_genes)
        self.global_innov[0] = innov

    def init_physics(self, bias_shop=True):
        """Reset physics state."""
        self.pos_x.fill(SCREEN_W / 2)
        self.pos_y.fill(SCREEN_H / 2)
        self.active.fill(0)
        self.active[IDX_P1::TOTAL_ENTITIES] = 1.0

        self.type.fill(0)
        self.type[IDX_SHOP::TOTAL_ENTITIES] = 3
        for i in range(50): self.type[(IDX_ENEMY_START + i)::TOTAL_ENTITIES] = 1
        for i in range(SWORD_COUNT): self.type[(IDX_SWORD1_START + i)::TOTAL_ENTITIES] = 2

        self.spawn_timers.fill(0)
        self.timers.fill(0)
        self.idle_timers.fill(0)
        self.kill_counts.fill(0)
        self.survival_frames.fill(0)
        self.fitness.fill(0)

        self.exp.fill(0)
        self.level.fill(1)
        self.stat_points.fill(0)
        self.stat_speed.fill(0)
        self.stat_atk.fill(0)
        self.stat_hp_max.fill(1)
        self.gold.fill(0)
        self.triple_shot_flag.fill(0)

        self.current_hp.fill(0)
        self.current_hp[IDX_P1::TOTAL_ENTITIES] = 1.0
        self.invincibility.fill(0)
        self.shop_reward_claimed.fill(0.0)

        if bias_shop:
            rng_vals = cp.random.rand(POP_SIZE, dtype=cp.float32)
            self.solid_timestamp = cp.where(rng_vals < 0.1, 0.0, -1.0).astype(cp.float32)
            self.solid_timestamp[0] = 0.0
        else:
            self.solid_timestamp.fill(-1.0)

        self.frame = 0

    def generate_plans(self):
        """Generate stat and shop plans."""
        grid_pop = (POP_SIZE + 255) // 256

        # Generate stat plans
        self.k_stat_plan_brain((grid_pop,), (256,),
            (self.stat_plan_current_weights.ravel(), self.stat_plan_outputs, POP_SIZE))

        stat_out_cpu = self.stat_plan_outputs.get().reshape(POP_SIZE, 10)
        stat_choices = np.clip(stat_out_cpu.astype(int), 0, 3)
        self.stat_plan_choices = cp.asarray(stat_choices.ravel())

        # Generate shop plans
        self.k_shop_plan_brain((grid_pop,), (256,),
            (self.shop_plan_current_weights.ravel(), self.shop_plan_outputs, POP_SIZE))

        shop_out_cpu = self.shop_plan_outputs.get().reshape(POP_SIZE, 4)
        shop_priorities = np.argsort(-shop_out_cpu, axis=1)
        self.shop_plan_priority = cp.asarray(shop_priorities.ravel())

    def transition_to_es(self, winning_genome_gpu):
        """Transition from NEAT to 3-network ES."""
        print("\n" + "=" * 60)
        print(" TRANSITIONING TO 3-NETWORK ES ")
        print("=" * 60 + "\n")
        self.training_mode = 'ES'

        winning_genome_np = winning_genome_gpu.get()
        mlp_params = convert_neat_to_mlp(winning_genome_np)
        self.gameplay_weights = cp.asarray(mlp_params)

        self.stat_plan_weights = cp.random.randn(STAT_PLAN_PARAMS, dtype=cp.float32) * 0.1
        self.shop_plan_weights = cp.random.randn(SHOP_PLAN_PARAMS, dtype=cp.float32) * 0.1

        self.gen = 1
        self.init_physics()

    def evolve_neat(self):
        """NEAT evolution."""
        best_idx = int(cp.argmax(self.fitness))
        max_fit = float(self.fitness[best_idx])
        print(f"Gen {self.gen} [NEAT] | MaxFit: {max_fit:.1f} | AvgKills: {float(cp.mean(self.kill_counts)):.2f}")

        if max_fit > 1000.0:
            self.transition_to_es(self.genes[best_idx])
            return

        grid = (POP_SIZE + 255) // 256
        self.k_neat_evolve((grid,), (256,),
                           (self.genes, self.new_genes, self.fitness, self.global_innov, self.rng, POP_SIZE))
        self.genes, self.new_genes = self.new_genes, self.genes
        self.gen += 1
        self.init_physics()

    def gradient_step_es(self):
        """ES optimization for all 3 networks."""
        ranks = cp.empty(POP_SIZE, cp.float32)
        ranks[cp.argsort(self.fitness)] = cp.linspace(-0.5, 0.5, POP_SIZE)

        gameplay_grad = cp.dot(self.gameplay_noise.T, ranks) / (POP_SIZE * SIGMA)
        self.gameplay_weights = self.gameplay_optimizer.update(self.gameplay_weights, gameplay_grad)
        self.gameplay_weights *= 0.9995

        stat_grad = cp.dot(self.stat_plan_noise.T, ranks) / (POP_SIZE * SIGMA)
        self.stat_plan_weights = self.stat_plan_optimizer.update(self.stat_plan_weights, stat_grad)

        shop_grad = cp.dot(self.shop_plan_noise.T, ranks) / (POP_SIZE * SIGMA)
        self.shop_plan_weights = self.shop_plan_optimizer.update(self.shop_plan_weights, shop_grad)

        print(f"Gen {self.gen} [3-Net ES] | MaxFit: {float(cp.max(self.fitness)):.1f} | AvgKills: {float(cp.mean(self.kill_counts)):.2f}")

        if self.gen % 50 == 0:
            self.print_best_plans()

        self.gen += 1
        self.init_physics()

    def print_best_plans(self):
        """Print best plans."""
        best_idx = int(cp.argmax(self.fitness))
        stat_choices = self.stat_plan_choices.get().reshape(POP_SIZE, 10)[best_idx]
        shop_priorities = self.shop_plan_priority.get().reshape(POP_SIZE, 4)[best_idx]

        stats = ["Speed", "Attack", "HP", "Save"]
        items = ["Speed", "Attack", "HP", "Triple"]

        print("\n" + "=" * 50)
        print(f"Best Build (Gen {self.gen}):")
        for i, choice in enumerate(stat_choices):
            print(f"  Lv{i+1}: {stats[choice]}")
        print(f"Shop Priority: {', '.join([items[i] for i in shop_priorities])}")
        print("=" * 50 + "\n")

    def update(self):
        """Run one frame."""
        self.frame += 1
        grid_pop = (POP_SIZE + 255) // 256

        if self.training_mode == 'NEAT':
            if self.frame == 1:
                self.k_neat_to_adj((grid_pop,), (256,), (self.genes, self.adj, POP_SIZE))
            if self.frame % 4 == 0:
                self.k_neat_brain((grid_pop,), (256,), (self.inputs, self.adj, self.outputs, POP_SIZE))
        else:
            if self.frame == 1:
                self.gameplay_noise = cp.random.randn(POP_SIZE, TOTAL_PARAMS, dtype=cp.float32)
                self.gameplay_current_weights = self.gameplay_weights + (SIGMA * self.gameplay_noise)

                self.stat_plan_noise = cp.random.randn(POP_SIZE, STAT_PLAN_PARAMS, dtype=cp.float32)
                self.stat_plan_current_weights = self.stat_plan_weights + (SIGMA * self.stat_plan_noise)

                self.shop_plan_noise = cp.random.randn(POP_SIZE, SHOP_PLAN_PARAMS, dtype=cp.float32)
                self.shop_plan_current_weights = self.shop_plan_weights + (SIGMA * self.shop_plan_noise)

                self.generate_plans()

            if self.frame % 4 == 0:
                self.k_gameplay_brain((grid_pop,), (256,),
                                (self.inputs, self.gameplay_current_weights.ravel(), self.outputs, POP_SIZE))

        self.k_upd_player((grid_pop,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active,
            self.timers, self.fitness, self.kill_counts, self.survival_frames, self.rng,
            self.outputs, self.inputs, self.spawn_timers, self.idle_timers,
            self.exp, self.level, self.stat_points, self.stat_speed, self.stat_atk, self.stat_hp_max,
            self.current_hp, self.invincibility, self.gold, self.triple_shot_flag,
            self.frame, self.solid_timestamp,
            cp.float32(self.shop_objective_active),
            self.shop_reward_claimed,
            self.stat_plan_choices,
            self.shop_plan_priority,
            POP_SIZE
        ))

        total_swords = POP_SIZE * SWORD_COUNT
        self.k_upd_swords(((total_swords + 255) // 256,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers,
            self.outputs, self.stat_atk, self.triple_shot_flag, self.rng,
            total_swords
        ))

        total_enemies = POP_SIZE * ENEMY_COUNT
        self.k_upd_enemies(((total_enemies + 255) // 256,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active,
            self.fitness, self.kill_counts, self.exp, self.gold,
            self.current_hp, self.invincibility, self.solid_timestamp,
            self.spawn_timers, self.rng,
            total_enemies, self.frame
        ))

        alive = cp.sum(self.active[IDX_P1::TOTAL_ENTITIES])
        if self.frame > 36000 or alive < 1:
            if self.training_mode == 'NEAT':
                self.evolve_neat()
            else:
                self.gradient_step_es()

    def pack_one_arena(self, arena_idx):
        """Pack render data."""
        grid_render = (TOTAL_ENTITIES + 255) // 256
        self.k_pack((grid_render,), (256,),
                    (self.pos_x, self.pos_y, self.angle, self.active, self.type, self.render_buf, int(arena_idx)))

    def prepare_for_human_play(self):
        """Reset for human play."""
        print("Resetting for Human Play...")
        self.init_physics(bias_shop=False)

    def apply_human_upgrade(self, choice):
        """Apply upgrade."""
        if choice == 1: self.stat_speed[0] += 1.0
        elif choice == 2: self.stat_atk[0] += 1.0
        elif choice == 3: self.stat_hp_max[0] += 1.0; self.current_hp[IDX_P1] += 1.0
        self.stat_points[0] -= 1.0

    def human_buy_item(self, item_id):
        """Buy item."""
        gold = float(self.gold[0])
        cost = 50.0 if item_id != 3 else 2500.0
        if gold >= cost:
            self.gold[0] -= cost
            if item_id == 0: self.stat_speed[0] += 2.0
            elif item_id == 1: self.stat_atk[0] += 1.0
            elif item_id == 2: self.stat_hp_max[0] += 2.0; self.current_hp[IDX_P1] = self.stat_hp_max[0]
            elif item_id == 3: self.triple_shot_flag[0] = 1.0

    def update_human_arena(self, actions):
        """Update human player."""
        act_gpu = cp.zeros(7, dtype=cp.float32)
        act_gpu[0] = actions['mx']
        act_gpu[1] = actions['my']
        act_gpu[2] = actions['angle']
        act_gpu[3] = actions['attack']
        self.outputs[0:7] = act_gpu

        grid_pop = (POP_SIZE + 255) // 256
        dummy_stat = cp.zeros(POP_SIZE * 10, cp.int32)
        dummy_shop = cp.zeros(POP_SIZE * 4, cp.int32)

        self.k_upd_player((grid_pop,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active,
            self.timers, self.fitness, self.kill_counts, self.survival_frames, self.rng,
            self.outputs, self.inputs, self.spawn_timers, self.idle_timers,
            self.exp, self.level, self.stat_points, self.stat_speed, self.stat_atk, self.stat_hp_max,
            self.current_hp, self.invincibility, self.gold, self.triple_shot_flag,
            self.frame, self.solid_timestamp,
            cp.float32(self.shop_objective_active),
            self.shop_reward_claimed,
            dummy_stat, dummy_shop, POP_SIZE
        ))

        total_swords = POP_SIZE * SWORD_COUNT
        self.k_upd_swords(((total_swords + 255) // 256,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers,
            self.outputs, self.stat_atk, self.triple_shot_flag, self.rng,
            total_swords
        ))

        total_enemies = POP_SIZE * ENEMY_COUNT
        self.k_upd_enemies(((total_enemies + 255) // 256,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active,
            self.fitness, self.kill_counts, self.exp, self.gold,
            self.current_hp, self.invincibility, self.solid_timestamp,
            self.spawn_timers, self.rng,
            total_enemies, self.frame
        ))

        self.pack_one_arena(0)

        s_idx, p_idx = IDX_SHOP, IDX_P1
        dx = self.pos_x[p_idx] - self.pos_x[s_idx]
        dy = self.pos_y[p_idx] - self.pos_y[s_idx]
        near_shop = (dx * dx + dy * dy) < 2500.0

        return {
            'alive': self.active[IDX_P1].get() > 0.5,
            'kills': int(self.kill_counts[0].get()),
            'lvl': int(self.level[0].get()),
            'points': int(self.stat_points[0].get()),
            'hp': int(self.current_hp[IDX_P1].get()),
            'max_hp': int(self.stat_hp_max[0].get()),
            'solid': self.solid_timestamp[0].get() != -1.0,
            'gold': int(self.gold[0].get()),
            'triple': self.triple_shot_flag[0].get() > 0.5,
            'near_shop': bool(near_shop.get())
        }