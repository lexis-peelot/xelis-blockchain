use log::trace;
use xelis_common::{
    block::BlockVersion,
    difficulty::Difficulty,
    network::Network,
    time::TimestampMillis,
    varuint::VarUint
};

use crate::config::{
    KILO_HASH,
    MEGA_HASH,
    GIGA_HASH,
    DEFAULT_MINIMUM_HASHRATE,
    MAINNET_MINIMUM_HASHRATE,
    MILLIS_PER_SECOND
};
use super::hard_fork::get_block_time_target_for_version;

mod v1;
mod v2;
mod v3;

// Kalman filter with unsigned integers only
// z: The observed value (latest hashrate calculated on current block time).
// x_est_prev: The previous hashrate estime.
// p_prev: The previous estimate covariance.
// Returns the new state estimate and covariance
fn kalman_filter(z: VarUint, x_est_prev: VarUint, p_prev: VarUint, shift: u64, left_shift: VarUint, process_noise_covar: VarUint) -> (VarUint, VarUint) {
    trace!("z: {}, x_est_prev: {}, p_prev: {}", z, x_est_prev, p_prev);
    // Scale up
    let z = z * left_shift;
    let r = z * 2;
    let x_est_prev = x_est_prev * left_shift;

    // Prediction step
    let p_pred = ((x_est_prev * process_noise_covar) >> shift) + p_prev;

    // Update step
    let k = (p_pred << shift) / (p_pred + r + VarUint::one());

    // Ensure positive numbers only
    let mut x_est_new = if z >= x_est_prev {
        x_est_prev + ((k * (z - x_est_prev)) >> shift)
    } else {
        x_est_prev - ((k * (x_est_prev - z)) >> shift)
    };

    trace!("x_est_new: {}, p pred: {}, noise covar: {}, p_prev: {}, k: {}", x_est_new, p_pred, process_noise_covar, p_prev, k);
    let p_new = ((left_shift - k) * p_pred) >> shift;

    // Scale down
    x_est_new >>= shift;

    (x_est_new, p_new)
}

// Calculate the required difficulty for versions that use a single solve-time
// measurement. V6 uses DAG-aware hashrate measurements through
// `calculate_difficulty_from_hashrate`.
pub fn calculate_difficulty(solve_time: TimestampMillis, previous_difficulty: Difficulty, p: VarUint, minimum_difficulty: Difficulty, version: BlockVersion) -> (Difficulty, VarUint) {
    let block_time_target = get_block_time_target_for_version(version);
    match version {
        BlockVersion::V0 => v1::calculate_difficulty(solve_time, previous_difficulty, p, minimum_difficulty, block_time_target),
        _ if version >= BlockVersion::V6 => (previous_difficulty.max(minimum_difficulty), p),
        _ => v2::calculate_difficulty(solve_time, previous_difficulty, p, minimum_difficulty, block_time_target, false),
    }
}

pub fn calculate_difficulty_from_hashrate(observed_hashrate: VarUint, previous_difficulty: Difficulty, p: VarUint, minimum_difficulty: Difficulty, version: BlockVersion, measurement_count: u64) -> (Difficulty, VarUint) {
    let block_time_target = get_block_time_target_for_version(version);
    v3::calculate_difficulty(observed_hashrate, previous_difficulty, p, minimum_difficulty, block_time_target, measurement_count)
}

// Get the initial DAA state for a version. The `p` storage/API name is
// historical: V1/V2 store covariance, while V3 stores packed filter state.
pub fn get_covariance_p(version: BlockVersion) -> VarUint {
    match version {
        BlockVersion::V0 => v1::P,
        BlockVersion::V6 => v3::P,
        _ => v2::P
    }
}

// Normalize the state stored in the historical `p` slot before V6 replay.
// Blocks before V6 store Kalman covariance there, not Gamma filter state.
pub fn normalize_daa_state_p(version: BlockVersion, p: VarUint) -> VarUint {
    if version >= BlockVersion::V6 { p } else { v3::P }
}

// Get the difficulty based on the hashrate and block time target
// NOTE: The caller must ensure that the block time provided is in milliseconds
pub const fn get_difficulty_with_target(hashrate: u64, block_time_target: u64) -> Difficulty {
    Difficulty::from_u64(hashrate * block_time_target / MILLIS_PER_SECOND)
}

// Get minimum difficulty based on the network
// Mainnet has a minimum difficulty to prevent spamming the network
// Testnet has a lower difficulty to allow faster block generation
pub const fn get_minimum_difficulty(network: &Network, version: BlockVersion) -> Difficulty {
    let hashrate = match network {
        Network::Mainnet => MAINNET_MINIMUM_HASHRATE,
        _ => DEFAULT_MINIMUM_HASHRATE,
    };

    let block_time_target = get_block_time_target_for_version(version);
    get_difficulty_with_target(hashrate, block_time_target)
}

// Get minimum difficulty at hard fork
pub const fn get_difficulty_at_hard_fork(network: &Network, version: BlockVersion) -> Option<Difficulty> {
    let hashrate = match network {
        Network::Mainnet => match version {
            BlockVersion::V0 | BlockVersion::V1 => 20 * KILO_HASH,
            BlockVersion::V2 => 2 * GIGA_HASH,
            BlockVersion::V3 => 200 * MEGA_HASH,
            BlockVersion::V4 | BlockVersion::V5 | BlockVersion::V6 => 400 * MEGA_HASH,
        },
        _ => return None,
    };

    let block_time_target = get_block_time_target_for_version(version);
    Some(get_difficulty_with_target(hashrate, block_time_target))
}

#[cfg(test)]
mod tests {
    use xelis_common::utils::format_hashrate;
    use crate::config::{HASH, MEGA_HASH};

    use super::*;

    #[test]
    fn test_difficulty_at_hard_fork() {
        // 20 KH/s
        assert_eq!(get_difficulty_at_hard_fork(&Network::Mainnet, BlockVersion::V0).unwrap(), Difficulty::from_u64(15 * 20 * KILO_HASH));
        // 2 GH/s
        assert_eq!(get_difficulty_at_hard_fork(&Network::Mainnet, BlockVersion::V2).unwrap(), Difficulty::from_u64(15 * 2 * GIGA_HASH));

        // 2 KH/s per second for whole testnet
        for version in [BlockVersion::V0, BlockVersion::V1, BlockVersion::V2, BlockVersion::V3] {
            assert!(get_difficulty_at_hard_fork(&Network::Testnet, version).is_none());
        }
    }

    #[test]
    fn test_daa_state_p_resets_before_v6() {
        let old_p = VarUint::from_u64(12345);

        assert_eq!(normalize_daa_state_p(BlockVersion::V5, old_p), v3::P);
        assert_eq!(normalize_daa_state_p(BlockVersion::V6, old_p), old_p);
    }

    #[test]
    fn test_const_hashrate_format() {
        assert_eq!(format_hashrate(HASH as f64), "1.00 H/s");
        assert_eq!(format_hashrate(KILO_HASH as f64), "1.00 KH/s");
        assert_eq!(format_hashrate(MEGA_HASH as f64), "1.00 MH/s");
        assert_eq!(format_hashrate(GIGA_HASH as f64), "1.00 GH/s");
    }
}
