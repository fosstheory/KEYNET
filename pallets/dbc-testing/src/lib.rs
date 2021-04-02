#![recursion_limit = "128"]
#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::traits::Currency;
use frame_support::{debug, decl_module, decl_storage, dispatch::DispatchResult};
use frame_system::{self as system, ensure_root, ensure_signed};
use phase_reward::PhaseReward;
use sp_std::{convert::TryInto, str};

type BalanceOf<T> =
    <<T as Config>::Currency as Currency<<T as system::Config>::AccountId>>::Balance;

// pub trait Config: frame_system::Config + timestamp::Config {}
pub trait Config: frame_system::Config + babe::Config {
    type Currency: Currency<Self::AccountId>;
    type PhaseReward: PhaseReward<Balance = BalanceOf<Self>>;
}

decl_storage! {
    trait Store for Module<T: Config> as DBCTesting {
        pub Thing1 get(fn thing1): u64;
        pub Thing2 get(fn thing2): u64;
        pub Thing3 get(fn thing3): u64;
    }
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {

        #[weight = 0]
        pub fn say_hello(origin) -> DispatchResult{
            let secs_per_block = babe::Module::<T>::slot_duration();
            let secs_per_block2 = <babe::Module::<T>>::slot_duration();

            // let secs_per_block =<T as babe::Config>::slot_duration();
            // let secs_per_block = <T as timestamp::Config>::MinimumPeriod::get();
            let caller = ensure_signed(origin)?;

            let mut output: [u8; 64] = [0; 64];
            let decoded = bs58::decode("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY").into(&mut output);
            debug::info!("########## decoded2 Alice: {:?}, {:?}", decoded, output);

            // TODO: convert to str
            debug::info!("######### Request sent by: {:?}, {:?}, {:?} #########", caller, secs_per_block, secs_per_block2);
            Ok(())
        }

        #[weight = 0]
        pub fn set_phase0_reward(origin, reward_balance: BalanceOf<T>) -> DispatchResult {
            ensure_root(origin)?;
            T::PhaseReward::set_phase0_reward(reward_balance);
            Ok(())
        }

        #[weight = 0]
        pub fn set_phase1_reward(origin, reward_balance :BalanceOf<T>) -> DispatchResult{
            ensure_root(origin)?;
            T::PhaseReward::set_phase1_reward(reward_balance);
            Ok(())
        }

        #[weight = 0]
        pub fn set_phase2_reward(origin,reward_balance:BalanceOf<T>) -> DispatchResult{
            ensure_root(origin)?;
            T::PhaseReward::set_phase2_reward(reward_balance);
            Ok(())
        }
    }
}

impl<T: Config> Module<T> {
    pub fn ss58_decode(s: &str) -> Option<[u8; 32]> {
        const CHECKSUM_LEN: usize = 2;
        let mut data: [u8; 36] = [0; 36];
        let decoded = bs58::decode(s).into(&mut data);
        debug::info!("####### decoded2: {:?}, {:?}", decoded, data);

        let (prefix_len, ident) = match data[0] {
            0..=63 => (1, data[0] as u16),
            64..=127 => {
                // let lower = (data[0] << 2) | (data[1] >> 6);
                // let upper = data[1] & 0b00111111;
                // (2, (lower as u16) | ((upper as u16) << 8))

                // TODO: should support in the future
                (1, 1)
            }
            _ => (1, 1), // TODO: return error here
        };

        return Some(data[1..35].try_into().unwrap());

        // return Some(data[1..34].try_into());
        // let public_key = data[1..34].clone();
    }
}
