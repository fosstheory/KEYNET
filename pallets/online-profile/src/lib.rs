#![recursion_limit = "256"]
#![cfg_attr(not(feature = "std"), no_std)]

use codec::{Decode, Encode};
use frame_support::{
    dispatch::DispatchResultWithPostInfo,
    pallet_prelude::*,
    traits::{
        Currency, ExistenceRequirement::AllowDeath, Get, LockIdentifier, LockableCurrency,
        Randomness, WithdrawReasons,
    },
    IterableStorageMap,
};
use frame_system::pallet_prelude::*;
use online_profile_machine::{CommOps, LCOps, OPOps};
use sp_core::H256;
use sp_runtime::{
    traits::{AccountIdConversion, BlakeTwo256, CheckedSub, SaturatedConversion, Zero},
    ModuleId, RandomNumberGenerator,
};
use sp_std::{collections::btree_set::BTreeSet, convert::TryInto, prelude::*, str};

pub mod types;
use types::*;

pub use pallet::*;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub const PALLET_LOCK_ID: LockIdentifier = *b"oprofile";
pub const PALLET_ID: ModuleId = ModuleId(*b"MCStake!");
pub const MAX_UNLOCKING_CHUNKS: usize = 32;

type BalanceOf<T> =
    <<T as pallet::Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

#[frame_support::pallet]
pub mod pallet {
    use super::*;

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type Event: From<Event<Self>> + IsType<<Self as frame_system::Config>::Event>;
        type Currency: LockableCurrency<Self::AccountId, Moment = Self::BlockNumber>;
        type RandomnessSource: Randomness<H256>;
        type BlockPerEra: Get<u32>;
        type BondingDuration: Get<EraIndex>;
    }

    #[pallet::pallet]
    #[pallet::generate_store(pub(super) trait Store)]
    pub struct Pallet<T>(_);

    // 存储用户机器在线收益
    #[pallet::type_value]
    pub fn HistoryDepthDefault<T: Config>() -> u32 {
        150
    }

    #[pallet::storage]
    #[pallet::getter(fn history_depth)]
    pub(super) type HistoryDepth<T: Config> =
        StorageValue<_, u32, ValueQuery, HistoryDepthDefault<T>>;

    // 用户提交绑定请求
    // 委员会可以查询可以抢单的机器
    #[pallet::storage]
    #[pallet::getter(fn bonding_queue)]
    pub type BondingQueue<T> = StorageMap<
        _,
        Blake2_128Concat,
        MachineId,
        BondingPair<<T as frame_system::Config>::AccountId>,
        ValueQuery,
    >;

    #[pallet::storage]
    #[pallet::getter(fn booking_queue)]
    pub type BookingQueue<T> = StorageMap<
        _,
        Blake2_128Concat,
        MachineId,
        BookingItem<<T as frame_system::Config>::BlockNumber>, // TODO: 修改类型 需要有height, who, machineid
    >;

    #[pallet::storage]
    #[pallet::getter(fn booked_queue)]
    pub type BookedQueue<T> = StorageMap<_, Blake2_128Concat, MachineId, u64, ValueQuery>; //TODO: 修改类型，保存已经被委员会预订的机器

    /// Machine has been bonded
    #[pallet::storage]
    #[pallet::getter(fn bonded_machine)]
    pub type BondedMachine<T> = StorageMap<_, Blake2_128Concat, MachineId, (), ValueQuery>;

    // 存储ocw获取的机器打分信息
    #[pallet::storage]
    #[pallet::getter(fn ocw_machine_grades)]
    pub type OCWMachineGrades<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        MachineId,
        ConfirmedMachine<T::AccountId, T::BlockNumber>,
        ValueQuery,
    >;

    // 存储ocw获取的机器估价信息
    #[pallet::storage]
    #[pallet::getter(fn ocw_machine_price)]
    pub type OCWMachinePrice<T> = StorageMap<_, Blake2_128Concat, MachineId, u64, ValueQuery>;

    // 存储委员会确认的信息
    #[pallet::storage]
    #[pallet::getter(fn machine_grade)]
    pub type MachineGrade<T> = StorageMap<
        _,
        Blake2_128Concat,
        MachineId,
        MachineGradeInfo<<T as frame_system::Config>::AccountId>,
        ValueQuery,
    >;

    // ocw查询, 并确认绑定结果
    /// store user's machine
    #[pallet::storage]
    #[pallet::getter(fn user_bonded_machine)]
    pub(super) type UserBondedMachine<T> = StorageMap<
        _,
        Blake2_128Concat,
        <T as frame_system::Config>::AccountId,
        Vec<MachineId>,
        ValueQuery,
    >;

    /// Map from all (unlocked) "controller" accounts to the info regarding the staking.
    #[pallet::storage]
    #[pallet::getter(fn ledger)]
    pub(super) type Ledger<T> = StorageDoubleMap<
        _,
        Blake2_128Concat,
        <T as frame_system::Config>::AccountId,
        Blake2_128Concat,
        MachineId,
        Option<StakingLedger<<T as frame_system::Config>::AccountId, BalanceOf<T>>>,
        ValueQuery,
    >;

    /// 机器等待奖励加入到能获取奖励的队列
    #[pallet::storage]
    #[pallet::getter(fn pending_machine)]
    pub(super) type PendingMachine<T> = StorageMap<_, Blake2_128Concat, MachineId, ()>;

    #[pallet::storage]
    #[pallet::getter(fn eras_reward_points)]
    pub(super) type ErasRewardPoints<T> = StorageMap<
        _,
        Blake2_128Concat,
        EraIndex,
        EraRewardPoints<<T as frame_system::Config>::AccountId>,
    >;

    /// MachineDetail
    #[pallet::storage]
    #[pallet::getter(fn machine_detail)]
    pub type MachineDetail<T> = StorageMap<
        _,
        Blake2_128Concat,
        MachineId,
        MachineMeta<<T as frame_system::Config>::AccountId>,
        ValueQuery,
    >;

    // pub RandNonce get(fn rand_nonce) config(): u64 = 0;
    // nonce to generate random number for selecting committee
    #[pallet::type_value]
    pub(super) fn RandNonceDefault<T: Config>() -> u64 {
        0
    }

    #[pallet::storage]
    pub(super) type RandNonce<T: Config> = StorageValue<_, u64, ValueQuery, RandNonceDefault<T>>;

    /// user machine total grades
    #[pallet::storage]
    #[pallet::getter(fn user_total_machine_grades)]
    pub(super) type UserTotalMachineGrades<T> =
        StorageMap<_, Blake2_128Concat, <T as frame_system::Config>::AccountId, u64>;

    // /// sum of user released reward
    #[pallet::storage]
    #[pallet::getter(fn user_released_reward)]
    pub(super) type UserReleasedReward<T> =
        StorageMap<_, Blake2_128Concat, <T as frame_system::Config>::AccountId, BalanceOf<T>>;

    /// user daily reward: record 150days of daily reward
    #[pallet::storage]
    #[pallet::getter(fn user_daily_reward)]
    pub(super) type UserDailyReward<T> =
        StorageMap<_, Blake2_128Concat, <T as frame_system::Config>::AccountId, Vec<BalanceOf<T>>>;

    /// total grade of machine
    #[pallet::storage]
    #[pallet::getter(fn total_machine_grade)]
    pub(super) type TotalMachineGrade<T> = StorageValue<_, u64>;

    // /// UserBondRecord
    #[pallet::storage]
    #[pallet::getter(fn user_bond_record)]
    pub(super) type UserBondRecord<T> = StorageMap<
        _,
        Blake2_128Concat,
        <T as frame_system::Config>::AccountId,
        Vec<
            MachineStakeInfo<
                <T as frame_system::Config>::AccountId,
                BalanceOf<T>,
                <T as frame_system::Config>::BlockNumber,
            >,
        >,
    >;

    /// User rewards payout time
    #[pallet::storage]
    #[pallet::getter(fn user_payout_era_index)]
    pub(super) type UserPayoutEraIndex<T> =
        StorageMap<_, Blake2_128Concat, <T as frame_system::Config>::AccountId, u32>;

    // /// release duration: 75% token will release in following 150 days
    // pub ProfitReleaseDuration get(fn profit_release_duration) config(): u64 = 150;
    #[pallet::type_value]
    pub(super) fn ProfitReleaseDurationDefault<T: Config>() -> u64 {
        150
    }

    #[pallet::storage]
    pub(super) type ProfitReleaseDuration<T: Config> =
        StorageValue<_, u64, ValueQuery, ProfitReleaseDurationDefault<T>>;

    /// Reward per year
    #[pallet::storage]
    #[pallet::getter(fn reward_per_year)]
    pub(super) type RewardPerYear<T> = StorageValue<_, BalanceOf<T>>;

    // 等于RewardPerYear * (era_duration / year_duration)
    #[pallet::storage]
    #[pallet::getter(fn eras_validator_reward)]
    pub(super) type ErasValidatorReward<T> =
        StorageMap<_, Blake2_128Concat, EraIndex, Option<BalanceOf<T>>>;

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        fn on_finalize(block_number: T::BlockNumber) {
            if (block_number.saturated_into::<u64>() + 1) / T::BlockPerEra::get() as u64 == 0 {
                Self::end_era()
            }
        }
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        // 将machine_id添加到绑定队列
        /// Bonding machine only remember caller-machine_id pair.
        /// OCW will check it and record machine info.
        #[pallet::weight(10000)]
        fn bond_machine(origin: OriginFor<T>, machine_id: MachineId) -> DispatchResultWithPostInfo {
            let caller = ensure_signed(origin)?;

            // 确保 BondingQueue 不包含该 machine_id
            ensure!(
                !BondingQueue::<T>::contains_key(&machine_id),
                Error::<T>::MachineInBondingQueue
            );

            ensure!(
                !BookingQueue::<T>::contains_key(&machine_id),
                Error::<T>::MachineInBookingQueue
            );

            ensure!(
                !BookedQueue::<T>::contains_key(&machine_id),
                Error::<T>::MachineInBookedQueue
            );

            // 该machine_id还未被绑定
            ensure!(
                !BondedMachine::<T>::contains_key(&machine_id),
                Error::<T>::MachineHasBonded
            );

            BondingQueue::<T>::insert(
                machine_id.clone(),
                BondingPair {
                    account_id: caller,
                    machine_id: machine_id,
                    request_count: 0,
                },
            );

            Ok(().into())
        }

        #[pallet::weight(10000)]
        fn add_bonded_token(
            origin: OriginFor<T>,
            machine_id: MachineId,
            bond_amount: BalanceOf<T>,
        ) -> DispatchResultWithPostInfo {
            let who = ensure_signed(origin)?;

            // 检查余额
            ensure!(
                <T as Config>::Currency::free_balance(&who) > bond_amount,
                Error::<T>::BalanceNotEnough
            );
            // 检查超过最小交易金额
            ensure!(
                bond_amount >= T::Currency::minimum_balance(),
                Error::<T>::InsufficientValue
            );

            // 检查用户已绑定了该机器
            let user_bonded_machine = UserBondedMachine::<T>::get(&who);
            if let Err(_) = user_bonded_machine.binary_search(&machine_id) {
                return Err(Error::<T>::MachineIdNotBonded.into());
            };

            let current_era = Self::current_era();
            let history_depth = Self::history_depth(); // TODO: add this
            let last_reward_era = current_era.saturating_sub(history_depth);

            let user_balance = T::Currency::free_balance(&who);
            let bond_amount = bond_amount.min(user_balance);

            Self::deposit_event(Event::AddBonded(
                who.clone(),
                machine_id.clone(),
                bond_amount,
            ));

            let item = StakingLedger {
                stash: who.clone(),
                total: bond_amount,
                active: bond_amount,
                unlocking: vec![],
                claimed_rewards: (last_reward_era..current_era).collect(),
            };

            Self::update_ledger(&who, &machine_id, &item);

            Ok(().into())
        }

        #[pallet::weight(10000)]
        fn bond_extra(
            origin: OriginFor<T>,
            machine_id: MachineId,
            max_additional: BalanceOf<T>,
        ) -> DispatchResultWithPostInfo {
            let who = ensure_signed(origin)?;

            let mut ledger = Self::ledger(&who, &machine_id).ok_or(Error::<T>::LedgerNotFound)?;
            let user_balance = T::Currency::free_balance(&who);
            if let Some(extra) = user_balance.checked_sub(&ledger.total) {
                let extra = extra.min(max_additional);
                ledger.total += extra;
                ledger.active += extra;

                ensure!(
                    ledger.active >= T::Currency::minimum_balance(),
                    Error::<T>::InsufficientValue
                );

                Self::deposit_event(Event::AddBonded(who.clone(), machine_id.clone(), extra));
                Self::update_ledger(&who, &machine_id, &ledger);
            }

            Ok(().into())
        }

        #[pallet::weight(10000)]
        fn reduce_bonded_token(
            origin: OriginFor<T>,
            machine_id: MachineId,
            amount: BalanceOf<T>,
        ) -> DispatchResultWithPostInfo {
            let who = ensure_signed(origin)?;

            let mut ledger = Self::ledger(&who, &machine_id).ok_or(Error::<T>::LedgerNotFound)?;

            ensure!(
                ledger.unlocking.len() < crate::MAX_UNLOCKING_CHUNKS,
                Error::<T>::NoMoreChunks
            );
            let mut value = amount.min(ledger.active);

            if !value.is_zero() {
                ledger.active -= value;

                if ledger.active < <T as Config>::Currency::minimum_balance() {
                    value += ledger.active;
                    ledger.active = Zero::zero();
                }

                let era = Self::current_era() + T::BondingDuration::get();
                ledger.unlocking.push(UnlockChunk { value, era });
                Self::update_ledger(&who, &machine_id, &ledger);
                Self::deposit_event(Event::RemoveBonded(who, machine_id, value));
            }

            Ok(().into())
        }

        #[pallet::weight(10000)]
        fn withdraw_unbonded(
            origin: OriginFor<T>,
            machine_id: MachineId,
        ) -> DispatchResultWithPostInfo {
            let who = ensure_signed(origin)?;
            let mut ledger = Self::ledger(&who, &machine_id).ok_or(Error::<T>::LedgerNotFound)?;

            let old_total = ledger.total;
            let current_era = Self::current_era();
            ledger = ledger.consolidate_unlock(current_era);
            if ledger.unlocking.is_empty() && ledger.active <= T::Currency::minimum_balance() {
                // TODO: 清除ledger相关存储
                T::Currency::remove_lock(crate::PALLET_LOCK_ID, &who);
            } else {
                Self::update_ledger(&who, &machine_id, &ledger);
            }

            if ledger.total < old_total {
                let value = old_total - ledger.total;
                Self::deposit_event(Event::Withdrawn(who, machine_id, value));
            }

            Ok(().into())
        }

        // // TODO: 重新实现这个函数
        // #[pallet::weight(10000)]
        // pub fn rm_bonded_machine(
        //     origin: OriginFor<T>,
        //     _machine_id: MachineId,
        // ) -> DispatchResultWithPostInfo {
        //     let _user = ensure_signed(origin)?;
        //     Ok(().into())
        // }

        // #[pallet::weight(10000)]
        // pub fn payout_rewards(origin: OriginFor<T>) -> DispatchResultWithPostInfo {
        //     let who = ensure_signed(origin)?;
        //     let current_era = Self::current_era();
        //     // Self::do_payout_stakers(who, era);

        //     // let reward_to_payout = UserReleasedReward::<T>::get(&user);
        //     // let _ = <T as Config>::Currency::deposit_into_existing(&user, reward_to_payout).ok();

        //     // <UserPayoutEraIndex<T>>::insert(user, current_era);
        //     Ok(().into())
        // }

        #[pallet::weight(0)]
        pub fn set_machine_price(
            origin: OriginFor<T>,
            machine_id: MachineId,
            machine_price: u64,
        ) -> DispatchResultWithPostInfo {
            let _ = ensure_root(origin);

            if !MachineDetail::<T>::contains_key(&machine_id) {
                MachineDetail::<T>::insert(
                    &machine_id,
                    MachineMeta {
                        machine_price: machine_price,
                        machine_grade: 0,
                        committee_confirm: vec![],
                    },
                );
                return Ok(().into());
            }

            let mut machine_detail = MachineDetail::<T>::get(&machine_id);
            machine_detail.machine_price = machine_price;

            MachineDetail::<T>::insert(&machine_id, machine_detail);
            Ok(().into())
        }

        #[pallet::weight(0)]
        pub fn donate_money(
            origin: OriginFor<T>,
            amount: BalanceOf<T>,
        ) -> DispatchResultWithPostInfo {
            let donor = ensure_signed(origin)?;

            <T as Config>::Currency::transfer(
                &donor,
                &Self::account_id(),
                amount,
                crate::AllowDeath,
            )
            .map_err(|_| DispatchError::Other("Can't make donation"))?;
            Self::deposit_event(Event::DonationReceived(donor, amount, Self::pot()));
            Ok(().into())
        }

        #[pallet::weight(0)]
        pub fn allocate(
            origin: OriginFor<T>,
            dest: T::AccountId,
            amount: BalanceOf<T>,
        ) -> DispatchResultWithPostInfo {
            let _ = ensure_root(origin)?;

            <T as Config>::Currency::transfer(
                &Self::account_id(),
                &dest,
                amount,
                crate::AllowDeath,
            )
            .map_err(|_| DispatchError::Other("Can't make allocation"))?;

            Self::deposit_event(Event::FundsAllocated(dest, amount, Self::pot()));
            Ok(().into())
        }
    }

    #[pallet::event]
    #[pallet::metadata(T::AccountId = "AccountId", BalanceOf<T> = "Balance")]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        BondMachine(T::AccountId, MachineId),
        AddBonded(T::AccountId, MachineId, BalanceOf<T>),
        RemoveBonded(T::AccountId, MachineId, BalanceOf<T>),
        DonationReceived(T::AccountId, BalanceOf<T>, BalanceOf<T>),
        FundsAllocated(T::AccountId, BalanceOf<T>, BalanceOf<T>),
        Withdrawn(T::AccountId, MachineId, BalanceOf<T>),
    }

    #[pallet::error]
    pub enum Error<T> {
        MachineIdNotBonded,
        MachineHasBonded,
        MachineInBondingQueue,
        MachineInBookingQueue,
        MachineInBookedQueue,
        TokenNotBonded,
        BondedNotEnough,
        HttpFetchingError,
        HttpDecodeError,
        BalanceNotEnough,
        NotMachineOwner,
        LedgerNotFound,
        NoMoreChunks,
        AlreadyAddedMachine,
        InsufficientValue,
        IndexOutOfRange,
        MachineURLEmpty,
        OffchainUnsignedTxError,
        InvalidEraToReward,
        AccountNotSame,
        NotInBookingList,
    }
}

impl<T: Config> Pallet<T> {
    // fn do_payout_stakers(who: T::AccountId, era: EraIndex) -> DispatchResult {
    //     let current_era = Self::current_era();
    //     ensure!(era <= current_era, Error::<T>::InvalidEraToReward);
    //     let history_depth = Self::history_depth();
    //     ensure!(
    //         era >= current_era.saturating_sub(history_depth),
    //         Error::<T>::InvalidEraToReward
    //     );

    //     let era_payout =
    //         <ErasValidatorReward<T>>::get(&era).ok_or_else(|| Error::<T>::InvalidEraToReward)?;

    //     Ok(())
    // }

    // TODO: 将ss58address转为public key
    // 参考：primitives/core/src/crypto.rs: impl Ss58Codec for AccountId32
    // from_ss58check_with_version
    pub fn wallet_match_account(who: T::AccountId, s: &Vec<u8>) -> bool {
        // const CHECKSUM_LEN: usize = 2;
        let mut data: [u8; 35] = [0; 35];
        let decoded = bs58::decode(s).into(&mut data);

        match decoded {
            Ok(length) => {
                if length != 35 {
                    return false;
                }
            }
            Err(_) => return false,
        }

        let (_prefix_len, _ident) = match data[0] {
            0..=63 => (1, data[0] as u16),
            64..=127 => {
                // let lower = (data[0] << 2) | (data[1] >> 6);
                // let upper = data[1] & 0b00111111;
                // (2, (lower as u16) | ((upper as u16) << 8))
                return false;
            }
            _ => return false,
        };

        let account_id32: [u8; 32] = data[1..33].try_into().unwrap();
        let wallet = T::AccountId::decode(&mut &account_id32[..]).unwrap_or_default();

        if who == wallet {
            return true;
        }
        return false;
    }

    fn end_era() {}

    // 增加随机性
    fn update_nonce() -> Vec<u8> {
        let nonce = RandNonce::<T>::get();
        let nonce: u64 = if nonce == u64::MAX {
            0
        } else {
            RandNonce::<T>::get() + 1
        };
        RandNonce::<T>::put(nonce);

        nonce.encode()
    }

    // 生成一个随机的u32
    pub fn random_num(max: u32) -> u32 {
        let subject = Self::update_nonce();
        let random_seed = T::RandomnessSource::random(&subject);
        let mut rng = <RandomNumberGenerator<BlakeTwo256>>::new(random_seed);
        rng.pick_u32(max)
    }

    // 更新用户的质押的ledger
    fn update_ledger(
        controller: &T::AccountId,
        machine_id: &MachineId,
        ledger: &StakingLedger<T::AccountId, BalanceOf<T>>,
    ) {
        T::Currency::set_lock(
            PALLET_LOCK_ID,
            &ledger.stash,
            ledger.total,
            WithdrawReasons::all(),
        );
        Ledger::<T>::insert(controller, machine_id, Some(ledger));
    }

    // 当前pallet的accountID
    pub fn account_id() -> T::AccountId {
        PALLET_ID.into_account()
    }

    // 当前pallet中的代币数量
    fn pot() -> BalanceOf<T> {
        <T as pallet::Config>::Currency::free_balance(&Self::account_id())
    }

    // 获取当前era index
    pub fn current_era() -> u32 {
        let current_block_height =
            <frame_system::Module<T>>::block_number().saturated_into::<u32>();
        return current_block_height / T::BlockPerEra::get();
    }

    pub fn block_per_era() -> u32 {
        T::BlockPerEra::get()
    }

    // TODO: 计算每个era中的用户总分数
    fn _user_machine_total_grades(user: T::AccountId) -> u64 {
        let user_machines = UserBondRecord::<T>::get(&user).unwrap();
        let current_era = Self::current_era();
        let mut user_grades = 0u64;
        for a_machine in &user_machines {
            // skip only staked one day's
            if current_era == a_machine.bond_era {
                continue;
            }
            let machine_grade = MachineDetail::<T>::get(&a_machine.machine_id).machine_grade;
            user_grades += machine_grade;
        }
        return user_grades;
    }

    // TODO: 计算用户奖励
    fn _update_user_reward(user: T::AccountId) {
        let release_daily: BalanceOf<T> = RewardPerYear::<T>::get().unwrap()
            * 100u64.saturated_into()
            / 36_525u64.saturated_into();
        let mut user_released_reward = UserReleasedReward::<T>::get(&user).unwrap();

        let current_era = Self::current_era();
        let daily_reward_index = current_era as usize / ProfitReleaseDuration::<T>::get() as usize;

        // 用户由于质押获得的奖励
        let daily_reward = release_daily
            * UserTotalMachineGrades::<T>::get(&user)
                .unwrap()
                .saturated_into()
            / TotalMachineGrade::<T>::get().unwrap().saturated_into();

        let locked_daily_reward = daily_reward * 75u64.saturated_into()
            / 100u64.saturated_into()
            / ProfitReleaseDuration::<T>::get().saturated_into();

        user_released_reward += daily_reward - locked_daily_reward;

        let mut user_daily_reward = UserDailyReward::<T>::get(&user).unwrap();

        user_released_reward += user_daily_reward[daily_reward_index];

        user_daily_reward[daily_reward_index] = daily_reward;

        UserReleasedReward::<T>::insert(&user, user_released_reward);
        UserDailyReward::<T>::insert(&user, user_daily_reward);
    }
}

impl<T: Config> LCOps for Pallet<T> {
    type MachineId = MachineId;
    type AccountId = T::AccountId;
    type BlockNumber = T::BlockNumber;

    fn bonding_queue_id() -> BTreeSet<Self::MachineId> {
        <BondingQueue<T> as IterableStorageMap<MachineId, BondingPair<T::AccountId>>>::iter()
            .map(|(machine_id, _)| machine_id)
            .collect::<BTreeSet<_>>()
    }

    fn booking_queue_id() -> BTreeSet<Self::MachineId> {
        <BookingQueue<T> as IterableStorageMap<MachineId, BookingItem<T::BlockNumber>>>::iter()
            .map(|(machine_id, _)| machine_id)
            .collect::<BTreeSet<_>>()
    }

    fn book_one_machine(_who: &T::AccountId, machine_id: MachineId) -> bool {
        let bonding_queue_id = Self::bonding_queue_id();
        if !bonding_queue_id.contains(&machine_id) {
            return false;
        }

        let booking_item = BookingItem {
            machine_id: machine_id.to_vec(),
            book_time: <frame_system::Module<T>>::block_number(),
        };

        BookingQueue::<T>::insert(&machine_id, booking_item.clone());
        BondingQueue::<T>::remove(&machine_id);
        true
    }

    fn booked_queue_id() -> BTreeSet<Self::MachineId> {
        <BookedQueue<T> as IterableStorageMap<MachineId, u64>>::iter()
            .map(|(machine_id, _)| machine_id)
            .collect::<BTreeSet<_>>()
    }

    fn bonded_machine_id() -> BTreeSet<Self::MachineId> {
        <BondedMachine<T> as IterableStorageMap<MachineId, ()>>::iter()
            .map(|(machine_id, _)| machine_id)
            .collect::<BTreeSet<_>>()
    }

    fn rm_booking_id(id: MachineId) {
        BookingQueue::<T>::remove(id);
    }

    fn add_booked_id(id: MachineId) {}

    fn confirm_machine_grade(who: T::AccountId, machine_id: MachineId, confirm: bool) {
        let mut machine_grade = OCWMachineGrades::<T>::get(&machine_id);

        machine_grade.committee_info.push(CommitteeInfo {
            account_id: who,
            block_height: <frame_system::Module<T>>::block_number(),
            confirm,
        });

        OCWMachineGrades::<T>::insert(&machine_id, machine_grade);
    }
}

impl<T: Config> CommOps for Pallet<T> {
    fn random_num(max: u32) -> u32 {
        Self::random_num(max)
    }
    fn current_era() -> u32 {
        Self::current_era()
    }
    fn block_per_era() -> u32 {
        Self::block_per_era()
    }
}

impl<T: Config> OPOps for Pallet<T> {
    type AccountId = T::AccountId;
    type BookingItem = BookingItem<T::BlockNumber>;
    type BondingPair = BondingPair<T::AccountId>;
    type ConfirmedMachine = ConfirmedMachine<T::AccountId, T::BlockNumber>;
    type MachineId = MachineId;

    fn get_bonding_pair(id: Self::MachineId) -> Self::BondingPair {
        BondingQueue::<T>::get(id)
    }

    fn add_machine_grades(id: Self::MachineId, machine_grade: Self::ConfirmedMachine) {
        OCWMachineGrades::<T>::insert(id, machine_grade)
    }

    fn add_machine_price(id: Self::MachineId, price: u64) {
        OCWMachinePrice::<T>::insert(id, price)
    }

    fn rm_bonding_id(id: Self::MachineId) {
        BondingQueue::<T>::remove(id);
    }

    fn add_booking_item(id: Self::MachineId, booking_item: Self::BookingItem) {
        BookingQueue::<T>::insert(id, booking_item);
    }

    fn wallet_match_account(who: T::AccountId, s: &Vec<u8>) -> bool {
        Self::wallet_match_account(who, s)
    }
}
