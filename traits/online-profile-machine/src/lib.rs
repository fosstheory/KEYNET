#![cfg_attr(not(feature = "std"), no_std)]

use sp_std::vec::Vec;

// lease-committee_ops
pub trait OCOps {
    type AccountId;
    type MachineId;
    type CommitteeUploadInfo;

    fn oc_booked_machine(id: Self::MachineId);
    fn oc_revert_booked_machine(id: Self::MachineId);

    fn oc_confirm_machine(who: Vec<Self::AccountId>, machine_info: Self::CommitteeUploadInfo) -> Result<(), ()>;
    fn oc_refuse_machine(machien_id: Self::MachineId, committee: Vec<Self::AccountId>) -> Result<(), ()>;
}

pub trait RTOps {
    type AccountId;
    type MachineId;
    type MachineStatus;
    type Balance;

    fn get_machine_price(machine_point: u64) -> Option<u64>;

    fn change_machine_status(
        machine_id: &Self::MachineId,
        new_status: Self::MachineStatus,
        renter: Option<Self::AccountId>,
        rent_duration: Option<u64>, // 不为None时，表示租用结束
    );

    fn change_machine_rent_fee(amount: Self::Balance, machine_id: Self::MachineId, is_burn: bool);
}

pub trait OPRPCQuery {
    type AccountId;
    type StashMachine;

    fn get_all_stash() -> Vec<Self::AccountId>;
    fn get_stash_machine(stash: Self::AccountId) -> Self::StashMachine;
}

pub trait ManageCommittee {
    type AccountId;
    type BalanceOf;
    type SlashReason;

    fn is_valid_committee(who: &Self::AccountId) -> bool;

    fn available_committee() -> Result<Vec<Self::AccountId>, ()>;
    fn change_used_stake(committee: Self::AccountId, amount: Self::BalanceOf, is_add: bool) -> Result<(), ()>;
    fn stake_per_order() -> Option<Self::BalanceOf>;
    fn add_reward(committee: Self::AccountId, reward: Self::BalanceOf);
    fn add_slash(
        who: Self::AccountId,
        amount: Self::BalanceOf,
        reward_to: Vec<Self::AccountId>,
        slash_reason: Self::SlashReason,
    );
}

pub trait DbcPrice {
    type Balance;

    fn get_dbc_amount_by_value(value: u64) -> Option<Self::Balance>;
}

pub trait MTOps {
    type AccountId;
    type MachineId;
    type FaultType;

    fn mt_machine_offline(
        reporter: Self::AccountId,
        committee: Vec<Self::AccountId>,
        machine_id: Self::MachineId,
        fault_type: Self::FaultType,
    );
}

pub trait GNOps {
    type AccountId;
    type Balance;

    fn slash_and_reward(
        slash_who: Vec<Self::AccountId>,
        each_slash: Self::Balance,
        reward_who: Vec<Self::AccountId>,
    ) -> Result<(), ()>;
}
