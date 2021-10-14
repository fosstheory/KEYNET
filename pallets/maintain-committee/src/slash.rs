use crate::{
    types::{MCSlashResult, ReportId, ReportResultType},
    Config, Pallet, PendingSlashReview, ReportResult, UnhandledReportResult,
};
use frame_support::IterableStorageMap;
use generic_func::ItemList;
use online_profile_machine::{GNOps, MTOps};
use sp_std::{vec, vec::Vec};

impl<T: Config> Pallet<T> {
    pub fn check_and_exec_slash() -> Result<(), ()> {
        let now = <frame_system::Module<T>>::block_number();
        let mut pending_unhandled_id = Self::unhandled_report_result();

        for slashed_report_id in pending_unhandled_id.clone() {
            let mut report_result_info = Self::report_result(&slashed_report_id);
            if now < report_result_info.slash_exec_time {
                continue
            }

            match report_result_info.report_result {
                ReportResultType::ReportSucceed => {
                    // slash unruly & inconsistent, reward to reward_committee & reporter
                    let mut slash_who = report_result_info.unruly_committee.clone();
                    for a_inconsistent in report_result_info.inconsistent_committee.clone() {
                        ItemList::add_item(&mut slash_who, a_inconsistent);
                    }

                    let mut reward_who = report_result_info.reward_committee.clone();
                    ItemList::add_item(&mut reward_who, report_result_info.reporter.clone());
                    let _ = T::SlashAndReward::slash_and_reward(
                        slash_who.clone(),
                        report_result_info.committee_stake,
                        reward_who,
                    );

                    let _ = Self::change_reporter_stake(
                        &report_result_info.reporter,
                        report_result_info.reporter_stake,
                        false,
                    );

                    let _ = Self::change_committee_stake(
                        report_result_info.reward_committee.clone(),
                        report_result_info.committee_stake,
                        false,
                    );

                    let _ = Self::change_committee_stake(slash_who, report_result_info.committee_stake, true);
                },
                ReportResultType::NoConsensus => {
                    // FIXME: check here
                    let _ = Self::change_committee_stake(
                        report_result_info.unruly_committee.clone(),
                        report_result_info.committee_stake,
                        true,
                    );

                    // only slash unruly_committee, no reward
                    let _ = T::SlashAndReward::slash_and_reward(
                        report_result_info.unruly_committee.clone(),
                        report_result_info.committee_stake,
                        vec![],
                    );
                },
                ReportResultType::ReportRefused => {
                    // slash reporter, slash committee
                    let _ = T::SlashAndReward::slash_and_reward(
                        vec![report_result_info.reporter.clone()],
                        report_result_info.reporter_stake,
                        report_result_info.reward_committee.clone(),
                    );

                    let mut slash_who = report_result_info.unruly_committee.clone();
                    for a_inconsistent in report_result_info.inconsistent_committee.clone() {
                        ItemList::add_item(&mut slash_who, a_inconsistent);
                    }

                    let _ = T::SlashAndReward::slash_and_reward(
                        slash_who.clone(),
                        report_result_info.committee_stake,
                        report_result_info.reward_committee.clone(),
                    );

                    let _ = Self::change_reporter_stake(
                        &report_result_info.reporter,
                        report_result_info.reporter_stake,
                        true,
                    );

                    let _ = Self::change_committee_stake(slash_who, report_result_info.committee_stake, true);
                    let _ = Self::change_committee_stake(
                        report_result_info.reward_committee.clone(),
                        report_result_info.committee_stake,
                        false,
                    );
                },
                ReportResultType::ReporterNotSubmitEncryptedInfo => {
                    // slash reporter, slash committee
                    let _ = T::SlashAndReward::slash_and_reward(
                        vec![report_result_info.reporter.clone()],
                        report_result_info.reporter_stake,
                        vec![],
                    );
                    let _ = T::SlashAndReward::slash_and_reward(
                        report_result_info.unruly_committee.clone(),
                        report_result_info.committee_stake,
                        vec![],
                    );

                    let _ = Self::change_reporter_stake(
                        &report_result_info.reporter,
                        report_result_info.reporter_stake,
                        true,
                    );
                    let _ = Self::change_committee_stake(
                        report_result_info.unruly_committee.clone(),
                        report_result_info.committee_stake,
                        true,
                    );
                    // TODO: ensure other committee has been unreserved
                },
            }

            ItemList::rm_item(&mut pending_unhandled_id, &slashed_report_id);
            report_result_info.slash_result = MCSlashResult::Executed;
            ReportResult::<T>::insert(slashed_report_id, report_result_info);
        }
        UnhandledReportResult::<T>::put(pending_unhandled_id);
        Ok(())
    }

    pub fn check_and_exec_pending_review() -> Result<(), ()> {
        let now = <frame_system::Module<T>>::block_number();
        let all_pending_review = <PendingSlashReview<T> as IterableStorageMap<ReportId, _>>::iter()
            .map(|(renter, _)| renter)
            .collect::<Vec<_>>();

        for a_pending_review in all_pending_review {
            let review_info = Self::pending_slash_review(a_pending_review);
            let report_result_info = Self::report_result(&a_pending_review);

            if review_info.expire_time < now {
                continue
            }

            let is_slashed_reporter = report_result_info.is_slashed_reporter(&review_info.applicant);
            let is_slashed_committee = report_result_info.is_slashed_committee(&review_info.applicant);
            let is_slashed_stash = report_result_info.is_slashed_stash(&review_info.applicant);

            if is_slashed_reporter {
                let _ = Self::change_reporter_stake(&review_info.applicant, review_info.staked_amount, true);
            } else if is_slashed_committee {
                let _ =
                    Self::change_committee_stake(vec![review_info.applicant.clone()], review_info.staked_amount, true);
            } else if is_slashed_stash {
                let _ = T::MTOps::mt_rm_stash_total_stake(review_info.applicant.clone(), review_info.staked_amount);
            }

            let _ = T::SlashAndReward::slash_and_reward(vec![review_info.applicant], review_info.staked_amount, vec![]);

            PendingSlashReview::<T>::remove(a_pending_review);
        }
        Ok(())
    }
}