"""Seeded official-core training on separately recorded repaired JA data."""
import prompt_budget_guard
import seeded_cost_batch_entry


if __name__ == '__main__':
    prompt_budget_guard.install()
    seeded_cost_batch_entry.main()
