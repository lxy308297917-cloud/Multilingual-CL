"""Fresh official-core JA run: audited prefix repair plus known allocator compatibility."""
import prompt_budget_guard
from allocator_compat_seeded_entry import main

if __name__=='__main__':
    prompt_budget_guard.install()
    main()
