# mlQCA Reproduction Fixture

`mlqca_vote_data.csv` is a mechanically converted test fixture from:

- Project: https://github.com/qinhuang-poliecon/mlQCA
- Source file: `data/voteData.rda`
- Source commit: `f5bce883ace6bdae394853539aca66d6576deba1`
- License: MIT

The R data frame was converted to CSV without analytical transformations.
The fixture contains 427 cases, the binary `vote` outcome, and 76 candidate
columns. `P_V181` is constant in the published data and is excluded from model
fitting because PyQCA rejects constant candidate conditions.
