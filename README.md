# election_fluxes

Playground exploring electoral behaviours.
Application to the 2026 Paris mayor election, to see vote transferts from round 1 to round 2. Ecological regression. 

The election data for the Paris election were dowloaded from https://www.data.gouv.fr/datasets/resultats-des-elections-municipales-2026-1. 
These data include all French municipalities. Extracting the relevent data from them I produced two csv files for Paris alone, for the first and second round respectively. 

- The notebook select_data.ipynb selects the relevant Paris data from the csv files covering all French municipalities, escluding some useless test fields (and summing together the rresults of the three far-left parties in the first round)
- The notebook plots_paris.ipynb plots some of the data to look for correlations and groups of polling booths with similar behaviour. It is quite evident (for instant looking at the second round behaviour of first round Bournzel voters) that one should at least consider two separate group: those where in the first round Dati prevailed, and those won by Grégoire in the first round.


