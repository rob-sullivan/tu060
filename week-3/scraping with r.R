#https://pokemondb.net/pokedex/all
install.packages("rvest")
library(rvest)
webpage <- read_html("https://pokemondb.net/pokedex/all")
tableNodes <- html_nodes(webpage, 'table')
table <- html_table(tableNodes)
pokemonData <- table[[1]]
View(pokemonData)

#how to write data to a file
write.table(pokemonData, "pokemon.txt", row.names = FALSE)
saveRDS(pokemonData, "pokemon.rds")
write_rds(pokemonData, "pokemon_readr.rds")
name <- pokemonData$Name
type <- pokemonData$Type
save(list = c('name', 'type'), file= "pokemon.RData")
