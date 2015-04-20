#freebase

import freebase
# freebase = freebase.HTTPMetawebSession('AIzaSyBhFNje4BnOGUmhErqbYWvyiRJ6Bnjxt0Y')

query = {
	"name" : 'lady of shalott'
}

# info = freebase.search(query='lady of shalott')

info = freebase.search(query)

for ii in info:
	print ii['id']