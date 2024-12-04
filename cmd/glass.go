package main

import (
	"glass/pkg/resources"

	"github.com/rs/zerolog/log"
)

func main() {
	log.Info().Msg("Cloudways Looking Glass")
	resources.CPU()
}
