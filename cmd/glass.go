package main

import (
	"glass/pkg/collectors"

	"github.com/rs/zerolog/log"
)

func main() {
	log.Info().Msg("Cloudways Looking Glass")
	collectors := collectors.RegisterCollectors()
	for _, collector := range collectors {
		//log.Info().Msgf("Collector %d", index)
		collector.Collector()
	}
}
