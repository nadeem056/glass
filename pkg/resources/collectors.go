package resources

import (
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v4/cpu"
)

func CPU() {
	times, err := cpu.Times(false)
	if err != nil {
		log.Err(err)
	}
	log.Info().Interface("cpu", times).Msg("total")
}
