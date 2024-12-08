package collectors

import (
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v4/net"
)

type NetworkCollector struct{}

func (n *NetworkCollector) Collector() error {
	connections, _ := net.Connections("tcp")
	netstat, err := net.IOCounters(false)
	if err != nil {
		log.Err(err).Msg("Error getting network info")
	}
	for _, stat := range netstat {
		log.Info().Str("name", stat.Name).Uint64("bytes-sent", stat.BytesSent).Uint64("bytes-received", stat.BytesRecv).Uint64("packets-sent", stat.PacketsSent).Uint64("packets-received", stat.PacketsRecv).Msg("")
	}
	for _, connection := range connections {
		log.Info().Interface("connection", connection).Msg("")
	}
	return nil
}
