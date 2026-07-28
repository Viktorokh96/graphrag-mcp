// Package color provides color utilities: projecting document embeddings
// into perceptual colors via Oklab color space.
package color

import (
	"hash/crc32"
	"math"
)

// HSL represents a color in HSL space (H: 0-360, S: 0-1, L: 0-1).
type HSL struct {
	H, S, L float64
}

// HSLToRGB converts HSL to RGB (0-255).
func HSLToRGB(hsl HSL) (r, g, b uint8) {
	h, s, l := hsl.H, hsl.S, hsl.L
	c := (1 - math.Abs(2*l-1)) * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := l - c/2

	var r1, g1, b1 float64
	switch {
	case h < 60:
		r1, g1, b1 = c, x, 0
	case h < 120:
		r1, g1, b1 = x, c, 0
	case h < 180:
		r1, g1, b1 = 0, c, x
	case h < 240:
		r1, g1, b1 = 0, x, c
	case h < 300:
		r1, g1, b1 = x, 0, c
	default:
		r1, g1, b1 = c, 0, x
	}

	return uint8(math.Round((r1 + m) * 255)),
		uint8(math.Round((g1 + m) * 255)),
		uint8(math.Round((b1 + m) * 255))
}

// WordHue returns a stable hue (0-360) for a word using CRC32.
func WordHue(word string) float64 {
	h := crc32.ChecksumIEEE([]byte(word))
	return float64(h%360) + float64(h%256)/256.0
}

// CircularMean computes the circular mean of angles in degrees [0, 360).
func CircularMean(angles []float64) float64 {
	if len(angles) == 0 {
		return 0
	}
	var sumSin, sumCos float64
	for _, a := range angles {
		rad := a * math.Pi / 180
		sumSin += math.Sin(rad)
		sumCos += math.Cos(rad)
	}
	mean := math.Atan2(sumSin, sumCos) * 180 / math.Pi
	if mean < 0 {
		mean += 360
	}
	return mean
}

// DocColor returns an HSL color for a document, averaging the hues of its words.
func DocColor(text string, saturation, lightness float64) HSL {
	words := splitWords(text)
	if len(words) == 0 {
		return HSL{H: 0, S: 0, L: lightness}
	}
	hues := make([]float64, len(words))
	for i, w := range words {
		hues[i] = WordHue(w)
	}
	return HSL{
		H: CircularMean(hues),
		S: saturation,
		L: lightness,
	}
}

// splitWords tokenizes text into lowercase alphabetic words.
func splitWords(text string) []string {
	var words []string
	var cur []byte
	for i := 0; i < len(text); i++ {
		c := text[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			cur = append(cur, c|0x20) // lowercase
		} else if len(cur) > 0 {
			words = append(words, string(cur))
			cur = cur[:0]
		}
	}
	if len(cur) > 0 {
		words = append(words, string(cur))
	}
	return words
}
