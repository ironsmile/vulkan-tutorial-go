package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
)

func init() {
	// This is needed to arrange that main() runs on main thread.
	// See documentation for functions that are only allowed to be called
	// from the main thread.
	runtime.LockOSThread()
}

const (
	title = "Vulkan Tutorial"
)

func main() {
	app := &VulkanTutorialApp{
		width:  1024,
		height: 768,
	}
	if err := app.Run(); err != nil {
		log.Fatalf("ERROR: %s", err)
	}
}

// VulkanTutorialApp is the first program from vulkan-tutorial.com
type VulkanTutorialApp struct {
	width  int
	height int

	window *glfw.Window
}

// Run runs the vulkan program.
func (a *VulkanTutorialApp) Run() error {
	if err := a.initWindow(); err != nil {
		return fmt.Errorf("initWindow: %w", err)
	}
	defer a.cleanWindow()

	if err := a.initVulkan(); err != nil {
		return fmt.Errorf("initVulkan: %w", err)
	}
	defer a.cleanupVulkan()

	if err := a.mainLoop(); err != nil {
		return fmt.Errorf("mainLoop: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) initWindow() error {
	if err := glfw.Init(); err != nil {
		return fmt.Errorf("glfw.Init: %w", err)
	}

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(a.width, a.height, title, nil, nil)
	if err != nil {
		return fmt.Errorf("creating window: %w", err)
	}

	a.window = window
	return nil
}

func (a *VulkanTutorialApp) cleanWindow() {
	a.window.Destroy()
	glfw.Terminate()
}

func (a *VulkanTutorialApp) initVulkan() error {
	return nil
}

func (a *VulkanTutorialApp) cleanupVulkan() {

}

func (a *VulkanTutorialApp) mainLoop() error {
	log.Printf("main loop!\n")

	for !a.window.ShouldClose() {
		glfw.PollEvents()
	}

	return nil
}
