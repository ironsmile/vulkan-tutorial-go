package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
)

func init() {
	// This is needed to arrange that main() runs on main thread.
	// See documentation for functions that are only allowed to be called
	// from the main thread.
	runtime.LockOSThread()

	flag.BoolVar(&args.debug, "debug", false, "Enable Vulkan validation layers")
}

var args struct {
	debug bool
}

const (
	title = "Vulkan Tutorial: Validation layers"
)

func main() {
	app := &VulkanTutorialApp{
		width:                  1024,
		height:                 768,
		enableValidationLayers: args.debug,
		validationLayers: []string{
			"VK_LAYER_KHRONOS_validation\x00",
		},
	}
	if err := app.Run(); err != nil {
		log.Fatalf("ERROR: %s", err)
	}
}

// VulkanTutorialApp is the first program from vulkan-tutorial.com
type VulkanTutorialApp struct {
	width  int
	height int

	// validationLayers is the list of required device extensions needed by this
	// program when the -debug flag is set.
	validationLayers       []string
	enableValidationLayers bool

	window   *glfw.Window
	instance vk.Instance
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
	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())

	if err := vk.Init(); err != nil {
		return fmt.Errorf("failed to init Vulkan Go: %w", err)
	}

	if err := a.createInstance(); err != nil {
		return fmt.Errorf("createInstance: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) cleanupVulkan() {
	vk.DestroyInstance(a.instance, nil)
}

func (a *VulkanTutorialApp) createInstance() error {
	if a.enableValidationLayers && !a.checkValidationSupport() {
		return fmt.Errorf("validation layers requested but not available")
	}

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   title + "\x00",
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        "No Engine\x00",
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         vk.ApiVersion10,
	}

	glfwExtensions := glfw.GetCurrentContext().GetRequiredInstanceExtensions()
	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(glfwExtensions)),
		PpEnabledExtensionNames: glfwExtensions,
	}

	if a.enableValidationLayers {
		createInfo.EnabledLayerCount = uint32(len(a.validationLayers))
		createInfo.PpEnabledLayerNames = a.validationLayers
	}

	var instance vk.Instance
	if res := vk.CreateInstance(&createInfo, nil, &instance); res != vk.Success {
		return fmt.Errorf("failed to create Vulkan instance: %w", vk.Error(res))
	}

	a.instance = instance
	return nil
}

func (a *VulkanTutorialApp) checkValidationSupport() bool {
	var count uint32
	if vk.EnumerateInstanceLayerProperties(&count, nil) != vk.Success {
		return false
	}
	availableLayers := make([]vk.LayerProperties, count)

	if vk.EnumerateInstanceLayerProperties(&count, availableLayers) != vk.Success {
		return false
	}

	availableLayersStr := make([]string, 0, count)
	for _, layer := range availableLayers {
		layer.Deref()

		layerName := vk.ToString(layer.LayerName[:])
		availableLayersStr = append(availableLayersStr, layerName+"\x00")
	}

	for _, validationLayer := range a.validationLayers {
		layerFound := false

		for _, instanceLayer := range availableLayersStr {
			if validationLayer == instanceLayer {
				layerFound = true
				break
			}
		}

		if !layerFound {
			return false
		}
	}

	return true
}

func (a *VulkanTutorialApp) mainLoop() error {
	log.Printf("main loop!\n")

	for !a.window.ShouldClose() {
		glfw.PollEvents()
	}

	return nil
}
