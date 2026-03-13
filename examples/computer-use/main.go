package main

// This example demonstrates the full agent loop for Anthropic's computer use
// tool. It shows how to:
//
//  1. Wire up the provider, model, and computer use tool.
//  2. Parse incoming tool calls with ParseComputerUseInput.
//  3. Execute the action (stubbed here — in production you would drive a
//     real VM, container, or VNC session).
//  4. Construct the result with the builder functions
//     (NewComputerUseScreenshotResult, NewComputerUseErrorResult, etc.).
//  5. Append the result to the prompt and call Generate again.
//  6. Exit when Claude stops requesting tool calls.

import (
	"context"
	"fmt"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

// takeScreenshot is a stub that simulates capturing a screenshot.
// In a real implementation this would capture the virtual display
// and return raw PNG bytes.
func takeScreenshot() ([]byte, error) {
	// Placeholder: a minimal PNG header.
	return []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	}, nil
}

// responseToAssistantParts converts response content into
// MessagePart values suitable for an assistant-role message. This
// preserves text and tool-call parts so Claude sees its own output
// in the conversation history on the next turn.
func responseToAssistantParts(content fantasy.ResponseContent) []fantasy.MessagePart {
	var parts []fantasy.MessagePart
	for _, c := range content {
		switch c.GetType() {
		case fantasy.ContentTypeText:
			tc, ok := fantasy.AsContentType[fantasy.TextContent](c)
			if ok {
				parts = append(parts, fantasy.TextPart{Text: tc.Text})
			}
		case fantasy.ContentTypeToolCall:
			tc, ok := fantasy.AsContentType[fantasy.ToolCallContent](c)
			if ok {
				parts = append(parts, fantasy.ToolCallPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
					Input:      tc.Input,
				})
			}
		}
	}
	return parts
}

func main() {
	// Set up the Anthropic provider.
	provider, err := anthropic.New(anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not create provider:", err)
		os.Exit(1)
	}

	ctx := context.Background()

	// Pick the model.
	model, err := provider.LanguageModel(ctx, "claude-opus-4-6")
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not get language model:", err)
		os.Exit(1)
	}

	// Create a computer use tool. This tells Claude the dimensions
	// of the virtual display it will be controlling.
	computerTool := anthropic.NewComputerUseTool(anthropic.ComputerUseToolOptions{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		ToolVersion:     anthropic.ComputerUse20251124,
	})

	// Build the initial Call with a prompt and the computer use tool.
	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			fantasy.NewUserMessage("Take a screenshot of the desktop"),
		},
		Tools: []fantasy.Tool{computerTool},
	}

	// --- Agent loop ---
	// Keep generating until Claude stops requesting tool calls.
	const maxIterations = 10
	for i := range maxIterations {
		fmt.Printf("\n=== Iteration %d ===\n", i+1)

		resp, err := model.Generate(ctx, call)
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate failed:", err)
			os.Exit(1)
		}

		// Print any text Claude included.
		if text := resp.Content.Text(); text != "" {
			fmt.Println("Claude said:", text)
		}

		// Collect tool calls from the response.
		toolCalls := resp.Content.ToolCalls()
		if len(toolCalls) == 0 {
			fmt.Println("No more tool calls — done.")
			break
		}

		// Process each tool call and build result messages.
		var results []fantasy.MessagePart
		for _, tc := range toolCalls {
			fmt.Printf("Tool call: %s (id=%s)\n", tc.ToolName, tc.ToolCallID)

			action, err := anthropic.ParseComputerUseInput(tc.Input)
			if err != nil {
				fmt.Fprintln(os.Stderr, "could not parse tool input:", err)
				result := anthropic.NewComputerUseErrorResult(tc.ToolCallID, err)
				results = append(results, result)
				continue
			}

			// Execute the action (stubbed) and build the result.
			var result fantasy.ToolResultPart
			switch action.Action {
			case anthropic.ActionScreenshot:
				fmt.Println("  -> capturing screenshot")
				png, err := takeScreenshot()
				if err != nil {
					result = anthropic.NewComputerUseErrorResult(tc.ToolCallID, err)
				} else {
					result = anthropic.NewComputerUseScreenshotResult(tc.ToolCallID, png)
				}

			case anthropic.ActionLeftClick:
				fmt.Printf("  -> left-click at %v\n", action.Coordinate)
				// In production: execute the click, then screenshot.
				png, err := takeScreenshot()
				if err != nil {
					result = anthropic.NewComputerUseErrorResult(tc.ToolCallID, err)
				} else {
					result = anthropic.NewComputerUseScreenshotResult(tc.ToolCallID, png)
				}

			case anthropic.ActionType:
				fmt.Printf("  -> typing %q\n", action.Text)
				png, err := takeScreenshot()
				if err != nil {
					result = anthropic.NewComputerUseErrorResult(tc.ToolCallID, err)
				} else {
					result = anthropic.NewComputerUseScreenshotResult(tc.ToolCallID, png)
				}

			default:
				fmt.Printf("  -> handling %s\n", action.Action)
				png, err := takeScreenshot()
				if err != nil {
					result = anthropic.NewComputerUseErrorResult(tc.ToolCallID, err)
				} else {
					result = anthropic.NewComputerUseScreenshotResult(tc.ToolCallID, png)
				}
			}

			results = append(results, result)
		}

		// Append the assistant response and tool results to the
		// prompt for the next iteration.
		call.Prompt = append(call.Prompt,
			// Echo back the assistant's response so Claude sees
			// its own tool calls in context.
			fantasy.Message{
				Role:    fantasy.MessageRoleAssistant,
				Content: responseToAssistantParts(resp.Content),
			},
			// Provide the tool results.
			fantasy.Message{
				Role:    fantasy.MessageRoleTool,
				Content: results,
			},
		)
	}
}
