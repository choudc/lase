from playwright.sync_api import sync_playwright
import time
import sys

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Capture console logs
        page.on("console", lambda msg: print(f"BROWSER_LOG: {msg.type}: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"BROWSER_ERROR: {exc}"))
        
        try:
            print("Navigating to http://localhost:5173...")
            page.goto("http://localhost:5173")
            
            # Wait a bit for React to mount (or fail)
            time.sleep(5)
            
            # Take screenshot
            page.screenshot(path="debug_screenshot.png")
            print("Screenshot saved to debug_screenshot.png")
            
            # Print body content to see if anything rendered
            # content = page.content()
            # print("Page Content length:", len(content))
            
        except Exception as e:
            print(f"Navigation failed: {e}")
            
        browser.close()

if __name__ == "__main__":
    run()
