import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/roboto/300.css";
import "@fontsource/roboto/400.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/700.css";
import router from "./app/Routes.tsx";
import { RouterProvider } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StoreContext, store } from "./lib/stores/store.ts";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <StoreContext.Provider value={store}>
            <QueryClientProvider client={queryClient}>
                <ReactQueryDevtools />
                <RouterProvider router={router} />
            </QueryClientProvider>
        </StoreContext.Provider>
    </StrictMode>
);
