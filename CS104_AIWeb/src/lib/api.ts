import axios from "axios";
import { store } from "./stores/store";

const api = axios.create({
    baseURL: "http://localhost:8000",
});
api.interceptors.request.use((config) => {
    store.uiStore.isBusy();
    return config;
});
api.interceptors.response.use(async (response) => {
    store.uiStore.isIdle();
    return response;
});
export default api;
