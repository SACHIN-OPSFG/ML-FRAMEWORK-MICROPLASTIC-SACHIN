import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()],server:{proxy:Object.fromEntries(['/health','/model-info','/predict','/history','/generate-report','/results','/samples','/sample-data'].map(path=>[path,'http://127.0.0.1:8000']))}});
