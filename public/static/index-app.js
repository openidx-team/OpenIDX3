const app = {
    data() {
        return {
            isVisible: false,
            videoExists: false
        };
    },
    methods: {
        toggleMenu() {
            this.isVisible = !this.isVisible;
        },
        checkVideoExistence() {
            const videoUrl = "../assets/background-video.mp4";
            fetch(videoUrl)
                .then(response => {
                    if (response.ok) {
                        this.videoExists = true;
                    }
                })
                .catch(error => {
                    this.videoExists = false;
                });
        }
    },
    delimiters: ['[[',']]']
}

Vue.createApp(app).mount('#app')
