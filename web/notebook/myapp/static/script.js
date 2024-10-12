document.addEventListener('DOMContentLoaded', () => {
    // file input
    previewImage = (event) => {
        const input = event.target;
        const wrapper = document.getElementById("form-input-wrapper");
        const label = document.querySelector(".form-label");

        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function (e) {
                wrapper.style.backgroundImage = `url(${e.target.result})`;
                wrapper.style.backgroundPosition = 'center center';
                wrapper.style.backgroundRepeat = 'no-repeat';
                wrapper.style.backgroundSize = 'contain';
                
                label.classList.add("hidden");
            };
            reader.readAsDataURL(input.files[0]);
        }
    };

    document
        .getElementById("form-input-wrapper")
        .addEventListener("click", () => {
            document.getElementById("photo").click();
        });


    // frame
    const frames = document.querySelectorAll('.result-frame');
    const widthImg = document.querySelector('.result-img').width;
    const heigthImg = document.querySelector('.result-img').height;

    frames.forEach((frame) => {
        let width = frame.getAttribute('data-width');
        let height = frame.getAttribute('data-height');
        let x = frame.getAttribute('data-x');
        let y = frame.getAttribute('data-y');

        frame.style.width = `calc(${width * widthImg} * 1px)`;
        frame.style.height = `calc(${height * heigthImg} * 1px)`;
        frame.style.left = `calc((${x} * 1px) + (${widthImg / 2} * 1px))`;
        frame.style.top = `calc((${y} * 1px) + (${heigthImg / 2} * 1px))`;
    })

    // modal
    const modal = document.querySelector('.modal');
    const modalOverlay = document.querySelector('.modal-overlay');
    const body = document.body;
    const btnModalOpen = document.querySelector('.send-btn');
    const btnClose = document.querySelector('.modal-close');
    const btnCloseSecond = document.querySelector('.modal-btn');

    btnModalOpen.addEventListener('click', () => {
        body.classList.add('modal-opened');
        modal.classList.remove('hidden');
        modalOverlay.style.display="block";
    })

    const closeBtnFunc = () => {
        body.classList.remove('modal-opened');
        modal.classList.add('hidden');
        modalOverlay.style.display="none";
    }

    btnClose.addEventListener('click', () => {
        closeBtnFunc();
    })

    btnCloseSecond.addEventListener('click', () => {
        closeBtnFunc();
    })
})