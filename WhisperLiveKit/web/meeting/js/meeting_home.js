document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.sidebar ul li a');
    const homeContent = document.getElementById('home-content');
    const meetingsContent = document.getElementById('meetings-content');        
    const viewMeetingsBtn = document.querySelector('[data-action="view-meetings"]');
    const btnNewMeeting = document.getElementById('btn-new-meeting');
    const btnNewMeeting2 = document.getElementById('btn-new-meeting2');
    const btnLiveView = document.getElementById('btn-live-view');
    const viewDetailBtns = document.querySelectorAll('.btn-view-detail');       

    function switchPage(pageName) {
        navLinks.forEach(function(link) {
            link.classList.remove('active');
            if (link.getAttribute('data-page') === pageName) {
                link.classList.add('active');
            }
        });

        if (pageName === 'home') {
            homeContent.style.display = 'block';
            meetingsContent.style.display = 'none';
        } else if (pageName === 'meetings') {
            homeContent.style.display = 'none';
            meetingsContent.style.display = 'block';
        }
    }

    navLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const pageName = this.getAttribute('data-page');
            if (pageName) {
                e.preventDefault();
                switchPage(pageName);
            }
        });
    });

    if (viewMeetingsBtn) {
        viewMeetingsBtn.addEventListener('click', function() {
            switchPage('meetings');
        });
    }

    if (btnNewMeeting) {
        btnNewMeeting.addEventListener('click', function() {
            window.location.href = 'meeting_transcription.html';
        });
    }

    if (btnNewMeeting2) {
        btnNewMeeting2.addEventListener('click', function() {
            window.location.href = 'meeting_transcription.html';
        });
    }

    if (btnLiveView) {
        btnLiveView.addEventListener('click', function() {
            window.location.href = 'meeting_transcription.html';
        });
    }

    viewDetailBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            window.location.href = 'meeting_detail.html';
        });
    });
});